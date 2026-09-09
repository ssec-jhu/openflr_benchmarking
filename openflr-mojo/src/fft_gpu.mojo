"""GPU radix-2 Cooley-Tukey FFT kernels.

One thread block handles one row (a length-`N` complex signal); the whole
row lives in shared memory for the duration of the transform. `N` must be a
power of two and small enough that `N/2` threads and `2*N` float32 words of
shared memory are within device limits (true up to N=2048 on all current
GPUs).
"""

from std.math import cos, sin, ceildiv
from std.gpu import thread_idx, block_idx
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from max.gpu.host import DeviceContext, DeviceBuffer

comptime PI: Float32 = 3.14159265358979323846



def ilog2_ct(n: Int) -> Int:
    var v = n
    var bits = 0
    while v > 1:
        v >>= 1
        bits += 1
    return bits


def bit_reverse_ct(x: Int, bits: Int) -> Int:
    var result = 0
    var v = x
    for _ in range(bits):
        result = (result << 1) | (v & 1)
        v >>= 1
    return result


# --- Shared-memory twiddle table -------------------------------------------
#
# Every butterfly in this file wants `exp(sign * 2*pi*i * k / L)` for some
# power-of-two stage length `L` dividing the transform length `N`, with
# `k < L/2`. That equals `exp(sign * 2*pi*i * (k * (N//L)) / N)`, so a
# single `N`-point table of `exp(-2*pi*i*j/N)` for `j` in `[0, N/2)` serves
# every stage of every kernel here, in both directions (the inverse
# direction is the same table conjugated). Callers pass `j = k * (N // L)`.
#
# The table is built once per block, by the block, into shared memory: one
# `cos`/`sin` pair per entry, `N/2` entries over `N/2` or `N/4` threads, so
# each thread evaluates one or two -- replacing the eleven it used to
# evaluate inline walking the stage schedule. See `fill_twiddle_table` for
# why paying shared memory for this is a different trade on an A100 than it
# was on the GPU optimizations.md measured.


def tw_entries(N: Int) -> Int:
    """Logical twiddle count for a length-`N` transform."""
    return N // 2


def tw_idx(j: Int) -> Int:
    """Physical slot for logical twiddle index `j`, skewed by `j >> 5` to
    break shared-memory bank conflicts.

    Every stage reads the table at a power-of-two stride: stage `s` of the
    warp head wants `j = k * (N >> (s+1))` with `k = lane & (2^s - 1)`, so
    at `N = 2048` stage 4 reads stride 64 and the stage-32 combine reads
    stride 32. Unskewed, those put 16 and 32 lanes of a warp on bank 0 --
    a 16- and 32-way conflict that would cost more than the `cos`/`sin`
    pair the table exists to replace. Adding `j >> 5` makes the stage-32
    read stride 33 words, whose banks are `33*lane % 32 == lane`, all
    distinct; every shallower stage's stride spreads the same way."""
    return j + (j >> 5)


def tw_alloc(N: Int, TW: Bool) -> Int:
    """Physical length to allocate for the table, skew included -- or 1 when
    the table is off, so a `TW=False` build keeps exactly the shared-memory
    footprint (and therefore the occupancy) it had before the table
    existed."""
    if not TW:
        return 1
    return tw_entries(N) + (tw_entries(N) >> 5) + 1


def fill_twiddle_table[N: Int, TW: Bool](
    t_re: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    t_im: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    tid: Int,
    nthreads: Int,
):
    """Fills the block's twiddle table and barriers. No-op when `TW` is off.

    Costs `N/2` complex entries of shared memory -- 8.25 KB at `N = 2048`,
    skew included. That is free on hardware whose occupancy is capped by
    threads or registers rather than shared memory: an A100 runs these
    1024-thread blocks two to an SM (the per-SM thread ceiling) at 31
    registers each, well inside both the 64 K register file and the 164 KB
    of shared memory, so the table displaces nothing. This is the reason
    it is worth trying a table again after the two attempts
    optimizations.md records as reverted -- both were measured on a GPU
    where these kernels were bandwidth-bound with transcendental math to
    spare, and where the table was paid for out of occupancy."""
    comptime if TW:
        comptime n_entries = tw_entries(N)
        var i = tid
        while i < n_entries:
            var angle = -2.0 * PI * Float32(i) / Float32(N)
            var p = tw_idx(i)
            t_re[p] = rebind[t_re.ElementType](cos(angle))
            t_im[p] = rebind[t_im.ElementType](sin(angle))
            i += nthreads
        barrier()


def twiddle[N: Int, L: Int, TW: Bool, invert: Bool](
    t_re: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    t_im: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    k: Int,
) -> SIMD[DType.float32, 2]:
    """The butterfly twiddle `exp(-2*pi*i*k/L)` as `(re, im)`, conjugated
    when `invert`, for a stage of length `L` inside a length-`N` transform.

    With the table on, `exp(-2*pi*i*k/L) == exp(-2*pi*i*(k*(N//L))/N)`, so
    the lookup is at logical index `k * (N // L)`. `L` divides `N` and both
    are powers of two, so that scale is a comptime shift.

    With the table off this evaluates `cos`/`sin` on the angle directly --
    the same expression, in the same Float32 association, that each of these
    call sites had inline before the table existed. A `TW=False` build is
    therefore bit-for-bit the old kernel, so the A/B measures the table and
    nothing else."""
    comptime if TW:
        var p = tw_idx(k * (N // L))
        var wr = rebind[Scalar[DType.float32]](t_re[p])
        var wi = rebind[Scalar[DType.float32]](t_im[p])
        comptime if invert:
            return SIMD[DType.float32, 2](wr, -wi)
        else:
            return SIMD[DType.float32, 2](wr, wi)
    else:
        comptime sign: Float32 = 1.0 if invert else -1.0
        var angle = sign * 2.0 * PI * Float32(k) / Float32(L)
        return SIMD[DType.float32, 2](cos(angle), sin(angle))


def fft_warp_head[N: Int, invert: Bool, TW: Bool](
    t_re: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    t_im: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    mut a_re: Scalar[DType.float32],
    mut a_im: Scalar[DType.float32],
    mut b_re: Scalar[DType.float32],
    mut b_im: Scalar[DType.float32],
    lane: Int,
):
    """The first six butterfly stages of a radix-2 DIT FFT, done entirely in
    registers via `shuffle_xor`.

    Stages `stage_half = 1, 2, 4, 8, 16, 32` only ever combine data within a
    single warp's own 64-element block of the array (see the index-algebra
    proof in optimizations.md: `i0 XOR i1 == stage_half` always, and every
    group of `size <= 64` elements is threaded by <= 32 consecutive tids,
    which is exactly one warp since `64 == 2*WARP_SIZE`). Doing them with
    register-resident values and warp shuffles instead of shared memory
    removes six `barrier()`s and the shared-memory traffic for six stages
    from every row FFT -- roughly half the stages at the production N=2048.

    On entry `a_re/a_im` and `b_re/b_im` hold the bit-reversed inputs for
    positions `64*warp + lane` and `64*warp + lane + 32`; on exit they hold
    that same pair of positions' values after stage 32. The caller stores
    them to shared memory and continues with `fft_lds_stages[..., 6, ...]`.
    Independent of `N`: it only ever touches one warp's 64 elements."""
    comptime sign: Float32 = 1.0 if invert else -1.0

    comptime for stage in range(5):
        comptime stage_half = 1 << stage
        comptime size = stage_half * 2
        var k = lane & (stage_half - 1)
        var angle_tw = twiddle[N, size, TW, invert](t_re, t_im, k)
        var wr = angle_tw[0]
        var wi = angle_tw[1]
        # Branchless role select: `(lane >> stage) & 1` is 0 for the "lo"
        # (i0) role, 1 for "hi" (i1). Computing both candidate results and
        # blending by a 0/1 float mask avoids an `if/else` here, which would
        # otherwise diverge within the warp (half the lanes idle while the
        # other half executes) -- on this hardware that divergence measured
        # far more expensive than the shared-memory + barrier() it was meant
        # to replace (see optimizations.md).
        var lo_f = Float32(1 - ((lane >> stage) & 1))
        var hi_f = 1.0 - lo_f

        var pa_re = shuffle_xor(a_re, UInt32(stage_half))
        var pa_im = shuffle_xor(a_im, UInt32(stage_half))
        var a_lo_tr = pa_re * wr - pa_im * wi
        var a_lo_ti = pa_re * wi + pa_im * wr
        var a_hi_tr = a_re * wr - a_im * wi
        var a_hi_ti = a_re * wi + a_im * wr
        var new_a_re_s = lo_f * (a_re + a_lo_tr) + hi_f * (pa_re - a_hi_tr)
        var new_a_im_s = lo_f * (a_im + a_lo_ti) + hi_f * (pa_im - a_hi_ti)
        a_re = new_a_re_s
        a_im = new_a_im_s

        var pb_re = shuffle_xor(b_re, UInt32(stage_half))
        var pb_im = shuffle_xor(b_im, UInt32(stage_half))
        var b_lo_tr = pb_re * wr - pb_im * wi
        var b_lo_ti = pb_re * wi + pb_im * wr
        var b_hi_tr = b_re * wr - b_im * wi
        var b_hi_ti = b_re * wi + b_im * wr
        var new_b_re_s = lo_f * (b_re + b_lo_tr) + hi_f * (pb_re - b_hi_tr)
        var new_b_im_s = lo_f * (b_im + b_lo_ti) + hi_f * (pb_im - b_hi_ti)
        b_re = new_b_re_s
        b_im = new_b_im_s

    # stage_half = 32 (size = 64): `a` and `b` are this same thread's own two
    # registers (idx_a's bit 5 is always 0, idx_b's always 1), so this
    # combine is a pure local computation -- no shuffle needed.
    var angle32_tw = twiddle[N, 64, TW, invert](t_re, t_im, lane)
    var wr32 = angle32_tw[0]
    var wi32 = angle32_tw[1]
    var tr32 = b_re * wr32 - b_im * wi32
    var ti32 = b_re * wi32 + b_im * wr32
    var new_a_re = a_re + tr32
    var new_a_im = a_im + ti32
    var new_b_re = a_re - tr32
    var new_b_im = a_im - ti32
    a_re = new_a_re
    a_im = new_a_im
    b_re = new_b_re
    b_im = new_b_im


def fft_lds_stages[
    N: Int, first_stage: Int, invert: Bool, TW: Bool, THREADS: Int, TN: Int = N
](
    t_re: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(TN, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    t_im: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(TN, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    s_re: TileTensor[
        DType.float32, type_of(row_major[N]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    s_im: TileTensor[
        DType.float32, type_of(row_major[N]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    tid: Int,
):
    """Butterfly stages `first_stage .. log2(N)-1` of a radix-2 DIT FFT over
    a length-`N` row already sitting in shared memory in natural order.

    Stages are run as fused radix-4 pairs: two chained radix-2 DIT stages
    rewritten algebraically as one radix-4 stage (see optimizations.md
    "Higher-radix FFT stages"), halving the `barrier()`/shared-memory round
    trips for these large-stride stages. Each fused stage has `N/4` tasks
    and each produces 4 outputs; the leftover odd stage (when
    `log2(N) - first_stage` is odd) is a plain radix-2 stage over `N/2`
    tasks.

    `THREADS` is the launching block's size, and both loops adapt to it:
    the fused stages have `N/4` tasks and the odd stage `N/2`, so each
    thread takes `ceil(tasks/THREADS)` of them. At the traditional `N/2` a
    fused stage leaves half the block idle at the guard while those warps
    still occupy the SM's thread budget and still have to reach every
    `barrier()`; smaller blocks give every thread work and fit more blocks
    per SM. Both loops degenerate to exactly the old code at
    `THREADS == N/2` (one iteration, `tt == tid`), so that configuration
    stays bit-identical.

    `first_stage` is 6 when the caller ran `fft_warp_head` first and 0 when
    it did not (`N < 64`, or no 32-wide warps). Leaves the finished spectrum
    in shared memory in natural order, with a trailing `barrier()`.

    `TN` is the length of the transform the twiddle *table* was built for,
    which is not always `N`: the real-input kernels run a length-`N/2`
    complex transform but share the enclosing length-`N` table (their
    unpack step needs its `N`-th roots). It defaults to `N` and only ever
    affects the `TW=True` lookup; with the table off `twiddle` ignores it
    entirely, so a `TW=False` build is unchanged by its existence."""
    comptime assert s_re.flat_rank == 1, "expected flat shared row"
    comptime assert s_im.flat_rank == 1, "expected flat shared row"
    comptime log2n = ilog2_ct(N)
    comptime sign: Float32 = 1.0 if invert else -1.0

    comptime num_pairs = (log2n - first_stage) // 2
    comptime for pair_i in range(num_pairs):
        comptime stage0 = first_stage + 2 * pair_i
        comptime m = 1 << (stage0 + 1)
        comptime half_m = m // 2
        comptime span = 2 * m
        comptime num_tasks = N // 4
        comptime for r in range(ceildiv(num_tasks, THREADS)):
            var tt = tid + r * THREADS
            if tt < num_tasks:
                var group = tt // half_m
                var k = tt % half_m
                var p0 = group * span + k
                var p1 = p0 + half_m
                var p2 = p0 + m
                var p3 = p2 + half_m

                var a0r = s_re[p0]
                var a0i = s_im[p0]
                var a1r = s_re[p1]
                var a1i = s_im[p1]
                var a2r = s_re[p2]
                var a2i = s_im[p2]
                var a3r = s_re[p3]
                var a3i = s_im[p3]

                var angle_a_tw = twiddle[TN, m, TW, invert](t_re, t_im, k)
                var war = angle_a_tw[0]
                var wai = angle_a_tw[1]
                var tar = a1r * war - a1i * wai
                var tai = a1r * wai + a1i * war
                var y0r = a0r + tar
                var y0i = a0i + tai
                var y1r = a0r - tar
                var y1i = a0i - tai
                var tbr = a3r * war - a3i * wai
                var tbi = a3r * wai + a3i * war
                var y2r = a2r + tbr
                var y2i = a2i + tbi
                var y3r = a2r - tbr
                var y3i = a2i - tbi

                var angle0_tw = twiddle[TN, span, TW, invert](t_re, t_im, k)
                var w0r = angle0_tw[0]
                var w0i = angle0_tw[1]
                var t0r = y2r * w0r - y2i * w0i
                var t0i = y2r * w0i + y2i * w0r
                var tcr = y3r * w0r - y3i * w0i
                var tci = y3r * w0i + y3i * w0r
                var t1r = -sign * tci
                var t1i = sign * tcr

                s_re[p0] = y0r + t0r
                s_im[p0] = y0i + t0i
                s_re[p2] = y0r - t0r
                s_im[p2] = y0i - t0i
                s_re[p1] = y1r + t1r
                s_im[p1] = y1i + t1i
                s_re[p3] = y1r - t1r
                s_im[p3] = y1i - t1i
        barrier()

    comptime if (log2n - first_stage) % 2 == 1:
        comptime stage = log2n - 1
        comptime size = 1 << (stage + 1)
        comptime stage_half = size // 2
        comptime assert stage_half % THREADS == 0, "THREADS must divide N/2"
        comptime for r in range(stage_half // THREADS):
            var tt = tid + r * THREADS
            var group = tt // stage_half
            var k = tt % stage_half
            var i0 = group * size + k
            var i1 = i0 + stage_half
            var angle_tw = twiddle[TN, size, TW, invert](t_re, t_im, k)
            var wr = angle_tw[0]
            var wi = angle_tw[1]
            var xr = s_re[i1]
            var xi = s_im[i1]
            var tr = xr * wr - xi * wi
            var ti = xr * wi + xi * wr
            var ur = s_re[i0]
            var ui = s_im[i0]
            s_re[i0] = ur + tr
            s_im[i0] = ui + ti
            s_re[i1] = ur - tr
            s_im[i1] = ui - ti
        barrier()


def fft_row_kernel[
    N: Int, LT: TensorLayout, invert: Bool, TW: Bool = False, THREADS: Int = N // 2
](
    re: TileTensor[DType.float32, LT, MutAnyOrigin],
    im: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """`re`/`im` have shape (num_rows, N); grid.x == num_rows,
    block.x == THREADS.

    `THREADS` defaults to the traditional `N // 2` -- one butterfly per
    thread in the warp head, two elements loaded and stored per thread.
    Halving it to `N // 4` gives each thread two of the head's 64-element
    blocks and four elements at the load and the store, which doubles the
    blocks resident per SM (these blocks are capped by the SM's thread
    budget, not by registers or shared memory) without changing the shared
    memory footprint. See optimizations.md sec. 5."""
    comptime assert re.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert im.flat_rank == 2, "expected (rows, N) tensor"
    comptime log2n = ilog2_ct(N)
    comptime half = N // 2
    comptime heads = half // THREADS       # 64-element head blocks per thread
    comptime elems = N // THREADS          # elements loaded/stored per thread
    comptime assert half % THREADS == 0, "THREADS must divide N/2"

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    comptime if N >= 64 and WARP_SIZE == 32:
        var warp = tid // 32
        var lane = tid % 32
        # One head block at a time rather than all `heads` at once, so the
        # live register set is the same as at THREADS == N/2; carrying four
        # elements simultaneously would raise register pressure and give
        # back the occupancy this is here to win. Striding by `num_warps`
        # rather than taking consecutive blocks also keeps the gather tight:
        # the head-block index lands in the *low* bits of the bit-reversed
        # source, so blocks `w` and `w + num_warps` read adjacent addresses.
        comptime num_warps = THREADS // 32
        comptime for h in range(heads):
            var blk = warp + h * num_warps
            var pos_a = 64 * blk + lane
            var pos_b = pos_a + 32
            var src_a = bit_reverse_ct(pos_a, log2n)
            var src_b = bit_reverse_ct(pos_b, log2n)
            var a_re = rebind[Scalar[DType.float32]](re[row, src_a])
            var a_im = rebind[Scalar[DType.float32]](im[row, src_a])
            var b_re = rebind[Scalar[DType.float32]](re[row, src_b])
            var b_im = rebind[Scalar[DType.float32]](im[row, src_b])

            fft_warp_head[N, invert, TW](t_re, t_im, a_re, a_im, b_re, b_im, lane)

            s_re[pos_a] = a_re
            s_im[pos_a] = a_im
            s_re[pos_b] = b_re
            s_im[pos_b] = b_im
        barrier()

        fft_lds_stages[N, 6, invert, TW, THREADS](t_re, t_im, s_re, s_im, tid)
    else:
        # Bit-reversed load: thread `tid` fills slots `tid + e*THREADS`.
        comptime for e in range(elems):
            var dst = tid + e * THREADS
            var src = bit_reverse_ct(dst, log2n)
            s_re[dst] = rebind[Scalar[DType.float32]](re[row, src])
            s_im[dst] = rebind[Scalar[DType.float32]](im[row, src])
        barrier()

        fft_lds_stages[N, 0, invert, TW, THREADS](t_re, t_im, s_re, s_im, tid)

    comptime if invert:
        comptime inv_n: Float32 = 1.0 / Float32(N)
        comptime for e in range(elems):
            var o = tid + e * THREADS
            re[row, o] = rebind[re.ElementType](s_re[o] * inv_n)
            im[row, o] = rebind[im.ElementType](s_im[o] * inv_n)
    else:
        comptime for e in range(elems):
            var o = tid + e * THREADS
            re[row, o] = rebind[re.ElementType](s_re[o])
            im[row, o] = rebind[im.ElementType](s_im[o])


def lds_swizzle(i: Int) -> Int:
    """Index swizzle for the shared-memory buffer that
    `fft_row_cmul_ifft_kernel` hands from its forward pass to its inverse
    pass.

    The inverse pass opens with a bit-reversed gather. Reading that gather
    straight out of shared memory is a 32-way bank conflict: for `N = 2048`
    (`log2n = 11`) the element warp `w` / lane `l` wants sits at
    `bit_reverse(64*w + l) == 64*rev5(l) + rev5(w)`, so every lane in the
    warp lands on bank `rev5(w)` -- the same one. Storing element `i` at
    `i ^ ((i >> 6) & 31)` instead leaves the 64-element block an element
    belongs to untouched (only bits 0-4 move) while making the gather's
    bank `rev5(w) ^ rev5(l)`, which is distinct for every lane. The
    permutation is an involution and a bijection within each 32-element
    aligned block, so it is a pure relabelling of shared memory: nothing
    about the arithmetic changes. For `N < 64` it is the identity."""
    return i ^ ((i >> 6) & 31)


def fft_row_cmul_ifft_kernel[
    N: Int, LT: TensorLayout, PLT: TensorLayout, TW: Bool = False
](
    re: TileTensor[DType.float32, LT, MutAnyOrigin],
    im: TileTensor[DType.float32, LT, MutAnyOrigin],
    p_re: TileTensor[DType.float32, PLT, MutAnyOrigin],
    p_im: TileTensor[DType.float32, PLT, MutAnyOrigin],
):
    """Forward FFT, elementwise complex multiply by `p`, and inverse FFT of
    a length-`N` row, all in one shared-memory residency.

    `re`/`im` and `p_re`/`p_im` have shape (num_rows, N); grid.x ==
    num_rows, block.x == N // 2. Computes `re/im <- ifft(fft(re/im) * p)`
    in place.

    This is the same arithmetic, in the same order, as
    `fft_row_kernel[N, ..., False]` -> complex multiply -> `fft_row_kernel[N,
    ..., True]`, but the intermediate spectrum never leaves shared memory:
    one kernel and one global read/write of the row instead of two kernels,
    two round trips through the row buffer, and (in the pipeline this was
    written for) the two transposes that used to sit between them. See
    optimizations.md sec. 3(a).

    The body is one `comptime for` over the two passes; the passes differ
    only in where the opening bit-reversed gather reads from (global for the
    forward pass, shared memory for the inverse) and what happens at the end
    (the multiply for the forward pass, the scaled global store for the
    inverse). Everything between is the identical stage schedule as
    `fft_row_kernel` -- `fft_warp_head` then `fft_lds_stages`.

    Unlike `fft_row_kernel` and `ifft_row_cmul_broadcast_kernel` this one is
    pinned to `block.x == N // 2` and takes no `THREADS` parameter: its
    inverse pass gathers out of the very shared buffer it then overwrites,
    so giving each thread more than one head block would mean holding all
    of them live across that barrier. Those extra registers work against
    the occupancy a smaller block exists to win, so the trade is not
    clearly positive here the way it is for the two kernels whose opening
    gather reads global memory."""
    comptime assert re.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert im.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert p_re.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert p_im.flat_rank == 2, "expected (rows, N) tensor"
    comptime log2n = ilog2_ct(N)
    comptime half = N // 2

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, N // 2)

    comptime for pass_i in range(2):
        comptime invert = pass_i == 1

        comptime if N >= 64 and WARP_SIZE == 32:
            var warp = tid // 32
            var lane = tid % 32
            var pos_a = 64 * warp + lane
            var pos_b = pos_a + 32
            var src_a = bit_reverse_ct(pos_a, log2n)
            var src_b = bit_reverse_ct(pos_b, log2n)

            var a_re: Scalar[DType.float32]
            var a_im: Scalar[DType.float32]
            var b_re: Scalar[DType.float32]
            var b_im: Scalar[DType.float32]
            comptime if pass_i == 0:
                a_re = rebind[Scalar[DType.float32]](re[row, src_a])
                a_im = rebind[Scalar[DType.float32]](im[row, src_a])
                b_re = rebind[Scalar[DType.float32]](re[row, src_b])
                b_im = rebind[Scalar[DType.float32]](im[row, src_b])
            else:
                # Same gather, but out of the shared-memory buffer the
                # forward pass left behind (in `lds_swizzle` order). Every
                # lane's read completes before the barrier, so the stores
                # below can overwrite the same buffer in place.
                var sa = lds_swizzle(src_a)
                var sb = lds_swizzle(src_b)
                a_re = rebind[Scalar[DType.float32]](s_re[sa])
                a_im = rebind[Scalar[DType.float32]](s_im[sa])
                b_re = rebind[Scalar[DType.float32]](s_re[sb])
                b_im = rebind[Scalar[DType.float32]](s_im[sb])
                barrier()

            fft_warp_head[N, invert, TW](t_re, t_im, a_re, a_im, b_re, b_im, lane)

            s_re[pos_a] = a_re
            s_im[pos_a] = a_im
            s_re[pos_b] = b_re
            s_im[pos_b] = b_im
            barrier()

            fft_lds_stages[N, 6, invert, TW, N // 2](t_re, t_im, s_re, s_im, tid)
        else:
            var src0 = bit_reverse_ct(tid, log2n)
            var src1 = bit_reverse_ct(tid + half, log2n)
            comptime if pass_i == 0:
                s_re[tid] = rebind[Scalar[DType.float32]](re[row, src0])
                s_im[tid] = rebind[Scalar[DType.float32]](im[row, src0])
                s_re[tid + half] = rebind[Scalar[DType.float32]](re[row, src1])
                s_im[tid + half] = rebind[Scalar[DType.float32]](im[row, src1])
            else:
                var v0r = rebind[Scalar[DType.float32]](s_re[lds_swizzle(src0)])
                var v0i = rebind[Scalar[DType.float32]](s_im[lds_swizzle(src0)])
                var v1r = rebind[Scalar[DType.float32]](s_re[lds_swizzle(src1)])
                var v1i = rebind[Scalar[DType.float32]](s_im[lds_swizzle(src1)])
                barrier()
                s_re[tid] = v0r
                s_im[tid] = v0i
                s_re[tid + half] = v1r
                s_im[tid + half] = v1i
            barrier()

            fft_lds_stages[N, 0, invert, TW, N // 2](t_re, t_im, s_re, s_im, tid)

        comptime if pass_i == 0:
            # The spectrum is in natural order in shared memory; multiply it
            # by `p` and hand it to the inverse pass in `lds_swizzle` order.
            # Reads are gathered into registers before the barrier so the
            # permuted stores can go back into the same buffer.
            var xr0 = rebind[Scalar[DType.float32]](s_re[tid])
            var xi0 = rebind[Scalar[DType.float32]](s_im[tid])
            var xr1 = rebind[Scalar[DType.float32]](s_re[tid + half])
            var xi1 = rebind[Scalar[DType.float32]](s_im[tid + half])
            var pr0 = rebind[Scalar[DType.float32]](p_re[row, tid])
            var pi0 = rebind[Scalar[DType.float32]](p_im[row, tid])
            var pr1 = rebind[Scalar[DType.float32]](p_re[row, tid + half])
            var pi1 = rebind[Scalar[DType.float32]](p_im[row, tid + half])
            barrier()
            var d0 = lds_swizzle(tid)
            var d1 = lds_swizzle(tid + half)
            s_re[d0] = xr0 * pr0 - xi0 * pi0
            s_im[d0] = xr0 * pi0 + xi0 * pr0
            s_re[d1] = xr1 * pr1 - xi1 * pi1
            s_im[d1] = xr1 * pi1 + xi1 * pr1
            barrier()
        else:
            comptime inv_n: Float32 = 1.0 / Float32(N)
            re[row, tid] = rebind[re.ElementType](s_re[tid] * inv_n)
            im[row, tid] = rebind[im.ElementType](s_im[tid] * inv_n)
            re[row, tid + half] = rebind[re.ElementType](s_re[tid + half] * inv_n)
            im[row, tid + half] = rebind[im.ElementType](s_im[tid + half] * inv_n)


def ifft_row_cmul_broadcast_kernel[
    N: Int, D: Int, W2: Int,
    BLT: TensorLayout, ALT: TensorLayout, OLT: TensorLayout,
    TW: Bool = False, THREADS: Int = N // 2,
](
    b_re: TileTensor[DType.float32, BLT, MutAnyOrigin],  # (D*W2, N)
    b_im: TileTensor[DType.float32, BLT, MutAnyOrigin],
    a_re: TileTensor[DType.float32, ALT, MutAnyOrigin],  # (W2, N), broadcast over depth
    a_im: TileTensor[DType.float32, ALT, MutAnyOrigin],
    dst_re: TileTensor[DType.float32, OLT, MutAnyOrigin],  # (D*W2, N)
    dst_im: TileTensor[DType.float32, OLT, MutAnyOrigin],
):
    """Broadcast-over-depth complex multiply fused into the *load* of an
    inverse length-`N` row FFT: computes `dst <- ifft(a * b)` reading each
    element of `a` and `b` exactly once, at the point the FFT's opening
    bit-reversed gather first needs it.

    `b` is a (D, W2, N) transposed spectrum flattened to (D*W2, N) rows; `a`
    is a single (W2, N) layer multiplied against every depth layer of `b`.
    grid.x == D*W2, block.x == THREADS (see `fft_row_kernel` for what
    `THREADS` trades).

    This replaces the `transpose_kernel_cmul_broadcast -> fft_row_kernel(inv)`
    pair the back-projection chain used to run. That pair existed only
    because the two operands were in the (D, H, W/2+1) layout and the column
    FFT needs (D, W/2+1, H); with both operands precomputed transposed
    (`rfft2_batched_gpu_t` for the PSF, `rfft2_batched_gpu_div_t` for the
    error image) the transpose has nothing left to do, and what remains --
    an elementwise multiply -- costs nothing folded into a load the kernel
    was already performing. The product buffer never reaches global memory,
    so the pair's 2066 MB becomes ~1394 MB in one dispatch. See
    optimizations.md sec. 3(b).

    **The block-index mapping matters as much as the fusion.** The broadcast
    row `a[w, :]` is shared by all `D` blocks with the same `w`. Mapping
    `row = block_idx.x` over the (D, W2, N) layout would make `w` the fast
    index, scheduling those `D` blocks `W2` apart, and the 16.8 MB operand
    would be swept `D` times -- reintroducing exactly the redundant traffic
    this kernel exists to remove (see optimizations.md sec. 4(d) for the
    measurement that diagnosed it in the kernel this replaces). Making `d`
    the fast index instead keeps the 16 KB shared row hot in cache across
    all `D` blocks that need it, and needs no layout change.

    The arithmetic is bit-identical to the two-kernel version: the same
    products in the same order, then the same stage schedule as
    `fft_row_kernel[N, ..., True]` -- the only difference is that the
    products are consumed from registers instead of from a round trip
    through global memory."""
    comptime assert b_re.flat_rank == 2, "expected (D*W2, N) tensor"
    comptime assert a_re.flat_rank == 2, "expected (W2, N) tensor"
    comptime assert dst_re.flat_rank == 2, "expected (D*W2, N) tensor"
    comptime log2n = ilog2_ct(N)
    comptime half = N // 2
    comptime heads = half // THREADS
    comptime elems = N // THREADS
    comptime assert half % THREADS == 0, "THREADS must divide N/2"

    # `d` fastest, `w` slowest -- see the docstring.
    var d = block_idx.x % D
    var w = block_idx.x // D
    var brow = d * W2 + w
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    comptime if N >= 64 and WARP_SIZE == 32:
        var warp = tid // 32
        var lane = tid % 32
        # One head block at a time; see `fft_row_kernel` for why this is a
        # sequential loop rather than four elements held at once, and why
        # the stride is `num_warps`.
        comptime num_warps = THREADS // 32
        comptime for h in range(heads):
            var blk = warp + h * num_warps
            var pos_a = 64 * blk + lane
            var pos_b = pos_a + 32
            var src_a = bit_reverse_ct(pos_a, log2n)
            var src_b = bit_reverse_ct(pos_b, log2n)

            var xar = rebind[Scalar[DType.float32]](a_re[w, src_a])
            var xai = rebind[Scalar[DType.float32]](a_im[w, src_a])
            var xbr = rebind[Scalar[DType.float32]](b_re[brow, src_a])
            var xbi = rebind[Scalar[DType.float32]](b_im[brow, src_a])
            var yar = rebind[Scalar[DType.float32]](a_re[w, src_b])
            var yai = rebind[Scalar[DType.float32]](a_im[w, src_b])
            var ybr = rebind[Scalar[DType.float32]](b_re[brow, src_b])
            var ybi = rebind[Scalar[DType.float32]](b_im[brow, src_b])

            var a_re_v = xar * xbr - xai * xbi
            var a_im_v = xar * xbi + xai * xbr
            var b_re_v = yar * ybr - yai * ybi
            var b_im_v = yar * ybi + yai * ybr

            fft_warp_head[N, True, TW](t_re, t_im, a_re_v, a_im_v, b_re_v, b_im_v, lane)

            s_re[pos_a] = a_re_v
            s_im[pos_a] = a_im_v
            s_re[pos_b] = b_re_v
            s_im[pos_b] = b_im_v
        barrier()

        fft_lds_stages[N, 6, True, TW, THREADS](t_re, t_im, s_re, s_im, tid)
    else:
        comptime for e in range(elems):
            var dst = tid + e * THREADS
            var src = bit_reverse_ct(dst, log2n)
            var xar = rebind[Scalar[DType.float32]](a_re[w, src])
            var xai = rebind[Scalar[DType.float32]](a_im[w, src])
            var xbr = rebind[Scalar[DType.float32]](b_re[brow, src])
            var xbi = rebind[Scalar[DType.float32]](b_im[brow, src])
            s_re[dst] = xar * xbr - xai * xbi
            s_im[dst] = xar * xbi + xai * xbr
        barrier()

        fft_lds_stages[N, 0, True, TW, THREADS](t_re, t_im, s_re, s_im, tid)

    comptime inv_n: Float32 = 1.0 / Float32(N)
    comptime for e in range(elems):
        var o = tid + e * THREADS
        dst_re[brow, o] = rebind[dst_re.ElementType](s_re[o] * inv_n)
        dst_im[brow, o] = rebind[dst_im.ElementType](s_im[o] * inv_n)


def rfft_row_kernel[
    N: Int, LT: TensorLayout, OutLT: TensorLayout, TW: Bool = False,
    THREADS: Int = N // 4,
](
    x: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    out_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Real-input FFT of each row of length `N` (real) into the
    non-redundant half spectrum of length `N/2+1`, via a single length-`N/2`
    complex FFT per row (mirrors `rfft_1d` in `fft_cpu`). `x` has shape
    (num_rows, N); `out_re`/`out_im` have shape (num_rows, N/2+1);
    grid.x == num_rows, block.x == THREADS.

    `THREADS` defaults to `N // 4`, which for the *internal* length-`N/2`
    complex transform is the traditional one-butterfly-per-thread `half/2`
    -- the same configuration `fft_row_kernel` starts from, one level down.
    Halving it to `N // 8` is exactly the `t4` trade: two of the warp
    head's 64-element blocks per thread and four unpack elements per
    thread, at unchanged shared memory, for twice the blocks resident per
    SM. See optimizations.md sec. 5 and the W-axis item in START HERE.

    The head schedule and the radix-4 stage schedule are `fft_warp_head`
    and `fft_lds_stages`, the same two the complex kernels use -- driven
    at the internal length `N/2` while still reading the enclosing
    length-`N` twiddle table (the `TN` parameter), because the unpack below
    needs that table's `N`-th roots."""
    comptime assert x.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert out_re.flat_rank == 2, "expected (rows, N/2+1) tensor"
    comptime half = N // 2
    comptime log2h = ilog2_ct(half)
    comptime heads = (half // 2) // THREADS  # 64-element head blocks per thread
    comptime elems = half // THREADS         # complex elements per thread
    comptime assert (half // 2) % THREADS == 0, "THREADS must divide N/4"

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    comptime if half >= 64 and WARP_SIZE == 32:
        # Same warp-shuffle fast path as `fft_row_kernel` (see its comment
        # for the index-algebra proof and for why the head blocks are taken
        # one at a time, strided by `num_warps`), applied to this kernel's
        # internal length-`half` complex FFT.
        var warp = tid // 32
        var lane = tid % 32
        comptime num_warps = THREADS // 32
        comptime for h in range(heads):
            var blk = warp + h * num_warps
            var pos_a = 64 * blk + lane
            var pos_b = pos_a + 32
            var src_a = bit_reverse_ct(pos_a, log2h)
            var src_b = bit_reverse_ct(pos_b, log2h)
            # Pack: z[n] = x[2n] + j*x[2n+1].
            var a_re = rebind[Scalar[DType.float32]](x[row, 2 * src_a])
            var a_im = rebind[Scalar[DType.float32]](x[row, 2 * src_a + 1])
            var b_re = rebind[Scalar[DType.float32]](x[row, 2 * src_b])
            var b_im = rebind[Scalar[DType.float32]](x[row, 2 * src_b + 1])

            fft_warp_head[N, False, TW](t_re, t_im, a_re, a_im, b_re, b_im, lane)

            s_re[pos_a] = a_re
            s_im[pos_a] = a_im
            s_re[pos_b] = b_re
            s_im[pos_b] = b_im
        barrier()

        fft_lds_stages[half, 6, False, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)
    else:
        # Pack: z[n] = x[2n] + j*x[2n+1], bit-reversed load into shared memory.
        comptime for e in range(elems):
            var dst = tid + e * THREADS
            var src = bit_reverse_ct(dst, log2h)
            s_re[dst] = rebind[Scalar[DType.float32]](x[row, 2 * src])
            s_im[dst] = rebind[Scalar[DType.float32]](x[row, 2 * src + 1])
        barrier()

        fft_lds_stages[half, 0, False, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)

    # Unpack Z (length `half`, in shared memory) into the length-N half
    # spectrum X[0 .. half] via the even/odd DFT symmetry.
    for i in range(elems):
        var k = tid + i * THREADS
        var km = (half - k) % half
        var zr = s_re[k]
        var zi = s_im[k]
        var zr_km = s_re[km]
        var zi_km = s_im[km]
        var xe_re = (zr + zr_km) * 0.5
        var xe_im = (zi - zi_km) * 0.5
        var xo_re = (zi + zi_km) * 0.5
        var xo_im = -(zr - zr_km) * 0.5
        var angle_tw = twiddle[N, N, TW, False](t_re, t_im, k)
        var wr = angle_tw[0]
        var wi = angle_tw[1]
        out_re[row, k] = rebind[out_re.ElementType](xe_re + (wr * xo_re - wi * xo_im))
        out_im[row, k] = rebind[out_im.ElementType](xe_im + (wr * xo_im + wi * xo_re))

    if tid == 0:
        out_re[row, half] = rebind[out_re.ElementType](s_re[0] - s_im[0])
        out_im[row, half] = rebind[out_im.ElementType](Scalar[DType.float32](0.0))


def rfft_row_kernel_div[
    N: Int, LT: TensorLayout, OutLT: TensorLayout, TW: Bool = False,
    THREADS: Int = N // 4,
](
    a: TileTensor[DType.float32, LT, MutAnyOrigin],
    b: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    out_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Fusion of a separate elementwise-divide kernel into `rfft_row_kernel`'s load
    stage: computes `rfft2(a / b)` directly, dividing `a` by `b` at the
    point each element is first read into shared memory instead of
    materializing `a / b` through a separate kernel launch and a full round
    trip through global memory first. Otherwise identical to
    `rfft_row_kernel` (kept intact and unchanged for other call sites, e.g.
    PSF prep, that don't need the division), `THREADS` included. `a`/`b`
    have shape (num_rows, N); `out_re`/`out_im` have shape (num_rows,
    N/2+1); grid.x == num_rows, block.x == THREADS."""
    comptime assert a.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert out_re.flat_rank == 2, "expected (rows, N/2+1) tensor"
    comptime half = N // 2
    comptime log2h = ilog2_ct(half)
    comptime heads = (half // 2) // THREADS
    comptime elems = half // THREADS
    comptime assert (half // 2) % THREADS == 0, "THREADS must divide N/4"

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    comptime if half >= 64 and WARP_SIZE == 32:
        # Same warp-shuffle fast path as `fft_row_kernel` (see its comment
        # for the index-algebra proof), applied to this kernel's internal
        # length-`half` complex FFT of `a / b`.
        var warp = tid // 32
        var lane = tid % 32
        comptime num_warps = THREADS // 32
        comptime for h in range(heads):
            var blk = warp + h * num_warps
            var pos_a_idx = 64 * blk + lane
            var pos_b_idx = pos_a_idx + 32
            var src_a = bit_reverse_ct(pos_a_idx, log2h)
            var src_b = bit_reverse_ct(pos_b_idx, log2h)
            var a_re = rebind[Scalar[DType.float32]](a[row, 2 * src_a]) / rebind[Scalar[DType.float32]](b[row, 2 * src_a])
            var a_im = rebind[Scalar[DType.float32]](a[row, 2 * src_a + 1]) / rebind[Scalar[DType.float32]](b[row, 2 * src_a + 1])
            var b_re = rebind[Scalar[DType.float32]](a[row, 2 * src_b]) / rebind[Scalar[DType.float32]](b[row, 2 * src_b])
            var b_im = rebind[Scalar[DType.float32]](a[row, 2 * src_b + 1]) / rebind[Scalar[DType.float32]](b[row, 2 * src_b + 1])

            fft_warp_head[N, False, TW](t_re, t_im, a_re, a_im, b_re, b_im, lane)

            s_re[pos_a_idx] = a_re
            s_im[pos_a_idx] = a_im
            s_re[pos_b_idx] = b_re
            s_im[pos_b_idx] = b_im
        barrier()

        fft_lds_stages[half, 6, False, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)
    else:
        # Pack: z[n] = x[2n] + j*x[2n+1] where x = a / b, bit-reversed load into
        # shared memory.
        comptime for e in range(elems):
            var dst = tid + e * THREADS
            var src = bit_reverse_ct(dst, log2h)
            s_re[dst] = rebind[Scalar[DType.float32]](a[row, 2 * src]) / rebind[Scalar[DType.float32]](b[row, 2 * src])
            s_im[dst] = rebind[Scalar[DType.float32]](a[row, 2 * src + 1]) / rebind[Scalar[DType.float32]](b[row, 2 * src + 1])
        barrier()

        fft_lds_stages[half, 0, False, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)

    # Unpack Z (length `half`, in shared memory) into the length-N half
    # spectrum X[0 .. half] via the even/odd DFT symmetry.
    for i in range(elems):
        var k = tid + i * THREADS
        var km = (half - k) % half
        var zr = s_re[k]
        var zi = s_im[k]
        var zr_km = s_re[km]
        var zi_km = s_im[km]
        var xe_re = (zr + zr_km) * 0.5
        var xe_im = (zi - zi_km) * 0.5
        var xo_re = (zi + zi_km) * 0.5
        var xo_im = -(zr - zr_km) * 0.5
        var angle_tw = twiddle[N, N, TW, False](t_re, t_im, k)
        var wr = angle_tw[0]
        var wi = angle_tw[1]
        out_re[row, k] = rebind[out_re.ElementType](xe_re + (wr * xo_re - wi * xo_im))
        out_im[row, k] = rebind[out_im.ElementType](xe_im + (wr * xo_im + wi * xo_re))

    if tid == 0:
        out_re[row, half] = rebind[out_re.ElementType](s_re[0] - s_im[0])
        out_im[row, half] = rebind[out_im.ElementType](Scalar[DType.float32](0.0))


def irfft_row_into_lds[
    N: Int, TW: Bool, InLT: TensorLayout, THREADS: Int = N // 4
](
    t_re: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    t_im: TileTensor[
        DType.float32, type_of(row_major[tw_alloc(N, TW)]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    in_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    in_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    s_re: TileTensor[
        DType.float32, type_of(row_major[N // 2]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    s_im: TileTensor[
        DType.float32, type_of(row_major[N // 2]()), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    row: Int,
    tid: Int,
):
    """Body of the row-wise real inverse FFT, shared by every kernel that
    ends in one: reads row `row` of a (num_rows, N/2+1) non-redundant complex
    half spectrum and leaves the length-`N` real result in the shared arrays
    `s_re`/`s_im`, interleaved as `out[2*n] = s_re[n]`, `out[2*n+1] =
    s_im[n]` and still unscaled by `1/(N/2)`.

    Everything after this point is a store, and the store is where the
    callers differ -- `irfft_row_kernel` writes the row out as-is, while
    `irfft_row_mul_kernel` multiplies it into a second operand and (for v1)
    rotates the destination index.

    `THREADS` is the launching block's size; see `rfft_row_kernel` for what
    the two settings mean. The reconstruct-and-head step takes
    `(N/4)/THREADS` of the warp head's 64-element blocks per thread and the
    stage schedule is `fft_lds_stages` driven at the internal length `N/2`
    against the enclosing length-`N` twiddle table."""
    comptime assert in_re.flat_rank == 2, "expected (rows, N/2+1) tensor"
    comptime half = N // 2
    comptime log2h = ilog2_ct(half)
    comptime heads = (half // 2) // THREADS
    comptime elems = half // THREADS
    comptime assert (half // 2) % THREADS == 0, "THREADS must divide N/4"

    comptime if half >= 64 and WARP_SIZE == 32:
        # Same warp-shuffle fast path as `fft_row_kernel` (see its comment
        # for the index-algebra proof). The "reconstruct" step below is a
        # scatter (source index `k`, destination `bit_reverse(k)`) rather
        # than a gather, but bit-reversal is self-inverse, so the value that
        # ends up owning final position `pos` is simply the reconstruction
        # of `k = bit_reverse(pos)` -- the same gather shape as the other
        # kernels' fast paths.
        var warp = tid // 32
        var lane = tid % 32
        comptime num_warps = THREADS // 32
        comptime for h in range(heads):
            var blk = warp + h * num_warps
            var pos_a = 64 * blk + lane
            var pos_b = pos_a + 32

            var ka = bit_reverse_ct(pos_a, log2h)
            var idxa = half - ka
            var aar = rebind[Scalar[DType.float32]](in_re[row, ka])
            var aai = rebind[Scalar[DType.float32]](in_im[row, ka])
            var abr = rebind[Scalar[DType.float32]](in_re[row, idxa])
            var abi = -rebind[Scalar[DType.float32]](in_im[row, idxa])
            var aer = (aar + abr) * 0.5
            var aei = (aai + abi) * 0.5
            var adr = (aar - abr) * 0.5
            var adi = (aai - abi) * 0.5
            var aangle_tw = twiddle[N, N, TW, True](t_re, t_im, ka)
            var awr = aangle_tw[0]
            var awi = aangle_tw[1]
            var aor = adr * awr - adi * awi
            var aoi = adr * awi + adi * awr
            var a_re = aer - aoi
            var a_im = aei + aor

            var kb = bit_reverse_ct(pos_b, log2h)
            var idxb = half - kb
            var bar = rebind[Scalar[DType.float32]](in_re[row, kb])
            var bai = rebind[Scalar[DType.float32]](in_im[row, kb])
            var bbr = rebind[Scalar[DType.float32]](in_re[row, idxb])
            var bbi = -rebind[Scalar[DType.float32]](in_im[row, idxb])
            var ber = (bar + bbr) * 0.5
            var bei = (bai + bbi) * 0.5
            var bdr = (bar - bbr) * 0.5
            var bdi = (bai - bbi) * 0.5
            var bangle_tw = twiddle[N, N, TW, True](t_re, t_im, kb)
            var bwr = bangle_tw[0]
            var bwi = bangle_tw[1]
            var bor = bdr * bwr - bdi * bwi
            var boi = bdr * bwi + bdi * bwr
            var b_re = ber - boi
            var b_im = bei + bor

            fft_warp_head[N, True, TW](t_re, t_im, a_re, a_im, b_re, b_im, lane)

            s_re[pos_a] = a_re
            s_im[pos_a] = a_im
            s_re[pos_b] = b_re
            s_im[pos_b] = b_im
        barrier()

        fft_lds_stages[half, 6, True, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)
    else:
        # Reconstruct Z = Xe + j*Xo (length `half`) from the half spectrum,
        # writing it bit-reversed into shared memory for the inverse FFT below.
        for i in range(elems):
            var k = tid + i * THREADS
            var idx = half - k
            var ar = rebind[Scalar[DType.float32]](in_re[row, k])
            var ai = rebind[Scalar[DType.float32]](in_im[row, k])
            var br = rebind[Scalar[DType.float32]](in_re[row, idx])
            var bi = -rebind[Scalar[DType.float32]](in_im[row, idx])
            var er = (ar + br) * 0.5
            var ei = (ai + bi) * 0.5
            var dr = (ar - br) * 0.5
            var di = (ai - bi) * 0.5
            var angle_tw = twiddle[N, N, TW, True](t_re, t_im, k)
            var wr = angle_tw[0]
            var wi = angle_tw[1]
            var or_ = dr * wr - di * wi
            var oi_ = dr * wi + di * wr
            var dst = bit_reverse_ct(k, log2h)
            s_re[dst] = er - oi_
            s_im[dst] = ei + or_
        barrier()

        fft_lds_stages[half, 0, True, TW, THREADS, N](t_re, t_im, s_re, s_im, tid)


def irfft_row_kernel[
    N: Int, InLT: TensorLayout, LT: TensorLayout, TW: Bool = False,
    THREADS: Int = N // 4,
](
    in_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    in_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    out_x: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """Inverse of `rfft_row_kernel`: (num_rows, N/2+1) non-redundant complex
    half spectrum -> (num_rows, N) real, via a single length-`N/2` inverse
    complex FFT per row (mirrors `irfft_1d` in `fft_cpu`).
    grid.x == num_rows, block.x == THREADS; see `rfft_row_kernel` for what
    `THREADS` trades."""
    comptime assert out_x.flat_rank == 2, "expected (rows, N) tensor"
    comptime half = N // 2
    comptime elems = half // THREADS

    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    irfft_row_into_lds[N, TW, InLT, THREADS](t_re, t_im, in_re, in_im, s_re, s_im, row, tid)

    comptime inv_half: Float32 = 1.0 / Float32(half)
    for i in range(elems):
        var n_ = tid + i * THREADS
        out_x[row, 2 * n_] = rebind[out_x.ElementType](s_re[n_] * inv_half)
        out_x[row, 2 * n_ + 1] = rebind[out_x.ElementType](s_im[n_] * inv_half)


def irfft_row_mul_kernel[
    N: Int, SHIFT: Bool, H: Int, InLT: TensorLayout, LT: TensorLayout,
    TW: Bool = False, THREADS: Int = N // 4,
](
    in_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    in_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    mul: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_x: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """`irfft_row_kernel` with the pipeline's final elementwise multiply --
    and, when `SHIFT`, the fftshift that precedes it -- folded into the
    store: computes `out = mul * irfft(in)` (or `out = mul *
    fftshift(irfft(in))`) in one kernel.

    The row is already finished and sitting in shared memory when
    `irfft_row_kernel` stores it, so the separate multiply kernel that used
    to follow existed only to read that row straight back out of global
    memory. Folding it in removes a full (D, H, W) write and its matching
    read; see optimizations.md sec. 3(c). `mul` and `out_x` are both
    (num_rows, N) with `num_rows == D * H`; they must not alias.

    `SHIFT` costs nothing in bandwidth because fftshift over the last two
    axes is a pure index rotation, and rotating the *destination* keeps both
    the `mul` read and the `out_x` store coalesced: the row rotation just
    picks a different destination row (`(rr + H/2) % H`, self-inverse for
    even `H`), and the column rotation turns each block's contiguous run of
    columns into two contiguous half-row segments. `H` is the height of one
    depth slice, used only to split the flat row index; it is ignored when
    `SHIFT` is False."""
    comptime assert mul.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert out_x.flat_rank == 2, "expected (rows, N) tensor"
    comptime half = N // 2
    comptime elems = half // THREADS

    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    var t_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    var t_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[tw_alloc(N, TW)]())
    fill_twiddle_table[N, TW](t_re, t_im, tid, THREADS)

    irfft_row_into_lds[N, TW, InLT, THREADS](t_re, t_im, in_re, in_im, s_re, s_im, row, tid)

    var out_row: Int
    comptime if SHIFT:
        out_row = (row // H) * H + (row % H + H // 2) % H
    else:
        out_row = row

    comptime inv_half: Float32 = 1.0 / Float32(half)
    for i in range(elems):
        var n_ = tid + i * THREADS
        var col: Int
        comptime if SHIFT:
            col = (2 * n_ + half) % N
        else:
            col = 2 * n_
        var m0 = rebind[Scalar[DType.float32]](mul[out_row, col])
        var m1 = rebind[Scalar[DType.float32]](mul[out_row, col + 1])
        out_x[out_row, col] = rebind[out_x.ElementType](s_re[n_] * inv_half * m0)
        out_x[out_row, col + 1] = rebind[out_x.ElementType](s_im[n_] * inv_half * m1)


def transpose_kernel[
    TILE: Int, H: Int, W: Int, InLT: TensorLayout, OutLT: TensorLayout
](
    src_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    src_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    dst_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    dst_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Transposes the last two axes of a (batch, H, W) tensor into a
    (batch, W, H) tensor, tiled through shared memory. grid = (ceil(W/TILE),
    ceil(H/TILE), batch); block = (TILE, TILE)."""
    comptime assert src_re.flat_rank == 3, "expected (batch, H, W) tensor"
    comptime assert dst_re.flat_rank == 3, "expected (batch, W, H) tensor"

    # Diagonal block-index remap (the classic NVIDIA-transpose-sample
    # trick). Whenever the destination's fast (last) axis length is an
    # exact power of two -- true here for both transpose directions in the
    # pipeline, since H = W = 2048 -- every concurrently-scheduled block
    # writes to addresses that collide on the same DRAM channel/bank
    # ("partition camping"), measured on this GPU as a 3-4.5x slowdown
    # (isolated microbenchmark: a destination pitch of 1024/2048 costs
    # ~43ms/29ms vs ~10ms for a non-power-of-two pitch of 1025, with grid
    # shape and data volume held identical). Remapping which logical tile a
    # given hardware-scheduled block index handles -- a pure bijection over
    # the same set of tiles, so it changes nothing about correctness --
    # decorrelates consecutively-scheduled blocks' destination addresses
    # from the power-of-two stride and restores full channel/bank spread.
    # Verified bit-exact against the un-remapped kernel and against numpy;
    # measured 3.4x faster on the previously-slow direction with no
    # regression on the already-fast one.
    comptime Gx = ceildiv(W, TILE)
    comptime Gy = ceildiv(H, TILE)

    var b = block_idx.z
    var tx = thread_idx.x
    var ty = thread_idx.y

    var bid = block_idx.x + Gx * block_idx.y
    var blk_y = bid % Gy
    var blk_x = ((bid // Gy) + blk_y) % Gx

    # Padded to TILE+1 columns: the un-padded read below (`tile_re[tx, ty]`
    # with tx varying across a warp) would stride by exactly TILE words,
    # putting every lane in the same shared-memory bank (TILE is a power of
    # two, a multiple of the 32-bank width) -- a 32-way bank conflict on
    # every read. The extra column breaks that stride so consecutive lanes
    # land in distinct banks.
    var tile_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())
    var tile_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())

    var x = blk_x * TILE + tx
    var y = blk_y * TILE + ty
    if y < H and x < W:
        tile_re[ty, tx] = rebind[Scalar[DType.float32]](src_re[b, y, x])
        tile_im[ty, tx] = rebind[Scalar[DType.float32]](src_im[b, y, x])
    barrier()

    var x2 = blk_y * TILE + tx
    var y2 = blk_x * TILE + ty
    if y2 < W and x2 < H:
        dst_re[b, y2, x2] = rebind[dst_re.ElementType](tile_re[tx, ty])
        dst_im[b, y2, x2] = rebind[dst_im.ElementType](tile_im[tx, ty])


def transpose_kernel_cmul[
    TILE: Int, H: Int, W: Int, InLT: TensorLayout, OutLT: TensorLayout
](
    a_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    a_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    b_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    b_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    dst_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    dst_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Fusion of an elementwise complex multiply into `transpose_kernel`'s
    load stage: transposes `a * b` directly, without materializing the
    product first. `a`, `b` have the same (batch, H, W) shape (no
    broadcasting; see `transpose_kernel_cmul_broadcast` for the broadcast
    case). Same diagonal block-index remap as `transpose_kernel`; see its
    docstring for why."""
    comptime assert a_re.flat_rank == 3, "expected (batch, H, W) tensor"
    comptime assert dst_re.flat_rank == 3, "expected (batch, W, H) tensor"
    comptime Gx = ceildiv(W, TILE)
    comptime Gy = ceildiv(H, TILE)

    var b = block_idx.z
    var tx = thread_idx.x
    var ty = thread_idx.y

    var bid = block_idx.x + Gx * block_idx.y
    var blk_y = bid % Gy
    var blk_x = ((bid // Gy) + blk_y) % Gx

    var tile_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())
    var tile_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())

    var x = blk_x * TILE + tx
    var y = blk_y * TILE + ty
    if y < H and x < W:
        var ar = rebind[Scalar[DType.float32]](a_re[b, y, x])
        var ai = rebind[Scalar[DType.float32]](a_im[b, y, x])
        var br = rebind[Scalar[DType.float32]](b_re[b, y, x])
        var bi = rebind[Scalar[DType.float32]](b_im[b, y, x])
        tile_re[ty, tx] = ar * br - ai * bi
        tile_im[ty, tx] = ar * bi + ai * br
    barrier()

    var x2 = blk_y * TILE + tx
    var y2 = blk_x * TILE + ty
    if y2 < W and x2 < H:
        dst_re[b, y2, x2] = rebind[dst_re.ElementType](tile_re[tx, ty])
        dst_im[b, y2, x2] = rebind[dst_im.ElementType](tile_im[tx, ty])


def transpose_kernel_cmul_broadcast[
    TILE: Int, H: Int, W: Int, ALT: TensorLayout, BLT: TensorLayout, OutLT: TensorLayout
](
    a_re: TileTensor[DType.float32, ALT, MutAnyOrigin],  # (H, W), broadcast over batch
    a_im: TileTensor[DType.float32, ALT, MutAnyOrigin],
    b_re: TileTensor[DType.float32, BLT, MutAnyOrigin],  # (batch, H, W)
    b_im: TileTensor[DType.float32, BLT, MutAnyOrigin],
    dst_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],  # (batch, W, H)
    dst_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Same fusion as `transpose_kernel_cmul`, but for the broadcast-over-
    depth complex multiply used elsewhere in the OpenFLR pipeline:
    `a` is a single (H, W) layer broadcast against every batch layer of
    `b`."""
    comptime assert a_re.flat_rank == 2, "expected (H, W) tensor"
    comptime assert b_re.flat_rank == 3, "expected (batch, H, W) tensor"
    comptime assert dst_re.flat_rank == 3, "expected (batch, W, H) tensor"
    comptime Gx = ceildiv(W, TILE)
    comptime Gy = ceildiv(H, TILE)

    var bb = block_idx.z
    var tx = thread_idx.x
    var ty = thread_idx.y

    var bid = block_idx.x + Gx * block_idx.y
    var blk_y = bid % Gy
    var blk_x = ((bid // Gy) + blk_y) % Gx

    var tile_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())
    var tile_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE + 1]())

    var x = blk_x * TILE + tx
    var y = blk_y * TILE + ty
    if y < H and x < W:
        var ar = rebind[Scalar[DType.float32]](a_re[y, x])
        var ai = rebind[Scalar[DType.float32]](a_im[y, x])
        var br = rebind[Scalar[DType.float32]](b_re[bb, y, x])
        var bi = rebind[Scalar[DType.float32]](b_im[bb, y, x])
        tile_re[ty, tx] = ar * br - ai * bi
        tile_im[ty, tx] = ar * bi + ai * br
    barrier()

    var x2 = blk_y * TILE + tx
    var y2 = blk_x * TILE + ty
    if y2 < W and x2 < H:
        dst_re[bb, y2, x2] = rebind[dst_re.ElementType](tile_re[tx, ty])
        dst_im[bb, y2, x2] = rebind[dst_im.ElementType](tile_im[tx, ty])


def fft2_batched_gpu[
    D: Int, H: Int, W: Int, TILE: Int, invert: Bool
](
    ctx: DeviceContext,
    mut re_buf: DeviceBuffer[DType.float32],
    mut im_buf: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """In-place 2D FFT (or inverse) of a (D, H, W) complex buffer pair over
    the last two axes. `H` and `W` must be powers of two. `t_re`/`t_im` are
    caller-owned scratch buffers of the same size as `re_buf`/`im_buf`, used
    to hold the intermediate transpose (so callers can reuse them across
    calls instead of paying an allocation every call)."""
    comptime row_layout_w = row_major[D * H, W]()
    var re_rows_w = TileTensor(re_buf, row_layout_w)
    var im_rows_w = TileTensor(im_buf, row_layout_w)
    comptime kernel_w = fft_row_kernel[W, type_of(row_layout_w), invert]
    ctx.enqueue_function[kernel_w](re_rows_w, im_rows_w, grid_dim=D * H, block_dim=W // 2)

    comptime in_layout_a = row_major[D, H, W]()
    comptime out_layout_a = row_major[D, W, H]()
    var re_in_a = TileTensor(re_buf, in_layout_a)
    var im_in_a = TileTensor(im_buf, in_layout_a)

    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel[TILE, H, W, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        re_in_a, im_in_a, re_out_a, im_out_a,
        grid_dim=(ceildiv(W, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), invert]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W, H]()
    comptime out_layout_b = row_major[D, H, W]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(re_buf, out_layout_b)
    var im_out_b = TileTensor(im_buf, out_layout_b)

    comptime tkernel_b = transpose_kernel[TILE, W, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W, TILE), D),
        block_dim=(TILE, TILE),
    )


def rfft2_batched_gpu[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Real-input 2D FFT of a (D, H, W) real buffer over the last two axes,
    mirroring `numpy.fft.rfft2`: a real-optimized FFT (`rfft_row_kernel`)
    along the width axis, then a full complex FFT along the height axis.
    `H` and `W` must be powers of two. Writes the non-redundant half
    spectrum, a (D, H, W/2+1) complex buffer pair, into `out_re`/`out_im`.
    `t_re`/`t_im` are caller-owned scratch buffers sized for (D, H, W/2+1)
    complex data, used to hold the intermediate transpose."""
    comptime W2 = W // 2 + 1
    comptime in_row_layout = row_major[D * H, W]()
    comptime out_row_layout = row_major[D * H, W2]()
    var x_rows = TileTensor(x_buf, in_row_layout)
    var re_rows = TileTensor(out_re, out_row_layout)
    var im_rows = TileTensor(out_im, out_row_layout)
    comptime kernel_w = rfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout)]
    ctx.enqueue_function[kernel_w](x_rows, re_rows, im_rows, grid_dim=D * H, block_dim=W // 4)

    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    var re_in_a = TileTensor(out_re, in_layout_a)
    var im_in_a = TileTensor(out_im, in_layout_a)
    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        re_in_a, im_in_a, re_out_a, im_out_a,
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W2, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), False]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W2, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(out_re, out_layout_b)
    var im_out_b = TileTensor(out_im, out_layout_b)

    comptime tkernel_b2 = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b2](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )


def rfft2_batched_gpu_div[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut a_buf: DeviceBuffer[DType.float32],
    mut b_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Fusion of a separate elementwise-divide kernel into `rfft2_batched_gpu`: computes
    `rfft2(a_buf / b_buf)` directly, dividing at the width-axis row-FFT's
    load stage (`rfft_row_kernel_div`) instead of materializing `a_buf /
    b_buf` through a separate kernel launch and buffer first. Otherwise
    identical to `rfft2_batched_gpu` (kept intact and unchanged for other
    call sites, e.g. PSF prep, that don't need a division)."""
    comptime W2 = W // 2 + 1
    comptime in_row_layout = row_major[D * H, W]()
    comptime out_row_layout = row_major[D * H, W2]()
    var a_rows = TileTensor(a_buf, in_row_layout)
    var b_rows = TileTensor(b_buf, in_row_layout)
    var re_rows = TileTensor(out_re, out_row_layout)
    var im_rows = TileTensor(out_im, out_row_layout)
    comptime kernel_w = rfft_row_kernel_div[W, type_of(in_row_layout), type_of(out_row_layout)]
    ctx.enqueue_function[kernel_w](a_rows, b_rows, re_rows, im_rows, grid_dim=D * H, block_dim=W // 4)

    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    var re_in_a = TileTensor(out_re, in_layout_a)
    var im_in_a = TileTensor(out_im, in_layout_a)
    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        re_in_a, im_in_a, re_out_a, im_out_a,
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W2, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), False]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W2, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(out_re, out_layout_b)
    var im_out_b = TileTensor(out_im, out_layout_b)

    comptime tkernel_b2 = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b2](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )


def irfft2_batched_gpu[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut in_re: DeviceBuffer[DType.float32],
    mut in_im: DeviceBuffer[DType.float32],
    mut out_x: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Inverse of `rfft2_batched_gpu`: a (D, H, W/2+1) non-redundant complex
    half spectrum -> a (D, H, W) real buffer, mirroring
    `numpy.fft.irfft2`. `H` and `W` must be powers of two. `in_re`/`in_im`
    are used as scratch and left in an undefined state; `t_re`/`t_im` are
    caller-owned scratch buffers sized for (D, H, W/2+1) complex data."""
    comptime W2 = W // 2 + 1
    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    var re_in_a = TileTensor(in_re, in_layout_a)
    var im_in_a = TileTensor(in_im, in_layout_a)
    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        re_in_a, im_in_a, re_out_a, im_out_a,
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W2, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), True]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W2, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(in_re, out_layout_b)
    var im_out_b = TileTensor(in_im, out_layout_b)

    comptime tkernel_b3 = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b3](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime in_row_layout = row_major[D * H, W2]()
    comptime out_row_layout = row_major[D * H, W]()
    var re_rows = TileTensor(in_re, in_row_layout)
    var im_rows = TileTensor(in_im, in_row_layout)
    var x_rows = TileTensor(out_x, out_row_layout)
    comptime kernel_w = irfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout)]
    ctx.enqueue_function[kernel_w](re_rows, im_rows, x_rows, grid_dim=D * H, block_dim=W // 4)


def irfft2_batched_gpu_cmul[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut a_re: DeviceBuffer[DType.float32],
    mut a_im: DeviceBuffer[DType.float32],
    mut b_re: DeviceBuffer[DType.float32],
    mut b_im: DeviceBuffer[DType.float32],
    mut out_x: DeviceBuffer[DType.float32],
    mut scratch_re: DeviceBuffer[DType.float32],
    mut scratch_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Fusion of an elementwise complex multiply (mirrors
    `complex_mul_kernel`) into `irfft2_batched_gpu`'s first transpose:
    computes `irfft2(a * b)` directly, without ever materializing the
    (D, H, W/2+1) complex product `a * b` -- the single biggest buffer in
    the OpenFLR pipeline. `a`, `b` have the same (D, H, W/2+1) shape (no
    broadcasting; see `irfft2_batched_gpu_cmul_broadcast` for the broadcast
    case). `scratch_re`/`scratch_im` are caller-owned (D, H, W/2+1) scratch
    used for the final-stage transpose (in the un-fused version this role
    was played by the product buffer itself, which no longer exists here).
    `t_re`/`t_im` are caller-owned scratch as in `irfft2_batched_gpu`.
    Otherwise identical to `irfft2_batched_gpu`."""
    comptime W2 = W // 2 + 1
    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    var a_re_t = TileTensor(a_re, in_layout_a)
    var a_im_t = TileTensor(a_im, in_layout_a)
    var b_re_t = TileTensor(b_re, in_layout_a)
    var b_im_t = TileTensor(b_im, in_layout_a)
    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel_cmul[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        a_re_t, a_im_t, b_re_t, b_im_t, re_out_a, im_out_a,
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W2, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), True]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W2, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(scratch_re, out_layout_b)
    var im_out_b = TileTensor(scratch_im, out_layout_b)

    comptime tkernel_b3 = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b3](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime in_row_layout = row_major[D * H, W2]()
    comptime out_row_layout = row_major[D * H, W]()
    var re_rows = TileTensor(scratch_re, in_row_layout)
    var im_rows = TileTensor(scratch_im, in_row_layout)
    var x_rows = TileTensor(out_x, out_row_layout)
    comptime kernel_w = irfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout)]
    ctx.enqueue_function[kernel_w](re_rows, im_rows, x_rows, grid_dim=D * H, block_dim=W // 4)


def irfft2_batched_gpu_cmul_broadcast[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut a_re: DeviceBuffer[DType.float32],  # (H, W/2+1), broadcast over depth
    mut a_im: DeviceBuffer[DType.float32],
    mut b_re: DeviceBuffer[DType.float32],  # (D, H, W/2+1)
    mut b_im: DeviceBuffer[DType.float32],
    mut out_x: DeviceBuffer[DType.float32],
    mut scratch_re: DeviceBuffer[DType.float32],
    mut scratch_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Same fusion as `irfft2_batched_gpu_cmul`, but for the broadcast-over-
    depth complex multiply used elsewhere in the OpenFLR pipeline:
    `a` is a single (H, W/2+1) layer broadcast against every depth layer of
    `b`."""
    comptime W2 = W // 2 + 1
    comptime a_layout = row_major[H, W2]()
    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    var a_re_t = TileTensor(a_re, a_layout)
    var a_im_t = TileTensor(a_im, a_layout)
    var b_re_t = TileTensor(b_re, in_layout_a)
    var b_im_t = TileTensor(b_im, in_layout_a)
    var re_out_a = TileTensor(t_re, out_layout_a)
    var im_out_a = TileTensor(t_im, out_layout_a)

    comptime tkernel_a = transpose_kernel_cmul_broadcast[TILE, H, W2, type_of(a_layout), type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        a_re_t, a_im_t, b_re_t, b_im_t, re_out_a, im_out_a,
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime row_layout_h = row_major[D * W2, H]()
    var re_rows_h = TileTensor(t_re, row_layout_h)
    var im_rows_h = TileTensor(t_im, row_layout_h)
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), True]
    ctx.enqueue_function[kernel_h](re_rows_h, im_rows_h, grid_dim=D * W2, block_dim=H // 2)

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    var re_in_b = TileTensor(t_re, in_layout_b)
    var im_in_b = TileTensor(t_im, in_layout_b)
    var re_out_b = TileTensor(scratch_re, out_layout_b)
    var im_out_b = TileTensor(scratch_im, out_layout_b)

    comptime tkernel_b3 = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b3](
        re_in_b, im_in_b, re_out_b, im_out_b,
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime in_row_layout = row_major[D * H, W2]()
    comptime out_row_layout = row_major[D * H, W]()
    var re_rows = TileTensor(scratch_re, in_row_layout)
    var im_rows = TileTensor(scratch_im, in_row_layout)
    var x_rows = TileTensor(out_x, out_row_layout)
    comptime kernel_w = irfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout)]
    ctx.enqueue_function[kernel_w](re_rows, im_rows, x_rows, grid_dim=D * H, block_dim=W // 4)



def rfft_w_div_transposed_gpu[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, WDIV: Int = 4
](
    ctx: DeviceContext,
    mut a_buf: DeviceBuffer[DType.float32],
    mut b_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """`rfft_w_transposed_gpu` with the elementwise divide of
    `rfft_row_kernel_div` folded into the width-axis row FFT's load: the
    width-axis real FFT of `a_buf / b_buf` plus the transpose, leaving the
    height axis untransformed and the result in the transposed
    (D, W/2+1, H) layout. `t_re`/`t_im` are caller-owned (D, H, W/2+1)
    scratch."""
    comptime W2 = W // 2 + 1
    comptime in_row_layout = row_major[D * H, W]()
    comptime out_row_layout = row_major[D * H, W2]()
    comptime WT = W // WDIV
    comptime kernel_w = rfft_row_kernel_div[W, type_of(in_row_layout), type_of(out_row_layout), TW, WT]
    ctx.enqueue_function[kernel_w](
        TileTensor(a_buf, in_row_layout), TileTensor(b_buf, in_row_layout),
        TileTensor(t_re, out_row_layout), TileTensor(t_im, out_row_layout),
        grid_dim=D * H, block_dim=WT,
    )

    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    comptime tkernel_a = transpose_kernel[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        TileTensor(t_re, in_layout_a), TileTensor(t_im, in_layout_a),
        TileTensor(out_re, out_layout_a), TileTensor(out_im, out_layout_a),
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )


def rfft2_batched_gpu_div_t[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, TDIV: Int = 2,
    WDIV: Int = 4,
](
    ctx: DeviceContext,
    mut a_buf: DeviceBuffer[DType.float32],
    mut b_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """`rfft2_batched_gpu_div` that stops one transpose early, leaving the
    half spectrum in the transposed (D, W/2+1, H) canonical layout -- the
    same relationship `rfft2_batched_gpu_t` has to `rfft2_batched_gpu`. Its
    consumer, `irfft2_batched_gpu_cmul_broadcast_t`, wants the transposed
    layout, so the final transpose is pure waste."""
    comptime W2 = W // 2 + 1
    rfft_w_div_transposed_gpu[D, H, W, TILE, TW, WDIV](ctx, a_buf, b_buf, out_re, out_im, t_re, t_im)

    comptime HT = H // TDIV
    comptime row_layout_h = row_major[D * W2, H]()
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), False, TW, HT]
    ctx.enqueue_function[kernel_h](
        TileTensor(out_re, row_layout_h), TileTensor(out_im, row_layout_h),
        grid_dim=D * W2, block_dim=HT,
    )


def irfft2_batched_gpu_cmul_broadcast_mul_t[
    D: Int, H: Int, W: Int, TILE: Int, SHIFT: Bool, TW: Bool = False,
    TDIV: Int = 2, WDIV: Int = 4,
](
    ctx: DeviceContext,
    mut a_re: DeviceBuffer[DType.float32],  # (W/2+1, H), broadcast over depth
    mut a_im: DeviceBuffer[DType.float32],
    mut b_re: DeviceBuffer[DType.float32],  # (D, W/2+1, H)
    mut b_im: DeviceBuffer[DType.float32],
    mut mul_buf: DeviceBuffer[DType.float32],  # (D, H, W) real
    mut out_x: DeviceBuffer[DType.float32],
    mut scratch_re: DeviceBuffer[DType.float32],
    mut scratch_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """`irfft2_batched_gpu_cmul_broadcast` for operands already in the
    transposed (D, W/2+1, H) canonical layout, with the pipeline's final
    real multiply folded into the last kernel's store: computes
    `mul_buf * irfft2(a * b)`, or `mul_buf * fftshift(irfft2(a * b))` when
    `SHIFT`, with `a` a single (W/2+1, H) layer broadcast against every
    depth layer of `b`.

    Three dispatches instead of five. The leading
    `transpose_kernel_cmul_broadcast` is gone outright -- with both operands
    transposed there is no transpose left to perform, and the multiply it
    carried is folded into the inverse column FFT's load
    (`ifft_row_cmul_broadcast_kernel`). That transpose was the worst kernel
    in the pipeline by a wide margin (81 GB/s in v1, 99 GB/s in v2, 17-18%
    of total runtime); it and the column FFT that followed it become one
    kernel moving 2/3 of the bytes. See optimizations.md sec. 3(b).

    `scratch_re`/`scratch_im` are caller-owned (D, H, W/2+1) scratch for the
    remaining transpose; `t_re`/`t_im` are (D, W/2+1, H) scratch holding the
    column-FFT result. Neither `a`, `b` nor `mul_buf` is modified, and
    `mul_buf` must not alias `out_x`. The trailing multiply used to be a
    separate full-size kernel reading back the (D, H, W) buffer the width
    inverse FFT had just written; see `irfft_row_mul_kernel` and
    optimizations.md sec. 3(c)."""
    comptime W2 = W // 2 + 1
    comptime a_row_layout = row_major[W2, H]()
    comptime b_row_layout = row_major[D * W2, H]()
    comptime HT = H // TDIV
    comptime kernel_h = ifft_row_cmul_broadcast_kernel[
        H, D, W2, type_of(b_row_layout), type_of(a_row_layout), type_of(b_row_layout), TW, HT
    ]
    ctx.enqueue_function[kernel_h](
        TileTensor(b_re, b_row_layout), TileTensor(b_im, b_row_layout),
        TileTensor(a_re, a_row_layout), TileTensor(a_im, a_row_layout),
        TileTensor(t_re, b_row_layout), TileTensor(t_im, b_row_layout),
        grid_dim=D * W2, block_dim=HT,
    )

    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    comptime tkernel_b = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b](
        TileTensor(t_re, in_layout_b), TileTensor(t_im, in_layout_b),
        TileTensor(scratch_re, out_layout_b), TileTensor(scratch_im, out_layout_b),
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime in_row_layout = row_major[D * H, W2]()
    comptime out_row_layout = row_major[D * H, W]()
    comptime WT = W // WDIV
    comptime kernel_w = irfft_row_mul_kernel[
        W, SHIFT, H, type_of(in_row_layout), type_of(out_row_layout), TW, WT
    ]
    ctx.enqueue_function[kernel_w](
        TileTensor(scratch_re, in_row_layout), TileTensor(scratch_im, in_row_layout),
        TileTensor(mul_buf, out_row_layout), TileTensor(out_x, out_row_layout),
        grid_dim=D * H, block_dim=WT,
    )


def rfft_w_transposed_gpu[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, WDIV: Int = 4
](
    ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """First half of `rfft2_batched_gpu`: the width-axis real FFT plus the
    transpose, leaving the height axis untransformed and the result in the
    transposed (D, W/2+1, H) layout that the column-FFT kernels consume
    directly. `t_re`/`t_im` are caller-owned (D, H, W/2+1) scratch."""
    comptime W2 = W // 2 + 1
    comptime in_row_layout = row_major[D * H, W]()
    comptime out_row_layout = row_major[D * H, W2]()
    var x_rows = TileTensor(x_buf, in_row_layout)
    var re_rows = TileTensor(t_re, out_row_layout)
    var im_rows = TileTensor(t_im, out_row_layout)
    comptime WT = W // WDIV
    comptime kernel_w = rfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout), TW, WT]
    ctx.enqueue_function[kernel_w](x_rows, re_rows, im_rows, grid_dim=D * H, block_dim=WT)

    comptime in_layout_a = row_major[D, H, W2]()
    comptime out_layout_a = row_major[D, W2, H]()
    comptime tkernel_a = transpose_kernel[TILE, H, W2, type_of(in_layout_a), type_of(out_layout_a)]
    ctx.enqueue_function[tkernel_a](
        TileTensor(t_re, in_layout_a), TileTensor(t_im, in_layout_a),
        TileTensor(out_re, out_layout_a), TileTensor(out_im, out_layout_a),
        grid_dim=(ceildiv(W2, TILE), ceildiv(H, TILE), D),
        block_dim=(TILE, TILE),
    )


def irfft_w_from_transposed_gpu[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, WDIV: Int = 4
](
    ctx: DeviceContext,
    mut in_re: DeviceBuffer[DType.float32],
    mut in_im: DeviceBuffer[DType.float32],
    mut out_x: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Inverse of `rfft_w_transposed_gpu`: a (D, W/2+1, H) buffer whose
    height axis has already been inverse-transformed -> a (D, H, W) real
    buffer. `t_re`/`t_im` are caller-owned (D, H, W/2+1) scratch."""
    comptime W2 = W // 2 + 1
    comptime in_layout_b = row_major[D, W2, H]()
    comptime out_layout_b = row_major[D, H, W2]()
    comptime tkernel_b = transpose_kernel[TILE, W2, H, type_of(in_layout_b), type_of(out_layout_b)]
    ctx.enqueue_function[tkernel_b](
        TileTensor(in_re, in_layout_b), TileTensor(in_im, in_layout_b),
        TileTensor(t_re, out_layout_b), TileTensor(t_im, out_layout_b),
        grid_dim=(ceildiv(H, TILE), ceildiv(W2, TILE), D),
        block_dim=(TILE, TILE),
    )

    comptime in_row_layout = row_major[D * H, W2]()
    comptime out_row_layout = row_major[D * H, W]()
    comptime WT = W // WDIV
    comptime kernel_w = irfft_row_kernel[W, type_of(in_row_layout), type_of(out_row_layout), TW, WT]
    ctx.enqueue_function[kernel_w](
        TileTensor(t_re, in_row_layout), TileTensor(t_im, in_row_layout),
        TileTensor(out_x, out_row_layout),
        grid_dim=D * H, block_dim=WT,
    )


def fft_col_cmul_ifft_gpu[
    D: Int, H: Int, W: Int, TW: Bool = False
](
    ctx: DeviceContext,
    mut re: DeviceBuffer[DType.float32],
    mut im: DeviceBuffer[DType.float32],
    mut p_re: DeviceBuffer[DType.float32],
    mut p_im: DeviceBuffer[DType.float32],
) raises:
    """Forward height-axis FFT, complex multiply by `p`, and inverse
    height-axis FFT of a (D, W/2+1, H) transposed spectrum, in place and in
    a single kernel. `p_re`/`p_im` must be in the same transposed layout.

    Replaces the four-dispatch `fft_row(H, fwd) -> transpose -> transpose-
    with-multiply -> fft_row(H, inv)` sequence the pipeline used to run
    between `rfft2` and `irfft2`, whose two transposes were exact inverses
    of each other. See optimizations.md sec. 3(a)."""
    comptime W2 = W // 2 + 1
    comptime row_layout = row_major[D * W2, H]()
    comptime kernel = fft_row_cmul_ifft_kernel[H, type_of(row_layout), type_of(row_layout), TW]
    ctx.enqueue_function[kernel](
        TileTensor(re, row_layout), TileTensor(im, row_layout),
        TileTensor(p_re, row_layout), TileTensor(p_im, row_layout),
        grid_dim=D * W2, block_dim=H // 2,
    )


def rfft2_batched_gpu_t[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, TDIV: Int = 2,
    WDIV: Int = 4,
](
    ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_re: DeviceBuffer[DType.float32],
    mut out_im: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """`rfft2_batched_gpu` that stops one transpose early, leaving the half
    spectrum in the transposed (D, W/2+1, H) canonical layout. Elementwise
    frequency-domain work and depth reductions commute with the transpose,
    so keeping the spectrum transposed costs nothing and saves the
    round-trip transpose pair that used to sit across every
    `rfft2 -> elementwise -> irfft2` boundary."""
    comptime W2 = W // 2 + 1
    rfft_w_transposed_gpu[D, H, W, TILE, TW, WDIV](ctx, x_buf, out_re, out_im, t_re, t_im)

    comptime HT = H // TDIV
    comptime row_layout_h = row_major[D * W2, H]()
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), False, TW, HT]
    ctx.enqueue_function[kernel_h](
        TileTensor(out_re, row_layout_h), TileTensor(out_im, row_layout_h),
        grid_dim=D * W2, block_dim=HT,
    )


def irfft2_batched_gpu_t[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, TDIV: Int = 2,
    WDIV: Int = 4,
](
    ctx: DeviceContext,
    mut in_re: DeviceBuffer[DType.float32],
    mut in_im: DeviceBuffer[DType.float32],
    mut out_x: DeviceBuffer[DType.float32],
    mut t_re: DeviceBuffer[DType.float32],
    mut t_im: DeviceBuffer[DType.float32],
) raises:
    """Inverse of `rfft2_batched_gpu_t`: a (D, W/2+1, H) transposed half
    spectrum -> a (D, H, W) real buffer. `in_re`/`in_im` are used as scratch
    and left in an undefined state."""
    comptime W2 = W // 2 + 1
    comptime HT = H // TDIV
    comptime row_layout_h = row_major[D * W2, H]()
    comptime kernel_h = fft_row_kernel[H, type_of(row_layout_h), True, TW, HT]
    ctx.enqueue_function[kernel_h](
        TileTensor(in_re, row_layout_h), TileTensor(in_im, row_layout_h),
        grid_dim=D * W2, block_dim=HT,
    )

    irfft_w_from_transposed_gpu[D, H, W, TILE, TW, WDIV](ctx, in_re, in_im, out_x, t_re, t_im)
