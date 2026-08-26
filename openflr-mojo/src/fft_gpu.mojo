"""GPU radix-2 Cooley-Tukey FFT kernels.

One thread block handles one row (a length-`N` complex signal); the whole
row lives in shared memory for the duration of the transform. `N` must be a
power of two and small enough that `N/2` threads and `2*N` float32 words of
shared memory are within device limits (true up to N=2048 on all current
GPUs).
"""

from std.math import cos, sin, ceildiv
from std.gpu import thread_idx, block_idx
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


def fft_row_kernel[
    N: Int, LT: TensorLayout, invert: Bool
](
    re: TileTensor[DType.float32, LT, MutAnyOrigin],
    im: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """`re`/`im` have shape (num_rows, N); grid.x == num_rows,
    block.x == N // 2."""
    comptime assert re.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert im.flat_rank == 2, "expected (rows, N) tensor"
    comptime log2n = ilog2_ct(N)
    comptime half = N // 2

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[N]())

    # Bit-reversed load: thread `tid` fills slots `tid` and `tid + half`.
    var src0 = bit_reverse_ct(tid, log2n)
    var src1 = bit_reverse_ct(tid + half, log2n)
    s_re[tid] = rebind[Scalar[DType.float32]](re[row, src0])
    s_im[tid] = rebind[Scalar[DType.float32]](im[row, src0])
    s_re[tid + half] = rebind[Scalar[DType.float32]](re[row, src1])
    s_im[tid + half] = rebind[Scalar[DType.float32]](im[row, src1])
    barrier()

    comptime sign: Float32 = 1.0 if invert else -1.0
    comptime for stage in range(log2n):
        comptime size = 1 << (stage + 1)
        comptime stage_half = size // 2
        var group = tid // stage_half
        var k = tid % stage_half
        var i0 = group * size + k
        var i1 = i0 + stage_half
        var angle = sign * 2.0 * PI * Float32(k) / Float32(size)
        var wr = cos(angle)
        var wi = sin(angle)
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

    comptime if invert:
        comptime inv_n: Float32 = 1.0 / Float32(N)
        re[row, tid] = rebind[re.ElementType](s_re[tid] * inv_n)
        im[row, tid] = rebind[im.ElementType](s_im[tid] * inv_n)
        re[row, tid + half] = rebind[re.ElementType](s_re[tid + half] * inv_n)
        im[row, tid + half] = rebind[im.ElementType](s_im[tid + half] * inv_n)
    else:
        re[row, tid] = rebind[re.ElementType](s_re[tid])
        im[row, tid] = rebind[im.ElementType](s_im[tid])
        re[row, tid + half] = rebind[re.ElementType](s_re[tid + half])
        im[row, tid + half] = rebind[im.ElementType](s_im[tid + half])


def rfft_row_kernel[
    N: Int, LT: TensorLayout, OutLT: TensorLayout
](
    x: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_re: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    out_im: TileTensor[DType.float32, OutLT, MutAnyOrigin],
):
    """Real-input FFT of each row of length `N` (real) into the
    non-redundant half spectrum of length `N/2+1`, via a single length-`N/2`
    complex FFT per row (mirrors `rfft_1d` in `fft_cpu`). `x` has shape
    (num_rows, N); `out_re`/`out_im` have shape (num_rows, N/2+1);
    grid.x == num_rows, block.x == N // 4."""
    comptime assert x.flat_rank == 2, "expected (rows, N) tensor"
    comptime assert out_re.flat_rank == 2, "expected (rows, N/2+1) tensor"
    comptime half = N // 2
    comptime half2 = N // 4
    comptime log2h = ilog2_ct(half)

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    # Pack: z[n] = x[2n] + j*x[2n+1], bit-reversed load into shared memory.
    var src0 = bit_reverse_ct(tid, log2h)
    var src1 = bit_reverse_ct(tid + half2, log2h)
    s_re[tid] = rebind[Scalar[DType.float32]](x[row, 2 * src0])
    s_im[tid] = rebind[Scalar[DType.float32]](x[row, 2 * src0 + 1])
    s_re[tid + half2] = rebind[Scalar[DType.float32]](x[row, 2 * src1])
    s_im[tid + half2] = rebind[Scalar[DType.float32]](x[row, 2 * src1 + 1])
    barrier()

    comptime for stage in range(log2h):
        comptime size = 1 << (stage + 1)
        comptime stage_half = size // 2
        var group = tid // stage_half
        var k = tid % stage_half
        var i0 = group * size + k
        var i1 = i0 + stage_half
        var angle = -2.0 * PI * Float32(k) / Float32(size)
        var wr = cos(angle)
        var wi = sin(angle)
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

    # Unpack Z (length `half`, in shared memory) into the length-N half
    # spectrum X[0 .. half] via the even/odd DFT symmetry.
    for i in range(2):
        var k = tid + i * half2
        var km = (half - k) % half
        var zr = s_re[k]
        var zi = s_im[k]
        var zr_km = s_re[km]
        var zi_km = s_im[km]
        var xe_re = (zr + zr_km) * 0.5
        var xe_im = (zi - zi_km) * 0.5
        var xo_re = (zi + zi_km) * 0.5
        var xo_im = -(zr - zr_km) * 0.5
        var angle = -2.0 * PI * Float32(k) / Float32(N)
        var wr = cos(angle)
        var wi = sin(angle)
        out_re[row, k] = rebind[out_re.ElementType](xe_re + (wr * xo_re - wi * xo_im))
        out_im[row, k] = rebind[out_im.ElementType](xe_im + (wr * xo_im + wi * xo_re))

    if tid == 0:
        out_re[row, half] = rebind[out_re.ElementType](s_re[0] - s_im[0])
        out_im[row, half] = rebind[out_im.ElementType](Scalar[DType.float32](0.0))


def irfft_row_kernel[
    N: Int, InLT: TensorLayout, LT: TensorLayout
](
    in_re: TileTensor[DType.float32, InLT, MutAnyOrigin],
    in_im: TileTensor[DType.float32, InLT, MutAnyOrigin],
    out_x: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """Inverse of `rfft_row_kernel`: (num_rows, N/2+1) non-redundant complex
    half spectrum -> (num_rows, N) real, via a single length-`N/2` inverse
    complex FFT per row (mirrors `irfft_1d` in `fft_cpu`).
    grid.x == num_rows, block.x == N // 4."""
    comptime assert in_re.flat_rank == 2, "expected (rows, N/2+1) tensor"
    comptime assert out_x.flat_rank == 2, "expected (rows, N) tensor"
    comptime half = N // 2
    comptime half2 = N // 4
    comptime log2h = ilog2_ct(half)

    var row = block_idx.x
    var tid = thread_idx.x

    var s_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())
    var s_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[half]())

    # Reconstruct Z = Xe + j*Xo (length `half`) from the half spectrum,
    # writing it bit-reversed into shared memory for the inverse FFT below.
    for i in range(2):
        var k = tid + i * half2
        var idx = half - k
        var ar = rebind[Scalar[DType.float32]](in_re[row, k])
        var ai = rebind[Scalar[DType.float32]](in_im[row, k])
        var br = rebind[Scalar[DType.float32]](in_re[row, idx])
        var bi = -rebind[Scalar[DType.float32]](in_im[row, idx])
        var er = (ar + br) * 0.5
        var ei = (ai + bi) * 0.5
        var dr = (ar - br) * 0.5
        var di = (ai - bi) * 0.5
        var angle = 2.0 * PI * Float32(k) / Float32(N)
        var wr = cos(angle)
        var wi = sin(angle)
        var or_ = dr * wr - di * wi
        var oi_ = dr * wi + di * wr
        var dst = bit_reverse_ct(k, log2h)
        s_re[dst] = er - oi_
        s_im[dst] = ei + or_
    barrier()

    comptime for stage in range(log2h):
        comptime size = 1 << (stage + 1)
        comptime stage_half = size // 2
        var group = tid // stage_half
        var k = tid % stage_half
        var i0 = group * size + k
        var i1 = i0 + stage_half
        var angle = 2.0 * PI * Float32(k) / Float32(size)
        var wr = cos(angle)
        var wi = sin(angle)
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

    comptime inv_half: Float32 = 1.0 / Float32(half)
    for i in range(2):
        var n_ = tid + i * half2
        out_x[row, 2 * n_] = rebind[out_x.ElementType](s_re[n_] * inv_half)
        out_x[row, 2 * n_ + 1] = rebind[out_x.ElementType](s_im[n_] * inv_half)


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

    var b = block_idx.z
    var tx = thread_idx.x
    var ty = thread_idx.y

    var tile_re = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE]())
    var tile_im = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TILE, TILE]())

    var x = block_idx.x * TILE + tx
    var y = block_idx.y * TILE + ty
    if y < H and x < W:
        tile_re[ty, tx] = rebind[Scalar[DType.float32]](src_re[b, y, x])
        tile_im[ty, tx] = rebind[Scalar[DType.float32]](src_im[b, y, x])
    barrier()

    var x2 = block_idx.y * TILE + tx
    var y2 = block_idx.x * TILE + ty
    if y2 < W and x2 < H:
        dst_re[b, y2, x2] = rebind[dst_re.ElementType](tile_re[tx, ty])
        dst_im[b, y2, x2] = rebind[dst_im.ElementType](tile_im[tx, ty])


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
