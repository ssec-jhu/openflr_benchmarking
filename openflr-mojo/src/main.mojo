"""CLI entrypoint: runs and benchmarks one OpenFLR Richardson-Lucy GPU
iteration on the real light-field dataset, mirroring the timing harness in
../../main.py (backend "mojo"). Also supports writing the output of a single
step to a file for offline correctness comparison against the numpy
reference.

Usage:
    mojo run main.mojo -- v2 20            # all arms, 20 iterations each
    mojo run main.mojo -- v2 20 t4         # one arm (what a profiler wants)
    mojo run main.mojo -- v2 1 t4 --dump out.bin   # one step, dumped

Arms: `base` (the pre-flag kernels), `t4w8c4` (kept as the cross-run
anchor), `t8w8c4G` (the best measured configuration), `t8w8c4Gi8` /
`t8w16c4G`, which re-ask two block-size questions that were last answered
before the coalesced gather landed, and `t8w8c4Gtw`, which re-asks the
twiddle-table question on hardware where the SFU:DRAM ratio is 1.23x worse
than the A100 it lost on. The first five are bit-identical -- they only
remap work across threads and memory; `t8w8c4Gtw` is not, because a table
is a different rounding of the same twiddle.
"""

from std.sys import argv, has_accelerator, stderr
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext, DeviceBuffer

from fft_gpu import rfft2_batched_gpu_t
from openflr_cpu import flip_hw, shift_hw
from openflr_gpu import run_v1_step_gpu, run_v2_step_gpu, OpenFlrScratch

comptime D = 41
comptime H = 2048
comptime W = 2048
comptime W2 = W // 2 + 1
comptime HW = H * W
comptime HW2 = H * W2
comptime N = D * HW
comptime N2 = D * HW2
comptime TILE = 32

comptime DATA_DIR = "data/"


def read_floats(path: String, n: Int) raises -> List[Float32]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) != n * 4:
        raise Error(
            "unexpected file size for " + path + ": got " + String(len(bytes))
            + " bytes, expected " + String(n * 4)
        )
    var out: List[Float32] = []
    var ptr = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        out.append(ptr[unsafe_offset=i])
    return out^


def write_floats(path: String, data: List[Float32], n: Int) raises:
    var f = open(path, "w")
    var ptr = data.unsafe_ptr().unsafe_bitcast[UInt8]()
    var span = Span[UInt8](unsafe_ptr=ptr, length=n * 4)
    f.write_bytes(span)
    f.close()


def upload[M: Int](ctx: DeviceContext, data: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](M)
    with buf.map_to_host() as h:
        var t = TileTensor(h, row_major[M]())
        for i in range(M):
            t[i] = data[i]
    return buf^


def download[M: Int](ctx: DeviceContext, buf: DeviceBuffer[DType.float32]) raises -> List[Float32]:
    var out: List[Float32] = []
    with buf.map_to_host() as h:
        var t = TileTensor(h, row_major[M]())
        for i in range(M):
            out.append(t[i])
    return out^


def global_sum(data: List[Float32], n: Int) -> Float32:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(data[i])
    return Float32(s)


def prepare_psf(ctx: DeviceContext, psf_raw: List[Float32]) raises -> Tuple[
    DeviceBuffer[DType.float32], DeviceBuffer[DType.float32],
    DeviceBuffer[DType.float32], DeviceBuffer[DType.float32],
    DeviceBuffer[DType.float32], DeviceBuffer[DType.float32],
]:
    """Normalizes the PSF and returns the rfft2 half spectrum of each:
    (psf_fft_re, psf_fft_im, psft_fft_v1_re, psft_fft_v1_im, psft_fft_v2_re,
    psft_fft_v2_im). All three are in the transposed canonical layout
    (D, W/2+1, H) that both the forward and the back-projection chains
    consume directly."""
    var total = global_sum(psf_raw, N)
    var psf: List[Float32] = []
    for i in range(N):
        psf.append(psf_raw[i] / total)

    var psf_flipped = flip_hw(psf.copy(), D, H, W)
    var psf_flipped_shifted = shift_hw(psf_flipped.copy(), D, H, W)

    var psf_buf = upload[N](ctx, psf^)
    var psf_fft_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psf_fft_im = ctx.enqueue_create_buffer[DType.float32](N2)
    var t_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var t_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu_t[D, H, W, TILE](ctx, psf_buf, psf_fft_re, psf_fft_im, t_re, t_im)

    var psft_v1_buf = upload[N](ctx, psf_flipped^)
    var psft_v1_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psft_v1_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu_t[D, H, W, TILE](ctx, psft_v1_buf, psft_v1_re, psft_v1_im, t_re, t_im)

    var psft_v2_buf = upload[N](ctx, psf_flipped_shifted^)
    var psft_v2_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psft_v2_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu_t[D, H, W, TILE](ctx, psft_v2_buf, psft_v2_re, psft_v2_im, t_re, t_im)

    return psf_fft_re^, psf_fft_im^, psft_v1_re^, psft_v1_im^, psft_v2_re^, psft_v2_im^


def mean_std(times: List[Float64], n: Int) -> Tuple[Float64, Float64]:
    var mean: Float64 = 0.0
    for i in range(n):
        mean += times[i]
    mean /= Float64(n)
    var var_sum: Float64 = 0.0
    for i in range(n):
        var d = times[i] - mean
        var_sum += d * d
    var std = (var_sum / Float64(n)) ** 0.5
    return mean, std


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var args = argv()
    var version = "v2"
    var n_iters = 20
    var dump_path = ""
    var arm = "all"
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--":
            pass
        elif a == "v1" or a == "v2":
            version = a
        elif a == "--dump":
            i += 1
            dump_path = String(args[i])
        elif (a == "base" or a == "t4w8c4" or a == "t8w8c4G"
              or a == "t8w8c4Gi8" or a == "t8w16c4G" or a == "t8w8c4Gtw"):
            arm = a
        else:
            n_iters = Int(a)
        i += 1

    if dump_path != "" and arm == "all":
        arm = "base"

    var ctx = DeviceContext()

    var image_raw = read_floats(DATA_DIR + "image.bin", HW)
    var psf_raw = read_floats(DATA_DIR + "psf.bin", N)

    var prep = prepare_psf(ctx, psf_raw^)
    var psf_fft_re = prep[0].copy()
    var psf_fft_im = prep[1].copy()
    var psft_v1_re = prep[2].copy()
    var psft_v1_im = prep[3].copy()
    var psft_v2_re = prep[4].copy()
    var psft_v2_im = prep[5].copy()

    var image_buf = upload[HW](ctx, image_raw^)

    var data_init: List[Float32] = []
    for _ in range(N):
        data_init.append(0.5)
    var data_buf = upload[N](ctx, data_init^)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var scratch = OpenFlrScratch[D, H, W](ctx)

    # Four configurations, benchmarked in one process on one set of clocks
    # so the numbers are directly comparable and a whole sweep costs one
    # batch job on a cluster with no interactive access:
    #
    #   base      TDIV=2 WDIV=4 CDIV=2 IDIV=2 CG=0
    #                           -- bit-identical to the pre-flag kernels
    #   t4w8c4    CG=0          -- the cross-run anchor: five runs at
    #                             0.01360-0.01362 on v2. If it moves more
    #                             than ~0.1% the run is not comparable to
    #                             the numbers in optimizations.md
    #   t8w8c4G   TDIV=8 WDIV=8 CDIV=4 IDIV=4 CG=2
    #                           -- the best measured configuration: every
    #                             row-FFT kernel reads its row coalesced and
    #                             bit-reverses in shared memory (CG=2, worth
    #                             -31.5% on v2), plus the block sizes three
    #                             earlier sessions settled
    #   t8w8c4Gi8 + IDIV=8      -- `ifft_row_cmul_broadcast_kernel` back to
    #                             256 threads. It lost 4.28% pre-gather, the
    #                             largest block-size effect measured here in
    #                             either direction; the gather fix already
    #                             moved one such answer by a third, so it is
    #                             worth one retest
    #   t8w16c4G  + WDIV=16     -- the width kernels at 128 threads, the
    #                             other stale block-size answer (-0.19% on
    #                             v2 pre-gather, close enough to nothing to
    #                             flip)
    #   t8w8c4Gtw + TW=True     -- the retired `tw` shared twiddle table,
    #                             un-retired for the H100. It lost 6-11% on
    #                             the A100 because Mojo's cos/sin lower to a
    #                             short SFU sequence there, so trading them
    #                             for shared-memory loads was a bad deal.
    #                             The deal is 1.23x worse on an H100 in one
    #                             direction only: A100 -> H100 NVL scales
    #                             DRAM by 1.91x but SMs x clock -- and with
    #                             them the SFU, the shared-memory pipe and
    #                             the instruction issue rate -- by only
    #                             1.55x, and mojo's own measured scaling is
    #                             1.49x against jax's 1.92x. So this arm
    #                             discriminates: if the SFU is what binds
    #                             these kernels it should now win, and if
    #                             shared memory is, it should lose by more
    #                             than it did on the A100. The one arm that
    #                             is *not* bit-exact with `base` -- a table
    #                             is a different rounding of the same
    #                             quantity -- so `verify_correctness.py`
    #                             holds it to the numpy tolerance only.
    #                             Measured: 2.1% of outputs differ from
    #                             `base` by at most 2.861e-06, against the
    #                             8.583e-06 the pipeline already differs
    #                             from numpy by.
    #
    # Retired after being measured: `tw` (a shared twiddle table -- back as
    # `t8w8c4Gtw` above, on the strength of the H100's different
    # SFU:DRAM ratio, not of any new argument about the A100), `t8`
    # (both column kernels at 256 threads -- a win and a larger loss added
    # together, see the `i8` session), `t4w16`, `t4w8c4i8`, `t4w8`, and
    # `t4w8c4g` (CG on `fft_row_kernel` alone, superseded by CG=2).
    #
    # Naming one of `base`, `t4w8`, `t4w8c4` runs just that arm, which is what a
    # profiler wants: every arm dispatches same-named kernels differing only
    # in their name hash. `--dump <path>` writes one step's output for
    # `verify_correctness.py` and defaults to `base` unless an arm is named.
    # Settle the clocks before any arm is timed, for a fixed wall-clock span
    # rather than a fixed iteration count.
    #
    # An `nsys` trace of an earlier run caught this GPU's clocks still
    # stepping ~1.7 s in -- the same unchanged kernel ran 18.8% faster one
    # dispatch later. A 5-iteration warmup is 75 ms on an A100, nowhere near
    # that, so whichever arm is timed first absorbed the ramp; the arms run
    # in a fixed order, so that was always `base`. It showed up
    # unmistakably: across two A100 nodes `tw`/`t4`/`t8` reproduced to
    # within 0.2% while `base` alone moved 4.4% and carried a 7-8% standard
    # deviation. Two seconds covers the ramp both on an A100 (~15 ms per
    # iteration) and on a far slower iGPU (~60 ms).
    # The warmup runs with *the arm that is about to be timed* (see the
    # first-arm block inside the sweep below), not with a hard-coded `base`.
    # That distinction only matters for a single-arm run, and there it
    # matters a lot: with a fixed `base` warmup, `nsys profile ... -- v2 3
    # t8w8c4G` captured ~135 `base` iterations and 3 of the arm, so any
    # aggregate report described `base` and merged the two wherever a kernel
    # name was unchanged. With `arm == "all"` the first arm executed is
    # `base` with exactly the parameters this loop used to hard-code, so
    # sweep numbers are unchanged by construction.
    comptime WARMUP_NS = 2_000_000_000
    var warmed = False

    # Two passes over the arms, the second in reverse order. On the A100 the
    # only drift was a clock ramp at the very start, which the warmup
    # covers. On an H100 NVL it is the opposite: the first two arms
    # reproduce to 0.04-0.11% while the third carries 3.2% -- drift that
    # develops *during* the run (thermal or power capping on a ~350 W card),
    # which a fixed arm order confounds with the arm itself. Sweeping twice
    # in opposite orders is the standard counterbalance: every arm is
    # measured once early and once late, the two positions sum to a
    # constant, so averaging the pair cancels linear drift and the spread
    # between them measures how much drift there was. Costs one extra second
    # of GPU time against a compile that takes minutes.
    #
    # For a head-to-head against another backend, prefer naming a single arm
    # (`-- v2 20 t8w8c4G`): a five-arm sweep heats the card in a way a
    # standalone `make time-jax-v2` does not, and that bias does not cancel
    # between processes.
    comptime for pass_i in range(2):
      comptime for slot in range(6):
        comptime arm_i = slot if pass_i == 0 else 5 - slot
        comptime TW = arm_i == 5
        comptime TDIV = 2 if arm_i == 0 else (4 if arm_i == 1 else 8)
        comptime WDIV = 4 if arm_i == 0 else (16 if arm_i == 4 else 8)
        comptime CDIV = 4 if arm_i >= 1 else 2
        # Pinned to 4 wherever `TDIV` is 8, so `TDIV` moves `fft_row_kernel`
        # and nothing else; the `i8` arm is the one that moves it.
        comptime IDIV = 8 if arm_i == 3 else (4 if arm_i >= 2 else TDIV)
        # 0 = off, 1 = `fft_row_kernel` only, 2 = every row-FFT kernel.
        comptime CG = 0 if arm_i <= 1 else 2
        comptime arm_name = "t4w8c4" if arm_i == 1 else (
            "t8w8c4G" if arm_i == 2 else ("t8w8c4Gi8" if arm_i == 3 else (
                "t8w16c4G" if arm_i == 4 else (
                    "t8w8c4Gtw" if arm_i == 5 else "base"))))

        if arm == "all" or arm == arm_name:
            if not warmed:
                warmed = True
                var w_start = perf_counter_ns()
                while perf_counter_ns() - w_start < WARMUP_NS:
                    if version == "v1":
                        run_v1_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                            ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
                        )
                    else:
                        run_v2_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                            ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v2_re, psft_v2_im, out_buf, scratch
                        )
                    ctx.synchronize()

            if version == "v1":
                run_v1_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                    ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
                )
            else:
                run_v2_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                    ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v2_re, psft_v2_im, out_buf, scratch
                )
            ctx.synchronize()

            if dump_path != "":
                var result = download[N](ctx, out_buf)
                write_floats(dump_path, result, N)
                print("wrote", dump_path, "(" + arm_name + ")")
                return

            var times: List[Float64] = []
            for _ in range(n_iters):
                var start = perf_counter_ns()
                if version == "v1":
                    run_v1_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                        ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
                    )
                else:
                    run_v2_step_gpu[D, H, W, TILE, TW, TDIV, WDIV, CDIV, IDIV, CG](
                        ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v2_re, psft_v2_im, out_buf, scratch
                    )
                ctx.synchronize()
                var elapsed = Float64(perf_counter_ns() - start) / 1.0e9
                times.append(elapsed)
                var tmp = data_buf
                data_buf = out_buf
                out_buf = tmp

            var stats = mean_std(times^, n_iters)
            print(
                "backend: mojo, version:", version, ", arm:", arm_name,
                ", pass:", pass_i,
                ", mean time:", stats[0], "s, std time:", stats[1], "s",
                file=stderr,
            )
            print("arm=" + arm_name, "pass=" + String(pass_i), stats[0], "\\pm", stats[1])
