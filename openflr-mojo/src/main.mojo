"""CLI entrypoint: runs and benchmarks one OpenFLR Richardson-Lucy GPU
iteration on the real light-field dataset, mirroring the timing harness in
../../main.py (backend "mojo"). Also supports writing the output of a single
step to a file for offline correctness comparison against the numpy
reference.

Usage:
    mojo run main.mojo -- v2 20            # all arms, 20 iterations each
    mojo run main.mojo -- v2 20 t4         # one arm (what a profiler wants)
    mojo run main.mojo -- v2 1 t4 --dump out.bin   # one step, dumped

Arms: `base` (the pre-flag kernels), `t4w8` (an intermediate kept as a
cross-run check), `t4w8c4` (the best measured configuration), `t8w8c4`
(`fft_row_kernel` alone at 256-thread blocks) and `t4w8c4g` (`fft_row_kernel`
reading its row coalesced and doing the bit-reversal in shared memory). All
five are bit-identical: they only remap work across threads and memory.
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
        elif (a == "base" or a == "t4w8" or a == "t4w8c4"
              or a == "t8w8c4" or a == "t4w8c4g"):
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
    #   base     TDIV=2 WDIV=4 CDIV=2 IDIV=2
    #                           -- bit-identical to the pre-flag kernels
    #   t4w8     TDIV=4 WDIV=8 -- 512-thread column blocks and 256-thread
    #                             width blocks. Superseded, but kept: it
    #                             was measured on two separate runs, so it
    #                             is the cross-run anchor that says whether
    #                             a new run is comparable to the old ones
    #   t4w8c4   + CDIV=4       -- plus 512-thread blocks on
    #                             `fft_row_cmul_ifft_kernel`. Measured
    #                             -4.95% on v1; the current best. v1 only,
    #                             so on v2 it duplicates `t4w8` and doubles
    #                             as a within-run reproducibility control
    #   t8w8c4   TDIV=8 IDIV=4  -- `t4w8c4` with `fft_row_kernel` alone
    #                             dropped to 256 threads,
    #                             `ifft_row_cmul_broadcast_kernel` left at
    #                             512. The other half of the split `t8`
    #                             conflated: `t8` moved both and lost 2.7%,
    #                             `t4w8c4i8` moved only the broadcast one
    #                             and lost 4.3%, which implies this
    #                             configuration *gains* ~1.6%. Affects both
    #                             versions
    #   t4w8c4g  + CG=True      -- `t4w8c4` with `fft_row_kernel`'s opening
    #                             bit-reversed gather moved off global
    #                             memory. That gather makes one warp's load
    #                             touch 32 distinct 32-byte sectors where a
    #                             coalesced one touches 4; reading the row
    #                             coalesced and permuting it in shared
    #                             memory instead costs two barriers and a
    #                             shared round trip. Mostly a v2 arm -- v1
    #                             dispatches this kernel only on the small
    #                             (1,H,W) stages
    #
    # Retired after being measured and losing: `tw` (a shared twiddle
    # table), `t8` (both column kernels at 256 threads), `t4w16`
    # (128-thread width blocks) and `t4w8c4i8` (256-thread
    # `ifft_row_cmul_broadcast_kernel`). See optimizations.md s.2, s.5, and
    # the `w8`, `c4` and `i8` sessions.
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
    comptime WARMUP_NS = 2_000_000_000
    var w_start = perf_counter_ns()
    while perf_counter_ns() - w_start < WARMUP_NS:
        if version == "v1":
            run_v1_step_gpu[D, H, W, TILE, False, 2](
                ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
            )
        else:
            run_v2_step_gpu[D, H, W, TILE, False, 2](
                ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v2_re, psft_v2_im, out_buf, scratch
            )
        ctx.synchronize()

    comptime for arm_i in range(5):
        comptime TW = False
        comptime TDIV = 2 if arm_i == 0 else (8 if arm_i == 3 else 4)
        comptime WDIV = 4 if arm_i == 0 else 8
        comptime CDIV = 4 if arm_i >= 2 else 2
        # Pinned to 4 on the `t8w8c4` arm so `TDIV` moves `fft_row_kernel`
        # and nothing else; everywhere else it tracks `TDIV`, as it always
        # did.
        comptime IDIV = 4 if arm_i == 3 else TDIV
        comptime CG = arm_i == 4
        comptime arm_name = "t4w8" if arm_i == 1 else (
            "t4w8c4" if arm_i == 2 else ("t8w8c4" if arm_i == 3 else (
                "t4w8c4g" if arm_i == 4 else "base")))

        if arm == "all" or arm == arm_name:
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
                ", mean time:", stats[0], "s, std time:", stats[1], "s",
                file=stderr,
            )
            print("arm=" + arm_name, stats[0], "\\pm", stats[1])
