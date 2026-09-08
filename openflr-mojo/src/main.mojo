"""CLI entrypoint: runs and benchmarks one OpenFLR Richardson-Lucy GPU
iteration on the real light-field dataset, mirroring the timing harness in
../../main.py (backend "mojo"). Also supports writing the output of a single
step to a file for offline correctness comparison against the numpy
reference.

Usage:
    mojo run main.mojo -- v1 20        # benchmark v1, 20 iterations
    mojo run main.mojo -- v2 20        # benchmark v2, 20 iterations
    mojo run main.mojo -- v1 1 --dump out.bin   # single step, dump result
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
        else:
            n_iters = Int(a)
        i += 1

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

    if version == "v1":
        run_v1_step_gpu[D, H, W, TILE](
            ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
        )
    else:
        run_v2_step_gpu[D, H, W, TILE](
            ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v2_re, psft_v2_im, out_buf, scratch
        )
    ctx.synchronize()

    if dump_path != "":
        var result = download[N](ctx, out_buf)
        write_floats(dump_path, result, N)
        print("wrote", dump_path)
        return

    var times: List[Float64] = []
    for _ in range(n_iters):
        var start = perf_counter_ns()
        if version == "v1":
            run_v1_step_gpu[D, H, W, TILE](
                ctx, data_buf, image_buf, psf_fft_re, psf_fft_im, psft_v1_re, psft_v1_im, out_buf, scratch
            )
        else:
            run_v2_step_gpu[D, H, W, TILE](
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
        "backend: mojo, version:", version,
        ", mean time:", stats[0], "s, std time:", stats[1], "s",
        file=stderr,
    )
    print(stats[0], "\\pm", stats[1])
