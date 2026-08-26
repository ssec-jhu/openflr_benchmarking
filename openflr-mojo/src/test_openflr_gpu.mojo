from std.testing import assert_true
from std.sys import has_accelerator
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext, DeviceBuffer

from fft_gpu import rfft2_batched_gpu
from openflr_cpu import flip_hw, shift_hw
from openflr_gpu import run_v1_step_gpu, run_v2_step_gpu, OpenFlrScratch
from test_openflr_data import test_data_openflr


def max_abs_diff(a: List[Float32], b: List[Float32], n: Int) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def upload[N: Int](ctx: DeviceContext, data: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](N)
    with buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            t[i] = data[i]
    return buf^


def download[N: Int](ctx: DeviceContext, buf: DeviceBuffer[DType.float32]) raises -> List[Float32]:
    var out: List[Float32] = []
    with buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            out.append(t[i])
    return out^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var result = test_data_openflr()
    var psf = result[0].copy()
    var image = result[1].copy()
    var expected_v1 = result[2].copy()
    var expected_v2 = result[3].copy()
    comptime D = 3
    comptime H = 8
    comptime W = 8
    comptime W2 = W // 2 + 1
    comptime HW = H * W
    comptime HW2 = H * W2
    comptime N = D * HW
    comptime N2 = D * HW2
    comptime TILE = 4

    var data: List[Float32] = []
    for _ in range(N):
        data.append(0.5)

    var psf_flipped = flip_hw(psf.copy(), D, H, W)
    var psf_flipped_shifted = shift_hw(psf_flipped.copy(), D, H, W)

    var psf_buf = upload[N](ctx, psf.copy())
    var psf_fft_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psf_fft_im = ctx.enqueue_create_buffer[DType.float32](N2)
    var t_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var t_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu[D, H, W, TILE](ctx, psf_buf, psf_fft_re, psf_fft_im, t_re, t_im)

    var psft_v1_buf = upload[N](ctx, psf_flipped.copy())
    var psft_fft_v1_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psft_fft_v1_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu[D, H, W, TILE](ctx, psft_v1_buf, psft_fft_v1_re, psft_fft_v1_im, t_re, t_im)

    var psft_v2_buf = upload[N](ctx, psf_flipped_shifted.copy())
    var psft_fft_v2_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var psft_fft_v2_im = ctx.enqueue_create_buffer[DType.float32](N2)
    rfft2_batched_gpu[D, H, W, TILE](ctx, psft_v2_buf, psft_fft_v2_re, psft_fft_v2_im, t_re, t_im)

    var data_buf_v1 = upload[N](ctx, data.copy())
    var image_buf = upload[HW](ctx, image.copy())
    var out_buf_v1 = ctx.enqueue_create_buffer[DType.float32](N)
    var scratch = OpenFlrScratch[D, H, W](ctx)

    run_v1_step_gpu[D, H, W, TILE](
        ctx, data_buf_v1, image_buf,
        psf_fft_re, psf_fft_im, psft_fft_v1_re, psft_fft_v1_im,
        out_buf_v1, scratch,
    )
    ctx.synchronize()
    var out_v1 = download[N](ctx, out_buf_v1)
    var diff_v1 = max_abs_diff(out_v1, expected_v1, N)
    print("GPU v1 max|diff| =", diff_v1)
    assert_true(diff_v1 < 1e-2, "gpu v1 mismatch vs numpy reference")

    var data_buf_v2 = upload[N](ctx, data.copy())
    var out_buf_v2 = ctx.enqueue_create_buffer[DType.float32](N)
    run_v2_step_gpu[D, H, W, TILE](
        ctx, data_buf_v2, image_buf,
        psf_fft_re, psf_fft_im, psft_fft_v2_re, psft_fft_v2_im,
        out_buf_v2, scratch,
    )
    ctx.synchronize()
    var out_v2 = download[N](ctx, out_buf_v2)
    var diff_v2 = max_abs_diff(out_v2, expected_v2, N)
    print("GPU v2 max|diff| =", diff_v2)
    assert_true(diff_v2 < 1e-2, "gpu v2 mismatch vs numpy reference")

    print("OpenFLR GPU v1/v2 tests passed.")
