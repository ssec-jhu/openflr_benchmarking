from std.testing import assert_true
from std.sys import has_accelerator
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from fft_gpu import fft2_batched_gpu, rfft2_batched_gpu, irfft2_batched_gpu
from test_fft2_data import test_data_2d


def max_abs_diff(a: List[Float32], b: List[Float32], n: Int) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var result = test_data_2d()
    var inp = result[0].copy()
    var exp_re = result[1].copy()
    var exp_im = result[2].copy()
    comptime D = 2
    comptime H = 4
    comptime W = 8
    comptime N = D * H * W

    var re_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var t_re = ctx.enqueue_create_buffer[DType.float32](N)
    var t_im = ctx.enqueue_create_buffer[DType.float32](N)
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            t[i] = inp[i]
    im_buf.enqueue_fill(0.0)

    fft2_batched_gpu[D, H, W, 4, False](ctx, re_buf, im_buf, t_re, t_im)
    ctx.synchronize()

    var out_re: List[Float32] = []
    var out_im: List[Float32] = []
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            out_re.append(t[i])
    with im_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            out_im.append(t[i])

    var diff_re = max_abs_diff(out_re, exp_re, N)
    var diff_im = max_abs_diff(out_im, exp_im, N)
    print("GPU fft2 max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "gpu fft2 real mismatch")
    assert_true(diff_im < 1e-3, "gpu fft2 imag mismatch")

    fft2_batched_gpu[D, H, W, 4, True](ctx, re_buf, im_buf, t_re, t_im)
    ctx.synchronize()

    var back_re: List[Float32] = []
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            back_re.append(t[i])
    var diff_round = max_abs_diff(back_re, inp, N)
    print("GPU fft2 round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "gpu fft2 round trip mismatch")

    # rfft2: the non-redundant half spectrum is the first W/2+1 columns of
    # every row of the full complex spectrum computed above.
    comptime W2 = W // 2 + 1
    comptime N2 = D * H * W2
    var rexp_re: List[Float32] = []
    var rexp_im: List[Float32] = []
    for b in range(D):
        for row in range(H):
            var base = (b * H + row) * W
            for col in range(W2):
                rexp_re.append(exp_re[base + col])
                rexp_im.append(exp_im[base + col])

    var rx_buf = ctx.enqueue_create_buffer[DType.float32](N)
    with rx_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            t[i] = inp[i]
    var r_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var r_im = ctx.enqueue_create_buffer[DType.float32](N2)
    var rt_re = ctx.enqueue_create_buffer[DType.float32](N2)
    var rt_im = ctx.enqueue_create_buffer[DType.float32](N2)

    rfft2_batched_gpu[D, H, W, 4](ctx, rx_buf, r_re, r_im, rt_re, rt_im)
    ctx.synchronize()

    var rout_re: List[Float32] = []
    var rout_im: List[Float32] = []
    with r_re.map_to_host() as h:
        var t = TileTensor(h, row_major[N2]())
        for i in range(N2):
            rout_re.append(t[i])
    with r_im.map_to_host() as h:
        var t = TileTensor(h, row_major[N2]())
        for i in range(N2):
            rout_im.append(t[i])

    var rdiff_re = max_abs_diff(rout_re, rexp_re, N2)
    var rdiff_im = max_abs_diff(rout_im, rexp_im, N2)
    print("GPU rfft2 max|dRe| =", rdiff_re, " max|dIm| =", rdiff_im)
    assert_true(rdiff_re < 1e-3, "gpu rfft2 real mismatch")
    assert_true(rdiff_im < 1e-3, "gpu rfft2 imag mismatch")

    var rback_buf = ctx.enqueue_create_buffer[DType.float32](N)
    irfft2_batched_gpu[D, H, W, 4](ctx, r_re, r_im, rback_buf, rt_re, rt_im)
    ctx.synchronize()

    var rback: List[Float32] = []
    with rback_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            rback.append(t[i])
    var rdiff_round = max_abs_diff(rback, inp, N)
    print("GPU irfft2 round-trip max|dRe| =", rdiff_round)
    assert_true(rdiff_round < 1e-3, "gpu irfft2 round trip mismatch")

    print("All 2D FFT GPU tests passed.")
