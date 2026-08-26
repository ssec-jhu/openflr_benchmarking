from std.testing import assert_true
from std.sys import has_accelerator
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from fft_gpu import fft_row_kernel, rfft_row_kernel, irfft_row_kernel
from test_fft_data import test_data_8, test_data_64


def max_abs_diff(a: List[Float32], b: List[Float32], n: Int) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def run_case[N: Int](
    ctx: DeviceContext,
    inp: List[Float32], exp_re: List[Float32], exp_im: List[Float32],
) raises:
    comptime layout = row_major[1, N]()

    var re_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](N)
    with re_buf.map_to_host() as h_re:
        var t = TileTensor(h_re, row_major[N]())
        for i in range(N):
            t[i] = inp[i]
    im_buf.enqueue_fill(0.0)

    var re_t = TileTensor(re_buf, layout)
    var im_t = TileTensor(im_buf, layout)

    comptime kernel = fft_row_kernel[N, type_of(layout), False]
    ctx.enqueue_function[kernel](re_t, im_t, grid_dim=1, block_dim=N // 2)
    ctx.synchronize()

    var out_re: List[Float32] = []
    var out_im: List[Float32] = []
    with re_buf.map_to_host() as h_re:
        var t = TileTensor(h_re, row_major[N]())
        for i in range(N):
            out_re.append(t[i])
    with im_buf.map_to_host() as h_im:
        var t = TileTensor(h_im, row_major[N]())
        for i in range(N):
            out_im.append(t[i])

    var diff_re = max_abs_diff(out_re, exp_re, N)
    var diff_im = max_abs_diff(out_im, exp_im, N)
    print("GPU N =", N, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "gpu real mismatch")
    assert_true(diff_im < 1e-3, "gpu imag mismatch")

    # round trip via inverse kernel
    comptime kernel_inv = fft_row_kernel[N, type_of(layout), True]
    ctx.enqueue_function[kernel_inv](re_t, im_t, grid_dim=1, block_dim=N // 2)
    ctx.synchronize()

    var back_re: List[Float32] = []
    with re_buf.map_to_host() as h_re:
        var t = TileTensor(h_re, row_major[N]())
        for i in range(N):
            back_re.append(t[i])
    var diff_round = max_abs_diff(back_re, inp, N)
    print("GPU N =", N, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "gpu round trip mismatch")


def run_rfft_case[N: Int](
    ctx: DeviceContext,
    inp: List[Float32], exp_re: List[Float32], exp_im: List[Float32],
) raises:
    """`exp_re`/`exp_im` are the full length-`N` spectrum; the GPU rfft
    kernel only needs to match its non-redundant half, the first `N/2+1`
    entries."""
    comptime W2 = N // 2 + 1
    comptime in_layout = row_major[1, N]()
    comptime out_layout = row_major[1, W2]()

    var x_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var re_buf = ctx.enqueue_create_buffer[DType.float32](W2)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](W2)
    with x_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            t[i] = inp[i]

    var x_t = TileTensor(x_buf, in_layout)
    var re_t = TileTensor(re_buf, out_layout)
    var im_t = TileTensor(im_buf, out_layout)

    comptime kernel = rfft_row_kernel[N, type_of(in_layout), type_of(out_layout)]
    ctx.enqueue_function[kernel](x_t, re_t, im_t, grid_dim=1, block_dim=N // 4)
    ctx.synchronize()

    var out_re: List[Float32] = []
    var out_im: List[Float32] = []
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[W2]())
        for i in range(W2):
            out_re.append(t[i])
    with im_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[W2]())
        for i in range(W2):
            out_im.append(t[i])

    var diff_re = max_abs_diff(out_re, exp_re, W2)
    var diff_im = max_abs_diff(out_im, exp_im, W2)
    print("GPU rfft N =", N, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "gpu rfft real mismatch")
    assert_true(diff_im < 1e-3, "gpu rfft imag mismatch")

    var back_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var back_t = TileTensor(back_buf, in_layout)
    comptime kernel_inv = irfft_row_kernel[N, type_of(out_layout), type_of(in_layout)]
    ctx.enqueue_function[kernel_inv](re_t, im_t, back_t, grid_dim=1, block_dim=N // 4)
    ctx.synchronize()

    var back: List[Float32] = []
    with back_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            back.append(t[i])
    var diff_round = max_abs_diff(back, inp, N)
    print("GPU rfft N =", N, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "gpu irfft round trip mismatch")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var d8 = test_data_8()
    run_case[8](ctx, d8[0].copy(), d8[1].copy(), d8[2].copy())
    run_rfft_case[8](ctx, d8[0].copy(), d8[1].copy(), d8[2].copy())

    var d64 = test_data_64()
    run_case[64](ctx, d64[0].copy(), d64[1].copy(), d64[2].copy())
    run_rfft_case[64](ctx, d64[0].copy(), d64[1].copy(), d64[2].copy())

    print("All FFT GPU tests passed.")
