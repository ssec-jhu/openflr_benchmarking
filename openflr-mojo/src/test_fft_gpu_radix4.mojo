"""Targeted correctness check for the radix-4 fused stages in `fft_gpu.mojo`.

The existing `test_fft_gpu.mojo` only exercises N=8 (the small-N "else"
branch) and N=64 (whose warp-shuffle fast path has an empty tail loop, since
log2(64)=6 exactly). Neither hits the radix-4-fused tail loop that runs for
N > 64 (`fft_row_kernel`) or `half` > 64 (`rfft_row_kernel`/
`irfft_row_kernel`). This file drives every kernel across N in
{128, 256, 512, 1024, 2048} -- chosen so log2(N) (and log2(N/2)) sweeps
through every fused-pair-count / leftover-stage parity combination -- against
the arbitrary-length CPU reference in `fft_cpu.mojo`, on deterministic
pseudo-random input.
"""

from std.testing import assert_true
from std.sys import has_accelerator
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from fft_gpu import fft_row_kernel, rfft_row_kernel, irfft_row_kernel
from fft_cpu import fft_1d_inplace, rfft_1d, irfft_1d


def lcg_floats(n: Int, seed: Int) -> List[Float32]:
    """Deterministic pseudo-random floats in [-1, 1), no stdlib RNG needed."""
    var out: List[Float32] = []
    var state: UInt64 = UInt64(seed)
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var top = (state >> 40) & 0xFFFFFF
        var f = Float32(Int(top)) / Float32(0x1000000)
        out.append(f * 2.0 - 1.0)
    return out^


def max_abs_diff(a: List[Float32], b: List[Float32], n: Int) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def run_fft_case[N: Int](ctx: DeviceContext, seed: Int) raises:
    var inp = lcg_floats(N, seed)

    var exp_re = inp.copy()
    var exp_im = lcg_floats(N, seed + 1)
    for i in range(N):
        exp_im[i] = 0.0
    fft_1d_inplace(exp_re, exp_im, N, False)

    comptime layout = row_major[1, N]()
    var re_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](N)
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
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
    print("radix4 fft N =", N, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-2, "radix4 fft real mismatch")
    assert_true(diff_im < 1e-2, "radix4 fft imag mismatch")

    comptime kernel_inv = fft_row_kernel[N, type_of(layout), True]
    ctx.enqueue_function[kernel_inv](re_t, im_t, grid_dim=1, block_dim=N // 2)
    ctx.synchronize()

    var back_re: List[Float32] = []
    with re_buf.map_to_host() as h:
        var t = TileTensor(h, row_major[N]())
        for i in range(N):
            back_re.append(t[i])
    var diff_round = max_abs_diff(back_re, inp, N)
    print("radix4 fft N =", N, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-2, "radix4 fft round trip mismatch")


def run_rfft_case[N: Int](ctx: DeviceContext, seed: Int) raises:
    comptime W2 = N // 2 + 1
    var inp = lcg_floats(N, seed)

    var expected = rfft_1d(inp.copy(), N)
    var ref_re = expected[0].copy()
    var ref_im = expected[1].copy()

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

    var diff_re = max_abs_diff(out_re, ref_re, W2)
    var diff_im = max_abs_diff(out_im, ref_im, W2)
    print("radix4 rfft N =", N, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-2, "radix4 rfft real mismatch")
    assert_true(diff_im < 1e-2, "radix4 rfft imag mismatch")

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
    print("radix4 rfft N =", N, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-2, "radix4 irfft round trip mismatch")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    run_fft_case[128](ctx, 1)
    run_fft_case[256](ctx, 2)
    run_fft_case[512](ctx, 3)
    run_fft_case[1024](ctx, 4)
    run_fft_case[2048](ctx, 5)

    run_rfft_case[128](ctx, 6)
    run_rfft_case[256](ctx, 7)
    run_rfft_case[512](ctx, 8)
    run_rfft_case[1024](ctx, 9)
    run_rfft_case[2048](ctx, 10)

    print("All radix-4 GPU tests passed.")
