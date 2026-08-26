from std.testing import assert_true

from fft_cpu import fft_1d_inplace, rfft_1d, irfft_1d
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


def check_forward(n: Int, inp: List[Float32], exp_re: List[Float32], exp_im: List[Float32]) raises:
    var re = inp.copy()
    var im: List[Float32] = []
    for _ in range(n):
        im.append(0.0)

    fft_1d_inplace(re, im, n, False)

    var diff_re = max_abs_diff(re, exp_re, n)
    var diff_im = max_abs_diff(im, exp_im, n)
    print("N =", n, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "real part mismatch")
    assert_true(diff_im < 1e-3, "imag part mismatch")

    # round trip: ifft(fft(x)) == x
    var re2 = re.copy()
    var im2 = im.copy()
    fft_1d_inplace(re2, im2, n, True)
    var diff_round = max_abs_diff(re2, inp, n)
    print("N =", n, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "round trip mismatch")


def check_rfft(n: Int, inp: List[Float32], exp_re: List[Float32], exp_im: List[Float32]) raises:
    """`exp_re`/`exp_im` are the full length-`n` spectrum; `rfft_1d` only
    needs to match its non-redundant half, the first `n/2+1` entries."""
    var w2 = n // 2 + 1
    var res = rfft_1d(inp.copy(), n)

    var diff_re = max_abs_diff(res[0].copy(), exp_re, w2)
    var diff_im = max_abs_diff(res[1].copy(), exp_im, w2)
    print("rfft N =", n, " max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "rfft real part mismatch")
    assert_true(diff_im < 1e-3, "rfft imag part mismatch")

    var back = irfft_1d(res[0].copy(), res[1].copy(), n)
    var diff_round = max_abs_diff(back, inp, n)
    print("rfft N =", n, " round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "irfft round trip mismatch")


def main() raises:
    var result8 = test_data_8()
    check_forward(8, result8[0].copy(), result8[1].copy(), result8[2].copy())
    check_rfft(8, result8[0].copy(), result8[1].copy(), result8[2].copy())

    var result64 = test_data_64()
    check_forward(64, result64[0].copy(), result64[1].copy(), result64[2].copy())
    check_rfft(64, result64[0].copy(), result64[1].copy(), result64[2].copy())

    print("All FFT CPU tests passed.")
