from std.testing import assert_true

from fft_cpu import fft2_batched, rfft2_batched, irfft2_batched
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
    var result = test_data_2d()
    var inp = result[0].copy()
    var exp_re = result[1].copy()
    var exp_im = result[2].copy()
    var d = result[3]
    var h = result[4]
    var w = result[5]
    var n = d * h * w

    var zeros: List[Float32] = []
    for _ in range(n):
        zeros.append(0.0)

    var out = fft2_batched(inp.copy(), zeros.copy(), d, h, w, False)
    var out_re = out[0].copy()
    var out_im = out[1].copy()

    var diff_re = max_abs_diff(out_re, exp_re, n)
    var diff_im = max_abs_diff(out_im, exp_im, n)
    print("fft2 max|dRe| =", diff_re, " max|dIm| =", diff_im)
    assert_true(diff_re < 1e-3, "fft2 real mismatch")
    assert_true(diff_im < 1e-3, "fft2 imag mismatch")

    # round trip
    var back = fft2_batched(out_re.copy(), out_im.copy(), d, h, w, True)
    var back_re = back[0].copy()
    var diff_round = max_abs_diff(back_re, inp, n)
    print("fft2 round-trip max|dRe| =", diff_round)
    assert_true(diff_round < 1e-3, "fft2 round trip mismatch")

    # rfft2: the non-redundant half spectrum is the first w/2+1 columns of
    # every row of the full complex spectrum computed above.
    var w2 = w // 2 + 1
    var n2 = d * h * w2
    var rexp_re: List[Float32] = []
    var rexp_im: List[Float32] = []
    for b in range(d):
        for row in range(h):
            var base = (b * h + row) * w
            for col in range(w2):
                rexp_re.append(exp_re[base + col])
                rexp_im.append(exp_im[base + col])

    var rout = rfft2_batched(inp.copy(), d, h, w)
    var rdiff_re = max_abs_diff(rout[0].copy(), rexp_re, n2)
    var rdiff_im = max_abs_diff(rout[1].copy(), rexp_im, n2)
    print("rfft2 max|dRe| =", rdiff_re, " max|dIm| =", rdiff_im)
    assert_true(rdiff_re < 1e-3, "rfft2 real mismatch")
    assert_true(rdiff_im < 1e-3, "rfft2 imag mismatch")

    var rback = irfft2_batched(rout[0].copy(), rout[1].copy(), d, h, w)
    var rdiff_round = max_abs_diff(rback, inp, n)
    print("irfft2 round-trip max|dRe| =", rdiff_round)
    assert_true(rdiff_round < 1e-3, "irfft2 round trip mismatch")

    print("All 2D FFT CPU tests passed.")
