from std.testing import assert_true

from fft_cpu import rfft2_batched
from openflr_cpu import flip_hw, shift_hw, zeros, run_v1_step_cpu, run_v2_step_cpu
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


def main() raises:
    var result = test_data_openflr()
    var psf = result[0].copy()
    var image = result[1].copy()
    var expected_v1 = result[2].copy()
    var expected_v2 = result[3].copy()
    var d = result[4]
    var h = result[5]
    var w = result[6]
    var hw = h * w
    var n = d * hw

    # data = 0.5 everywhere
    var data = zeros(n)
    for i in range(n):
        data[i] = 0.5

    var psf_fft = rfft2_batched(psf.copy(), d, h, w)

    var psf_flipped = flip_hw(psf.copy(), d, h, w)
    var psft_fft_v1 = rfft2_batched(psf_flipped.copy(), d, h, w)

    var psf_flipped_shifted = shift_hw(psf_flipped.copy(), d, h, w)
    var psft_fft_v2 = rfft2_batched(psf_flipped_shifted.copy(), d, h, w)

    var out_v1 = run_v1_step_cpu(
        data.copy(), image.copy(),
        psf_fft[0].copy(), psf_fft[1].copy(),
        psft_fft_v1[0].copy(), psft_fft_v1[1].copy(),
        d, h, w,
    )
    var diff_v1 = max_abs_diff(out_v1, expected_v1, n)
    print("v1 max|diff| =", diff_v1)
    assert_true(diff_v1 < 1e-3, "v1 mismatch vs numpy reference")

    var out_v2 = run_v2_step_cpu(
        data.copy(), image.copy(),
        psf_fft[0].copy(), psf_fft[1].copy(),
        psft_fft_v2[0].copy(), psft_fft_v2[1].copy(),
        d, h, w,
    )
    var diff_v2 = max_abs_diff(out_v2, expected_v2, n)
    print("v2 max|diff| =", diff_v2)
    assert_true(diff_v2 < 1e-3, "v2 mismatch vs numpy reference")

    print("OpenFLR CPU v1/v2 tests passed.")
