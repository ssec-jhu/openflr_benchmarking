"""CPU reference implementation of one OpenFLR Richardson-Lucy iteration
(v1 and v2), using the real-input-optimized radix-2 rfft2/irfft2 from
`fft_cpu`.

All tensors are flattened row-major (depth, height, width) `List[Float32]`
buffers. This mirrors `run_v1_step`/`run_v2_step` in the Python reference,
including its use of `numpy.fft.rfft2`/`irfft2` (real input, non-redundant
half spectrum) rather than a full complex FFT.
"""

from fft_cpu import rfft2_batched, irfft2_batched


def zeros(n: Int) -> List[Float32]:
    var out: List[Float32] = []
    for _ in range(n):
        out.append(0.0)
    return out^


def flip_hw(x: List[Float32], d: Int, h: Int, w: Int) -> List[Float32]:
    var out = zeros(d * h * w)
    for b in range(d):
        var base = b * h * w
        for row in range(h):
            for col in range(w):
                var src = base + row * w + col
                var dst = base + (h - 1 - row) * w + (w - 1 - col)
                out[dst] = x[src]
    return out^


def shift_hw(x: List[Float32], d: Int, h: Int, w: Int) -> List[Float32]:
    """fftshift/ifftshift over the last two axes (equivalent for even h, w)."""
    var out = zeros(d * h * w)
    var hh = h // 2
    var hw = w // 2
    for b in range(d):
        var base = b * h * w
        for row in range(h):
            for col in range(w):
                var src = base + row * w + col
                var dst_row = (row + hh) % h
                var dst_col = (col + hw) % w
                var dst = base + dst_row * w + dst_col
                out[dst] = x[src]
    return out^


def complex_mul(
    a_re: List[Float32], a_im: List[Float32],
    b_re: List[Float32], b_im: List[Float32],
    n: Int,
) -> Tuple[List[Float32], List[Float32]]:
    var out_re = zeros(n)
    var out_im = zeros(n)
    for i in range(n):
        out_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i]
        out_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i]
    return out_re^, out_im^


def complex_mul_broadcast_depth(
    a_re: List[Float32], a_im: List[Float32],  # (1, h, w)
    b_re: List[Float32], b_im: List[Float32],  # (d, h, w)
    d: Int, h: Int, w: Int,
) -> Tuple[List[Float32], List[Float32]]:
    var hw = h * w
    var out_re = zeros(d * hw)
    var out_im = zeros(d * hw)
    for b in range(d):
        var base = b * hw
        for i in range(hw):
            var ar = a_re[i]
            var ai = a_im[i]
            var br = b_re[base + i]
            var bi = b_im[base + i]
            out_re[base + i] = ar * br - ai * bi
            out_im[base + i] = ar * bi + ai * br
    return out_re^, out_im^


def sum_over_depth(
    x_re: List[Float32], x_im: List[Float32], d: Int, hw: Int
) -> Tuple[List[Float32], List[Float32]]:
    var out_re = zeros(hw)
    var out_im = zeros(hw)
    for b in range(d):
        var base = b * hw
        for i in range(hw):
            out_re[i] += x_re[base + i]
            out_im[i] += x_im[base + i]
    return out_re^, out_im^


def sum_over_depth_real(x: List[Float32], d: Int, hw: Int) -> List[Float32]:
    var out = zeros(hw)
    for b in range(d):
        var base = b * hw
        for i in range(hw):
            out[i] += x[base + i]
    return out^


def elementwise_div_broadcast(
    a: List[Float32], b: List[Float32], n: Int
) -> List[Float32]:
    var out = zeros(n)
    for i in range(n):
        out[i] = a[i] / b[i]
    return out^


def elementwise_mul_broadcast_depth(
    a: List[Float32], b: List[Float32], d: Int, hw: Int
) -> List[Float32]:
    """a: (d, hw), b: (d, hw). Plain elementwise (no broadcast needed here
    since both operands already carry the full depth dimension)."""
    var n = d * hw
    var out = zeros(n)
    for i in range(n):
        out[i] = a[i] * b[i]
    return out^


def run_v1_step_cpu(
    data: List[Float32], image: List[Float32],
    psf_fft_re: List[Float32], psf_fft_im: List[Float32],
    psft_fft_re: List[Float32], psft_fft_im: List[Float32],
    d: Int, h: Int, w: Int,
) -> List[Float32]:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v1 step -- `psf_fft_*`/
    `psft_fft_*` are the `rfft2_batched` half spectrum of the (real) PSF,
    shape (d, h, w/2+1)."""
    var hw = h * w
    var w2 = w // 2 + 1
    var hw2 = h * w2
    var data_fft = rfft2_batched(data.copy(), d, h, w)

    var prod = complex_mul(psf_fft_re, psf_fft_im, data_fft[0].copy(), data_fft[1].copy(), d * hw2)
    var conv = irfft2_batched(prod[0].copy(), prod[1].copy(), d, h, w)

    var denom = sum_over_depth_real(conv, d, hw)
    var img_err = elementwise_div_broadcast(image, denom, hw)

    var err_fft = rfft2_batched(img_err.copy(), 1, h, w)

    var prod2 = complex_mul_broadcast_depth(
        err_fft[0].copy(), err_fft[1].copy(), psft_fft_re, psft_fft_im, d, h, w2
    )
    var back = irfft2_batched(prod2[0].copy(), prod2[1].copy(), d, h, w)
    var back_shifted = shift_hw(back, d, h, w)

    return elementwise_mul_broadcast_depth(data, back_shifted, d, hw)


def run_v2_step_cpu(
    data: List[Float32], image: List[Float32],
    psf_fft_re: List[Float32], psf_fft_im: List[Float32],
    psft_fft_re: List[Float32], psft_fft_im: List[Float32],
    d: Int, h: Int, w: Int,
) -> List[Float32]:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v2 step -- `psf_fft_*`/
    `psft_fft_*` are the `rfft2_batched` half spectrum of the (real) PSF,
    shape (d, h, w/2+1)."""
    var hw = h * w
    var w2 = w // 2 + 1
    var hw2 = h * w2
    var data_fft = rfft2_batched(data.copy(), d, h, w)

    var prod = complex_mul(psf_fft_re, psf_fft_im, data_fft[0].copy(), data_fft[1].copy(), d * hw2)
    var freq_sum = sum_over_depth(prod[0].copy(), prod[1].copy(), d, hw2)

    var denom = irfft2_batched(freq_sum[0].copy(), freq_sum[1].copy(), 1, h, w)
    var img_err = elementwise_div_broadcast(image, denom, hw)

    var err_fft = rfft2_batched(img_err.copy(), 1, h, w)

    var prod2 = complex_mul_broadcast_depth(
        err_fft[0].copy(), err_fft[1].copy(), psft_fft_re, psft_fft_im, d, h, w2
    )
    var back = irfft2_batched(prod2[0].copy(), prod2[1].copy(), d, h, w)

    return elementwise_mul_broadcast_depth(data, back, d, hw)
