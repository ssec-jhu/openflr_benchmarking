"""CPU reference implementation of a radix-2 Cooley-Tukey FFT.

Operates on separate real/imaginary `List[Float32]` buffers in place.
Only supports power-of-two lengths.
"""

from std.math import cos, sin

comptime PI: Float32 = 3.14159265358979323846


def zeros(n: Int) -> List[Float32]:
    var out: List[Float32] = []
    for _ in range(n):
        out.append(0.0)
    return out^


def bit_reverse(x: Int, bits: Int) -> Int:
    var result = 0
    var v = x
    for _ in range(bits):
        result = (result << 1) | (v & 1)
        v >>= 1
    return result


def ilog2(n: Int) -> Int:
    var v = n
    var bits = 0
    while v > 1:
        v >>= 1
        bits += 1
    return bits


def fft_1d_inplace(mut re: List[Float32], mut im: List[Float32], n: Int, invert: Bool):
    """In-place radix-2 DIT FFT (or inverse, unnormalized-then-scaled) over
    `n` elements starting at index 0 of `re`/`im`. `n` must be a power of 2.
    """
    var log2n = ilog2(n)

    for i in range(n):
        var j = bit_reverse(i, log2n)
        if j > i:
            var tr = re[i]
            var ti = im[i]
            re[i] = re[j]
            im[i] = im[j]
            re[j] = tr
            im[j] = ti

    var sign: Float32 = 1.0 if invert else -1.0
    var size = 2
    while size <= n:
        var half = size // 2
        for start in range(0, n, size):
            for k in range(half):
                var angle = sign * 2.0 * PI * Float32(k) / Float32(size)
                var wr = cos(angle)
                var wi = sin(angle)
                var i0 = start + k
                var i1 = i0 + half
                var xr = re[i1]
                var xi = im[i1]
                var tr = xr * wr - xi * wi
                var ti = xr * wi + xi * wr
                var ur = re[i0]
                var ui = im[i0]
                re[i0] = ur + tr
                im[i0] = ui + ti
                re[i1] = ur - tr
                im[i1] = ui - ti
        size *= 2

    if invert:
        for i in range(n):
            re[i] /= Float32(n)
            im[i] /= Float32(n)


def rfft_1d(x: List[Float32], n: Int) -> Tuple[List[Float32], List[Float32]]:
    """Real-input FFT of length `n` (power of two, `n` >= 2) computed via a
    single length-`n/2` complex FFT (the standard "pack two reals into one
    complex FFT" trick): even/odd samples of `x` are packed into one complex
    signal `z`, `Z = FFT(z)` is computed, and `Z` is unpacked into the
    non-redundant half spectrum `X[0 .. n/2]` (length `n/2 + 1`) using the
    even/odd DFT symmetry. Roughly half the arithmetic of a full-length
    complex FFT with a zeroed imaginary part.
    """
    var half = n // 2
    var w2 = half + 1

    var zre: List[Float32] = []
    var zim: List[Float32] = []
    for i in range(half):
        zre.append(x[2 * i])
        zim.append(x[2 * i + 1])
    fft_1d_inplace(zre, zim, half, False)

    var out_re = zeros(w2)
    var out_im = zeros(w2)
    for k in range(half):
        var km = (half - k) % half
        var zr_km = zre[km]
        var zi_km = zim[km]
        var xe_re = (zre[k] + zr_km) * 0.5
        var xe_im = (zim[k] - zi_km) * 0.5
        var xo_re = (zim[k] + zi_km) * 0.5
        var xo_im = -(zre[k] - zr_km) * 0.5
        var angle = -2.0 * PI * Float32(k) / Float32(n)
        var wr = cos(angle)
        var wi = sin(angle)
        out_re[k] = xe_re + (wr * xo_re - wi * xo_im)
        out_im[k] = xe_im + (wr * xo_im + wi * xo_re)

    out_re[half] = zre[0] - zim[0]
    out_im[half] = 0.0

    return out_re^, out_im^


def irfft_1d(re: List[Float32], im: List[Float32], n: Int) -> List[Float32]:
    """Inverse of `rfft_1d`: given the non-redundant half spectrum (length
    `n/2 + 1`) of a real length-`n` signal, reconstructs the length-`n` real
    signal via a single length-`n/2` inverse complex FFT.
    """
    var half = n // 2

    var xe_re = zeros(half)
    var xe_im = zeros(half)
    var xo_re = zeros(half)
    var xo_im = zeros(half)
    for k in range(half):
        var idx = half - k
        var ar = re[k]
        var ai = im[k]
        var br = re[idx]
        var bi = -im[idx]
        var er = (ar + br) * 0.5
        var ei = (ai + bi) * 0.5
        var dr = (ar - br) * 0.5
        var di = (ai - bi) * 0.5
        var angle = 2.0 * PI * Float32(k) / Float32(n)
        var wr = cos(angle)
        var wi = sin(angle)
        xe_re[k] = er
        xe_im[k] = ei
        xo_re[k] = dr * wr - di * wi
        xo_im[k] = dr * wi + di * wr

    var zre = zeros(half)
    var zim = zeros(half)
    for k in range(half):
        zre[k] = xe_re[k] - xo_im[k]
        zim[k] = xe_im[k] + xo_re[k]
    fft_1d_inplace(zre, zim, half, True)

    var out = zeros(n)
    for i in range(half):
        out[2 * i] = zre[i]
        out[2 * i + 1] = zim[i]
    return out^


def rfft_rows(x: List[Float32], n_rows: Int, row_len: Int) -> Tuple[List[Float32], List[Float32]]:
    """Runs an independent real-input FFT (`rfft_1d`) over each contiguous
    row of length `row_len` in a flat (n_rows, row_len) real buffer,
    producing a flat (n_rows, row_len/2+1) complex buffer pair.
    """
    var w2 = row_len // 2 + 1
    var out_re = zeros(n_rows * w2)
    var out_im = zeros(n_rows * w2)
    for r in range(n_rows):
        var row: List[Float32] = []
        var base = r * row_len
        for c in range(row_len):
            row.append(x[base + c])
        var res = rfft_1d(row, row_len)
        var obase = r * w2
        for c in range(w2):
            out_re[obase + c] = res[0][c]
            out_im[obase + c] = res[1][c]
    return out_re^, out_im^


def irfft_rows(re: List[Float32], im: List[Float32], n_rows: Int, row_len: Int) -> List[Float32]:
    """Inverse of `rfft_rows`: (n_rows, row_len/2+1) complex -> (n_rows,
    row_len) real."""
    var w2 = row_len // 2 + 1
    var out = zeros(n_rows * row_len)
    for r in range(n_rows):
        var row_re: List[Float32] = []
        var row_im: List[Float32] = []
        var base = r * w2
        for c in range(w2):
            row_re.append(re[base + c])
            row_im.append(im[base + c])
        var res = irfft_1d(row_re, row_im, row_len)
        var obase = r * row_len
        for c in range(row_len):
            out[obase + c] = res[c]
    return out^


def fft_rows_inplace(mut re: List[Float32], mut im: List[Float32], n_rows: Int, row_len: Int, invert: Bool):
    """Runs an independent 1D FFT over each contiguous row of length
    `row_len` in a flat (n_rows, row_len) buffer.
    """
    for r in range(n_rows):
        var row_re: List[Float32] = []
        var row_im: List[Float32] = []
        var base = r * row_len
        for c in range(row_len):
            row_re.append(re[base + c])
            row_im.append(im[base + c])
        fft_1d_inplace(row_re, row_im, row_len, invert)
        for c in range(row_len):
            re[base + c] = row_re[c]
            im[base + c] = row_im[c]


def transpose_batched(
    re: List[Float32], im: List[Float32], batch: Int, h: Int, w: Int
) -> Tuple[List[Float32], List[Float32]]:
    """Transposes the last two axes of a (batch, h, w) flat buffer into a
    (batch, w, h) flat buffer.
    """
    var out_re: List[Float32] = []
    var out_im: List[Float32] = []
    var total = batch * h * w
    for _ in range(total):
        out_re.append(0.0)
        out_im.append(0.0)

    for b in range(batch):
        var in_base = b * h * w
        var out_base = b * w * h
        for row in range(h):
            for col in range(w):
                var src = in_base + row * w + col
                var dst = out_base + col * h + row
                out_re[dst] = re[src]
                out_im[dst] = im[src]

    return out_re^, out_im^


def fft2_batched(
    re: List[Float32], im: List[Float32], batch: Int, h: Int, w: Int, invert: Bool
) -> Tuple[List[Float32], List[Float32]]:
    """2D FFT (or inverse) over the last two axes of a (batch, h, w) flat
    buffer of complex data (separate real/imag parts). `h` and `w` must be
    powers of 2.
    """
    var cur_re = re.copy()
    var cur_im = im.copy()

    # Transform along the last axis (width) for every row.
    fft_rows_inplace(cur_re, cur_im, batch * h, w, invert)

    # Transpose so height becomes the last axis, transform, transpose back.
    var t = transpose_batched(cur_re, cur_im, batch, h, w)
    var t_re = t[0].copy()
    var t_im = t[1].copy()
    fft_rows_inplace(t_re, t_im, batch * w, h, invert)

    var back = transpose_batched(t_re, t_im, batch, w, h)
    return back^


def rfft2_batched(
    x: List[Float32], batch: Int, h: Int, w: Int
) -> Tuple[List[Float32], List[Float32]]:
    """Real-input 2D FFT over the last two axes of a (batch, h, w) real
    buffer: a real-optimized FFT (`rfft_1d`) along the width axis, then a
    full complex FFT along the height axis -- mirrors `numpy.fft.rfft2`.
    `h` and `w` must be powers of 2. Returns the non-redundant half spectrum,
    a (batch, h, w/2+1) complex buffer pair.
    """
    var w2 = w // 2 + 1
    var rows = rfft_rows(x, batch * h, w)

    var t = transpose_batched(rows[0].copy(), rows[1].copy(), batch, h, w2)
    var t_re = t[0].copy()
    var t_im = t[1].copy()
    fft_rows_inplace(t_re, t_im, batch * w2, h, False)

    var back = transpose_batched(t_re, t_im, batch, w2, h)
    return back^


def irfft2_batched(
    re: List[Float32], im: List[Float32], batch: Int, h: Int, w: Int
) -> List[Float32]:
    """Inverse of `rfft2_batched`: (batch, h, w/2+1) non-redundant complex
    half spectrum -> (batch, h, w) real -- mirrors `numpy.fft.irfft2`. `h`
    and `w` must be powers of 2.
    """
    var w2 = w // 2 + 1
    var t = transpose_batched(re, im, batch, h, w2)
    var t_re = t[0].copy()
    var t_im = t[1].copy()
    fft_rows_inplace(t_re, t_im, batch * w2, h, True)

    var back = transpose_batched(t_re, t_im, batch, w2, h)
    var back_re = back[0].copy()
    var back_im = back[1].copy()
    return irfft_rows(back_re, back_im, batch * h, w)
