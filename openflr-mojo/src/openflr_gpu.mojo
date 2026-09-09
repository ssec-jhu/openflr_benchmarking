"""GPU implementation of one OpenFLR Richardson-Lucy iteration (v1 and v2),
built on top of the radix-2 FFT kernels in `fft_gpu`.
"""

from std.math import ceildiv
from std.gpu import global_idx
from layout import TileTensor, TensorLayout, row_major
from max.gpu.host import DeviceContext, DeviceBuffer

from fft_gpu import (
    rfft2_batched_gpu_t,
    rfft2_batched_gpu_div_t,
    irfft2_batched_gpu_t,
    irfft2_batched_gpu_cmul_broadcast_mul_t,
    rfft_w_transposed_gpu,
    irfft_w_from_transposed_gpu,
    fft_col_cmul_ifft_gpu,
)

comptime BLOCK_1D = 256


struct OpenFlrScratch[D: Int, H: Int, W: Int](Movable):
    """Pre-allocated device scratch for one `run_v1_step_gpu`/
    `run_v2_step_gpu` call, sized once and reused across benchmark
    iterations instead of allocating fresh buffers every call. Frequency-
    domain buffers are sized for the real-input-optimized rfft2/irfft2 half
    spectrum (`W2 = W//2+1` instead of `W`)."""
    var data_fft_re: DeviceBuffer[DType.float32]
    var data_fft_im: DeviceBuffer[DType.float32]
    var prod_re: DeviceBuffer[DType.float32]
    var prod_im: DeviceBuffer[DType.float32]
    var conv: DeviceBuffer[DType.float32]
    var reduce_re: DeviceBuffer[DType.float32]
    var reduce_im: DeviceBuffer[DType.float32]
    var denom: DeviceBuffer[DType.float32]
    var err_fft_re: DeviceBuffer[DType.float32]
    var err_fft_im: DeviceBuffer[DType.float32]
    var prod2_re: DeviceBuffer[DType.float32]
    var prod2_im: DeviceBuffer[DType.float32]
    var t_re_dhw2: DeviceBuffer[DType.float32]
    var t_im_dhw2: DeviceBuffer[DType.float32]
    var t_re_hw2: DeviceBuffer[DType.float32]
    var t_im_hw2: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime W2 = Self.W // 2 + 1
        comptime HW = Self.H * Self.W
        comptime HW2 = Self.H * W2
        comptime DHW = Self.D * HW
        comptime DHW2 = Self.D * HW2
        self.data_fft_re = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.data_fft_im = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.prod_re = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.prod_im = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.conv = ctx.enqueue_create_buffer[DType.float32](DHW)
        self.reduce_re = ctx.enqueue_create_buffer[DType.float32](HW2)
        self.reduce_im = ctx.enqueue_create_buffer[DType.float32](HW2)
        self.denom = ctx.enqueue_create_buffer[DType.float32](HW)
        self.err_fft_re = ctx.enqueue_create_buffer[DType.float32](HW2)
        self.err_fft_im = ctx.enqueue_create_buffer[DType.float32](HW2)
        self.prod2_re = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.prod2_im = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.t_re_dhw2 = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.t_im_dhw2 = ctx.enqueue_create_buffer[DType.float32](DHW2)
        self.t_re_hw2 = ctx.enqueue_create_buffer[DType.float32](HW2)
        self.t_im_hw2 = ctx.enqueue_create_buffer[DType.float32](HW2)


def complex_mul_kernel[
    LT: TensorLayout
](
    a_re: TileTensor[DType.float32, LT, MutAnyOrigin],
    a_im: TileTensor[DType.float32, LT, MutAnyOrigin],
    b_re: TileTensor[DType.float32, LT, MutAnyOrigin],
    b_im: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_re: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_im: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int32,
):
    """Elementwise complex multiply over `n` flat elements; a and b have the
    same shape (no broadcasting)."""
    comptime assert a_re.flat_rank == 1, "expected flat tensor"
    var i = global_idx.x
    if i < Int(n):
        var ar = rebind[Scalar[DType.float32]](a_re[i])
        var ai = rebind[Scalar[DType.float32]](a_im[i])
        var br = rebind[Scalar[DType.float32]](b_re[i])
        var bi = rebind[Scalar[DType.float32]](b_im[i])
        out_re[i] = rebind[out_re.ElementType](ar * br - ai * bi)
        out_im[i] = rebind[out_im.ElementType](ar * bi + ai * br)


def sum_over_depth_kernel[
    XLT: TensorLayout, OLT: TensorLayout
](
    x_re: TileTensor[DType.float32, XLT, MutAnyOrigin],  # (d*hw,)
    x_im: TileTensor[DType.float32, XLT, MutAnyOrigin],
    out_re: TileTensor[DType.float32, OLT, MutAnyOrigin],  # (hw,)
    out_im: TileTensor[DType.float32, OLT, MutAnyOrigin],
    d: Int32,
    hw: Int32,
):
    comptime assert x_re.flat_rank == 1, "expected flat tensor"
    comptime assert out_re.flat_rank == 1, "expected flat tensor"
    var i = global_idx.x
    var hw_i = Int(hw)
    if i < hw_i:
        var acc_re: Scalar[DType.float32] = 0.0
        var acc_im: Scalar[DType.float32] = 0.0
        for b in range(Int(d)):
            acc_re += rebind[Scalar[DType.float32]](x_re[b * hw_i + i])
            acc_im += rebind[Scalar[DType.float32]](x_im[b * hw_i + i])
        out_re[i] = rebind[out_re.ElementType](acc_re)
        out_im[i] = rebind[out_im.ElementType](acc_im)


def sum_over_depth_real_kernel[
    XLT: TensorLayout, OLT: TensorLayout
](
    x: TileTensor[DType.float32, XLT, MutAnyOrigin],  # (d*hw,) real
    out_t: TileTensor[DType.float32, OLT, MutAnyOrigin],  # (hw,) real
    d: Int32,
    hw: Int32,
):
    comptime assert x.flat_rank == 1, "expected flat tensor"
    comptime assert out_t.flat_rank == 1, "expected flat tensor"
    var i = global_idx.x
    var hw_i = Int(hw)
    if i < hw_i:
        var acc: Scalar[DType.float32] = 0.0
        for b in range(Int(d)):
            acc += rebind[Scalar[DType.float32]](x[b * hw_i + i])
        out_t[i] = rebind[out_t.ElementType](acc)


def run_v1_step_gpu[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, TDIV: Int = 2
](
    ctx: DeviceContext,
    mut data_buf: DeviceBuffer[DType.float32],
    mut image_buf: DeviceBuffer[DType.float32],
    mut psf_fft_re: DeviceBuffer[DType.float32], mut psf_fft_im: DeviceBuffer[DType.float32],
    mut psft_fft_re: DeviceBuffer[DType.float32], mut psft_fft_im: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut scratch: OpenFlrScratch[D, H, W],
) raises:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v1 step.

    `psf_fft_*` and `psft_fft_*` are both `rfft2_batched_gpu_t` half spectra
    of the (real) PSF in the transposed canonical layout, shape
    (D, W/2+1, H) -- see optimizations.md sec. 3(a) for the forward chain and
    sec. 3(b) for the back-projection chain."""
    comptime W2 = W // 2 + 1
    comptime HW = H * W
    comptime HW2 = H * W2
    comptime DHW = D * HW
    comptime DHW2 = D * HW2
    comptime layout_dhw = row_major[DHW]()
    comptime layout_hw = row_major[HW]()
    comptime layout_hw2 = row_major[HW2]()
    comptime layout_dhw2 = row_major[DHW2]()

    # conv = irfft2(psf_fft * rfft2(data)), with the frequency-domain
    # buffers kept in the transposed (D, W/2+1, H) canonical layout. The old
    # `rfft2 -> irfft2_cmul` pair ran a transpose immediately followed by its
    # own inverse across that boundary (the elementwise multiply between them
    # commutes with the transpose, so both were redundant), and then ran the
    # forward and inverse height-axis FFTs as two separate kernels with a
    # full global round trip between them. Keeping the spectrum transposed
    # deletes both transposes; `fft_col_cmul_ifft_gpu` then does forward FFT,
    # multiply and inverse FFT in one shared-memory residency. Eight
    # dispatches become three, moving 2/3 of the bytes. `psf_fft_*` is
    # precomputed in the same transposed layout (free -- PSF prep is outside
    # the timed loop). See optimizations.md sec. 3(a).
    rfft_w_transposed_gpu[D, H, W, TILE, TW](
        ctx, data_buf, scratch.t_re_dhw2, scratch.t_im_dhw2,
        scratch.data_fft_re, scratch.data_fft_im,
    )
    fft_col_cmul_ifft_gpu[D, H, W, TW](
        ctx, scratch.t_re_dhw2, scratch.t_im_dhw2, psf_fft_re, psf_fft_im,
    )
    irfft_w_from_transposed_gpu[D, H, W, TILE, TW](
        ctx, scratch.t_re_dhw2, scratch.t_im_dhw2, scratch.conv,
        scratch.data_fft_re, scratch.data_fft_im,
    )

    comptime sumk = sum_over_depth_real_kernel[type_of(layout_dhw), type_of(layout_hw)]
    ctx.enqueue_function[sumk](
        TileTensor(scratch.conv, layout_dhw), TileTensor(scratch.denom, layout_hw),
        Int32(D), Int32(HW), grid_dim=ceildiv(HW, BLOCK_1D), block_dim=BLOCK_1D,
    )

    # Fused: rfft2(image_buf / denom) computed directly by
    # rfft2_batched_gpu_div_t, dividing at the row-FFT's load stage instead
    # of materializing image_buf / denom into a separate img_err buffer
    # first, and leaving the result in the transposed (1, W/2+1, H) layout
    # its consumer below wants.
    rfft2_batched_gpu_div_t[1, H, W, TILE, TW, TDIV](ctx, image_buf, scratch.denom, scratch.err_fft_re, scratch.err_fft_im, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: the whole tail of the update -- `data * fftshift(irfft2(err_fft
    # * psft_fft))` -- computed by irfft2_batched_gpu_cmul_broadcast_mul_t in
    # three kernels. The broadcast complex product is never materialized, and
    # with both operands in the transposed (D, W/2+1, H) layout the transpose
    # that used to carry that multiply is gone too; it now happens at the
    # inverse column FFT's load (optimizations.md sec. 3(b)). The closing
    # fftshift-and-multiply against `data_buf` is likewise folded into the
    # width inverse FFT's store, which already holds the finished row in
    # shared memory, so the (D, H, W) backprojection buffer is never written
    # or read back (sec. 3(c)) -- SHIFT=True rotates the destination index.
    # `scratch.prod2_re`/`scratch.prod2_im` are reused as the fused
    # function's internal scratch.
    irfft2_batched_gpu_cmul_broadcast_mul_t[D, H, W, TILE, True, TW, TDIV](
        ctx, scratch.err_fft_re, scratch.err_fft_im, psft_fft_re, psft_fft_im,
        data_buf, out_buf, scratch.prod2_re, scratch.prod2_im, scratch.t_re_dhw2, scratch.t_im_dhw2,
    )


def run_v2_step_gpu[
    D: Int, H: Int, W: Int, TILE: Int, TW: Bool = False, TDIV: Int = 2
](
    ctx: DeviceContext,
    mut data_buf: DeviceBuffer[DType.float32],
    mut image_buf: DeviceBuffer[DType.float32],
    mut psf_fft_re: DeviceBuffer[DType.float32], mut psf_fft_im: DeviceBuffer[DType.float32],
    mut psft_fft_re: DeviceBuffer[DType.float32], mut psft_fft_im: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut scratch: OpenFlrScratch[D, H, W],
) raises:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v2 step.

    `psf_fft_*` and `psft_fft_*` are both `rfft2_batched_gpu_t` half spectra
    of the (real) PSF in the transposed canonical layout, shape
    (D, W/2+1, H) -- see optimizations.md sec. 3(a) for the forward chain and
    sec. 3(b) for the back-projection chain."""
    comptime W2 = W // 2 + 1
    comptime HW = H * W
    comptime HW2 = H * W2
    comptime DHW2 = D * HW2
    comptime layout_hw2 = row_major[HW2]()
    comptime layout_dhw2 = row_major[DHW2]()

    # Transposed (D, W/2+1, H) canonical spectrum, as in v1: the final
    # transpose of `rfft2` and the leading transpose of the `irfft2` below
    # were exact inverses of each other, and everything in between
    # (`complex_mul_kernel`, `sum_over_depth_kernel`) is elementwise or a
    # depth reduction, both of which commute with a transpose of the last two
    # axes. `psf_fft_*` is precomputed transposed to match. See
    # optimizations.md sec. 3(a).
    rfft2_batched_gpu_t[D, H, W, TILE, TW, TDIV](ctx, data_buf, scratch.data_fft_re, scratch.data_fft_im, scratch.t_re_dhw2, scratch.t_im_dhw2)

    comptime cmul = complex_mul_kernel[type_of(layout_dhw2)]
    ctx.enqueue_function[cmul](
        TileTensor(psf_fft_re, layout_dhw2), TileTensor(psf_fft_im, layout_dhw2),
        TileTensor(scratch.data_fft_re, layout_dhw2), TileTensor(scratch.data_fft_im, layout_dhw2),
        TileTensor(scratch.prod_re, layout_dhw2), TileTensor(scratch.prod_im, layout_dhw2),
        Int32(DHW2), grid_dim=ceildiv(DHW2, BLOCK_1D), block_dim=BLOCK_1D,
    )

    comptime sumk = sum_over_depth_kernel[type_of(layout_dhw2), type_of(layout_hw2)]
    ctx.enqueue_function[sumk](
        TileTensor(scratch.prod_re, layout_dhw2), TileTensor(scratch.prod_im, layout_dhw2),
        TileTensor(scratch.reduce_re, layout_hw2), TileTensor(scratch.reduce_im, layout_hw2),
        Int32(D), Int32(HW2), grid_dim=ceildiv(HW2, BLOCK_1D), block_dim=BLOCK_1D,
    )

    irfft2_batched_gpu_t[1, H, W, TILE, TW, TDIV](ctx, scratch.reduce_re, scratch.reduce_im, scratch.denom, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: rfft2(image_buf / denom) computed directly by
    # rfft2_batched_gpu_div_t, dividing at the row-FFT's load stage instead
    # of materializing image_buf / denom into a separate img_err buffer
    # first, and leaving the result in the transposed (1, W/2+1, H) layout
    # its consumer below wants.
    rfft2_batched_gpu_div_t[1, H, W, TILE, TW, TDIV](ctx, image_buf, scratch.denom, scratch.err_fft_re, scratch.err_fft_im, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: the whole tail of the update -- `data * irfft2(err_fft *
    # psft_fft)` -- computed by irfft2_batched_gpu_cmul_broadcast_mul_t in
    # three kernels; see the same call in `run_v1_step_gpu` for what each
    # fusion removes. v2 applies no fftshift, hence SHIFT=False.
    # `scratch.prod2_re`/`scratch.prod2_im` are reused as the fused
    # function's internal scratch.
    irfft2_batched_gpu_cmul_broadcast_mul_t[D, H, W, TILE, False, TW, TDIV](
        ctx, scratch.err_fft_re, scratch.err_fft_im, psft_fft_re, psft_fft_im,
        data_buf, out_buf, scratch.prod2_re, scratch.prod2_im, scratch.t_re_dhw2, scratch.t_im_dhw2,
    )
