"""GPU implementation of one OpenFLR Richardson-Lucy iteration (v1 and v2),
built on top of the radix-2 FFT kernels in `fft_gpu`.
"""

from std.math import ceildiv
from std.gpu import global_idx, thread_idx, block_idx
from layout import TileTensor, TensorLayout, row_major
from max.gpu.host import DeviceContext, DeviceBuffer

from fft_gpu import (
    rfft2_batched_gpu,
    rfft2_batched_gpu_div,
    irfft2_batched_gpu,
    irfft2_batched_gpu_cmul,
    irfft2_batched_gpu_cmul_broadcast,
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
    var back: DeviceBuffer[DType.float32]
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
        self.back = ctx.enqueue_create_buffer[DType.float32](DHW)
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


def elementwise_mul_kernel[
    LT: TensorLayout
](
    a: TileTensor[DType.float32, LT, MutAnyOrigin],
    b: TileTensor[DType.float32, LT, MutAnyOrigin],
    out_t: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int32,
):
    comptime assert a.flat_rank == 1, "expected flat tensor"
    var i = global_idx.x
    if i < Int(n):
        out_t[i] = rebind[out_t.ElementType](
            rebind[Scalar[DType.float32]](a[i]) * rebind[Scalar[DType.float32]](b[i])
        )


def run_v1_step_gpu[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut data_buf: DeviceBuffer[DType.float32],
    mut image_buf: DeviceBuffer[DType.float32],
    mut psf_fft_re: DeviceBuffer[DType.float32], mut psf_fft_im: DeviceBuffer[DType.float32],
    mut psft_fft_re: DeviceBuffer[DType.float32], mut psft_fft_im: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut scratch: OpenFlrScratch[D, H, W],
) raises:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v1 step -- `psf_fft_*`/
    `psft_fft_*` are the `rfft2_batched_gpu` half spectrum of the (real)
    PSF, shape (D, H, W/2+1)."""
    comptime W2 = W // 2 + 1
    comptime HW = H * W
    comptime HW2 = H * W2
    comptime DHW = D * HW
    comptime DHW2 = D * HW2
    comptime layout_dhw = row_major[DHW]()
    comptime layout_hw = row_major[HW]()
    comptime layout_hw2 = row_major[HW2]()
    comptime layout_dhw2 = row_major[DHW2]()

    rfft2_batched_gpu[D, H, W, TILE](ctx, data_buf, scratch.data_fft_re, scratch.data_fft_im, scratch.t_re_dhw2, scratch.t_im_dhw2)

    # Fused: irfft2(psf_fft * data_fft) computed directly by
    # irfft2_batched_gpu_cmul, never materializing the (D, H, W/2+1) complex
    # product -- the single biggest buffer in the pipeline. `scratch.prod_re`/
    # `scratch.prod_im` are reused as the fused function's internal scratch
    # (they no longer hold the product itself).
    irfft2_batched_gpu_cmul[D, H, W, TILE](
        ctx, psf_fft_re, psf_fft_im, scratch.data_fft_re, scratch.data_fft_im,
        scratch.conv, scratch.prod_re, scratch.prod_im, scratch.t_re_dhw2, scratch.t_im_dhw2,
    )

    comptime sumk = sum_over_depth_real_kernel[type_of(layout_dhw), type_of(layout_hw)]
    ctx.enqueue_function[sumk](
        TileTensor(scratch.conv, layout_dhw), TileTensor(scratch.denom, layout_hw),
        Int32(D), Int32(HW), grid_dim=ceildiv(HW, BLOCK_1D), block_dim=BLOCK_1D,
    )

    # Fused: rfft2(image_buf / denom) computed directly by
    # rfft2_batched_gpu_div, dividing at the row-FFT's load stage instead of
    # materializing image_buf / denom into a separate img_err buffer first.
    rfft2_batched_gpu_div[1, H, W, TILE](ctx, image_buf, scratch.denom, scratch.err_fft_re, scratch.err_fft_im, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: irfft2(err_fft * psft_fft) computed directly by
    # irfft2_batched_gpu_cmul_broadcast, never materializing the broadcast
    # complex product. `scratch.prod2_re`/`scratch.prod2_im` are reused as
    # the fused function's internal scratch.
    irfft2_batched_gpu_cmul_broadcast[D, H, W, TILE](
        ctx, scratch.err_fft_re, scratch.err_fft_im, psft_fft_re, psft_fft_im,
        scratch.back, scratch.prod2_re, scratch.prod2_im, scratch.t_re_dhw2, scratch.t_im_dhw2,
    )

    # v1 applies an fftshift over (-2, -1) to the backprojection before the
    # final multiply; fused into a single kernel that reads `back` at the
    # shift-computed index and multiplies against `data_buf` directly,
    # eliminating the separate `shifted` scratch buffer and kernel launch.
    comptime layout_dhw3 = row_major[D, H, W]()
    comptime smk = shift_mul_kernel[H, W, type_of(layout_dhw3)]
    ctx.enqueue_function[smk](
        TileTensor(data_buf, layout_dhw3), TileTensor(scratch.back, layout_dhw3),
        TileTensor(out_buf, layout_dhw3),
        grid_dim=(ceildiv(W, 16), ceildiv(H, 16), D), block_dim=(16, 16),
    )


def run_v2_step_gpu[
    D: Int, H: Int, W: Int, TILE: Int
](
    ctx: DeviceContext,
    mut data_buf: DeviceBuffer[DType.float32],
    mut image_buf: DeviceBuffer[DType.float32],
    mut psf_fft_re: DeviceBuffer[DType.float32], mut psf_fft_im: DeviceBuffer[DType.float32],
    mut psft_fft_re: DeviceBuffer[DType.float32], mut psft_fft_im: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut scratch: OpenFlrScratch[D, H, W],
) raises:
    """Real-input-optimized (rfft2/irfft2) OpenFLR v2 step -- `psf_fft_*`/
    `psft_fft_*` are the `rfft2_batched_gpu` half spectrum of the (real)
    PSF, shape (D, H, W/2+1)."""
    comptime W2 = W // 2 + 1
    comptime HW = H * W
    comptime HW2 = H * W2
    comptime DHW = D * HW
    comptime DHW2 = D * HW2
    comptime layout_dhw = row_major[DHW]()
    comptime layout_hw = row_major[HW]()
    comptime layout_hw2 = row_major[HW2]()
    comptime layout_dhw2 = row_major[DHW2]()

    rfft2_batched_gpu[D, H, W, TILE](ctx, data_buf, scratch.data_fft_re, scratch.data_fft_im, scratch.t_re_dhw2, scratch.t_im_dhw2)

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

    irfft2_batched_gpu[1, H, W, TILE](ctx, scratch.reduce_re, scratch.reduce_im, scratch.denom, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: rfft2(image_buf / denom) computed directly by
    # rfft2_batched_gpu_div, dividing at the row-FFT's load stage instead of
    # materializing image_buf / denom into a separate img_err buffer first.
    rfft2_batched_gpu_div[1, H, W, TILE](ctx, image_buf, scratch.denom, scratch.err_fft_re, scratch.err_fft_im, scratch.t_re_hw2, scratch.t_im_hw2)

    # Fused: irfft2(err_fft * psft_fft) computed directly by
    # irfft2_batched_gpu_cmul_broadcast, never materializing the broadcast
    # complex product. `scratch.prod2_re`/`scratch.prod2_im` are reused as
    # the fused function's internal scratch.
    irfft2_batched_gpu_cmul_broadcast[D, H, W, TILE](
        ctx, scratch.err_fft_re, scratch.err_fft_im, psft_fft_re, psft_fft_im,
        scratch.back, scratch.prod2_re, scratch.prod2_im, scratch.t_re_dhw2, scratch.t_im_dhw2,
    )

    comptime mulk = elementwise_mul_kernel[type_of(layout_dhw)]
    ctx.enqueue_function[mulk](
        TileTensor(data_buf, layout_dhw), TileTensor(scratch.back, layout_dhw),
        TileTensor(out_buf, layout_dhw),
        Int32(DHW), grid_dim=ceildiv(DHW, BLOCK_1D), block_dim=BLOCK_1D,
    )


def shift_mul_kernel[
    H: Int, W: Int, LT: TensorLayout
](
    data: TileTensor[DType.float32, LT, MutAnyOrigin],  # (D, H, W)
    back: TileTensor[DType.float32, LT, MutAnyOrigin],  # (D, H, W), unshifted
    out_t: TileTensor[DType.float32, LT, MutAnyOrigin],  # (D, H, W)
):
    """Fusion of an fftshift over the last two axes with the following
    elementwise multiply: computes `out = data * fftshift(back)` directly,
    reading `back` at the shift-computed index and multiplying against
    `data` in the same kernel, instead of materializing the shifted buffer
    through a separate kernel launch and a full round trip through global
    memory. Valid because fftshift is its own inverse for even H, W:
    fftshift(back)[r, c] = back[(r+H/2)%H, (c+W/2)%W]."""
    comptime assert data.flat_rank == 3, "expected (D, H, W) tensor"
    comptime hh = H // 2
    comptime hw2 = W // 2
    var b = block_idx.z
    var row = block_idx.y * 16 + thread_idx.y
    var col = block_idx.x * 16 + thread_idx.x
    if row < H and col < W:
        var src_row = (row + hh) % H
        var src_col = (col + hw2) % W
        var shifted_val = rebind[Scalar[DType.float32]](back[b, src_row, src_col])
        var data_val = rebind[Scalar[DType.float32]](data[b, row, col])
        out_t[b, row, col] = rebind[out_t.ElementType](data_val * shifted_val)


