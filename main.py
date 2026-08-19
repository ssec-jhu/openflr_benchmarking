
from pathlib import Path
import time

import numpy as np
from pyolaf.aliasing import lanczosfft, LFM_computeDepthAdaptiveWidth
from pyolaf.geometry import LFM_computeGeometryParameters, LFM_setCameraParams
from pyolaf.lf import LFM_computeLFMatrixOperators
from pyolaf.project import LFM_forwardProject, LFM_backwardProject
from pyolaf.transform import LFM_retrieveTransformation, format_transform, get_transformed_shape, transform_img


try:
    import cupy
    from cupy.fft import fftshift, ifft2, fft2
    has_cupy = True
except ImportError:
    import numpy as cupy
    from numpy.fft import fftshift, ifft2, fft2
    has_cupy = False

if has_cupy:
    mempool = cupy.get_default_memory_pool()
    mempool.set_limit(7.5 * 2**30)



def main():
    data_location = Path(__file__).parent / "data" / "pyolaf"
    run_olaf(data_location)



# olaf code adapted from here:
# https://github.com/lambdaloop/pyolaf/blob/main/examples/deconvolve_image.py
def run_olaf(
    data_location:str|Path,
    *,
    n_iters:int = 10,
    new_spacing_px:int = 15,
    depth_range:tuple[int,int] = (-300, 300),
    depth_step:int = 150,
    super_resolution_factor:int = 5,
    lanczos_window_size:int = 4,
    filter_flag:bool = True,
    use_gpu:bool=False,
) -> None:
    import tifffile

    calibration = Path(data_location) / "calib.tif"
    config = Path(data_location) / "config.yaml"
    image = Path(data_location) / "example_fly.tif"

    white_image = tifffile.imread(calibration)
    lenslet_image = tifffile.imread(image)

    camera = LFM_setCameraParams(config, new_spacing_px)

    (
        lenslet_centers,
        resolution,
        lenslet_grid_model,
        new_lenslet_grid_model,
     ) = LFM_computeGeometryParameters(
        camera,
        white_image,
        depth_range,
        depth_step,
        super_resolution_factor,
    )

    h, ht = LFM_computeLFMatrixOperators(camera, resolution, lenslet_centers)

    fix_all = LFM_retrieveTransformation(lenslet_grid_model, new_lenslet_grid_model)

    transform = format_transform(fix_all)
    img_size = get_transformed_shape(white_image.shape, transform)
    img_size = img_size + (1 - np.remainder(img_size, 2))

    tex_size = np.ceil(np.multiply(img_size, resolution["texScaleFactor"])).astype(np.int32)
    tex_size = tex_size + (1 - np.remainder(tex_size, 2))

    n_depths = len(resolution["depths"])
    volume_size = np.append(tex_size, n_depths).astype(np.int32)

    widths = LFM_computeDepthAdaptiveWidth(camera, resolution)
    kernel_fft = lanczosfft(volume_size, widths, lanczos_window_size)

    img = cupy.array(lenslet_image, dtype="float32")
    new = transform_img(img, transform, lenslet_centers["offset"])
    new_norm = (new - np.min(new)) / (np.max(new) - np.min(new))
    lf_image = new_norm

    camera_range = camera["range"]
    init_volume = np.ones(volume_size, dtype="float32")

    ones_forward = LFM_forwardProject(h, init_volume, lenslet_centers, resolution, img_size, camera_range, step=8)
    ones_back = LFM_backwardProject(ht, ones_forward, lenslet_centers, resolution, tex_size, camera_range, step=8)

    lf_image = cupy.asarray(lf_image)
    recon_volume = cupy.asarray(np.copy(init_volume))

    start = time.time()

    for i in range(n_iters):
        if i == 0:
            lf_image_guess = ones_forward
        else:
            lf_image_guess = LFM_forwardProject(h, recon_volume, lenslet_centers, resolution, img_size, camera_range, step=10)
        if has_cupy:
            mempool.free_all_blocks()

        error_lf_image = lf_image / lf_image_guess * ones_forward
        error_lf_image[~cupy.isfinite(error_lf_image)] = 0

        error_back = LFM_backwardProject(ht, error_lf_image, lenslet_centers, resolution, tex_size, camera_range, step=10)

        recon_volume *= error_back

        if filter_flag:
            for j in  range(error_back.shape[2]):
                recon_volume[:,:,j] = cupy.abs(fftshift(ifft2(kernel_fft[:,:,j] * fft2(recon_volume[:,:,j]))))

        recon_volume[~cupy.isfinite(recon_volume)] = 0
        if has_cupy:
            mempool.free_all_blocks()


    if has_cupy:
        recon_volume_np = cupy.asnumpy(recon_volume)
    else:
        recon_volume_np = recon_volume

    print(f"Finished {n_iters} iterations in {time.time() - start:.2f} seconds.")

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    plt.imshow(recon_volume_np[:, :, 0])
    plt.draw()
    plt.show()

if __name__ == "__main__":
    main()
