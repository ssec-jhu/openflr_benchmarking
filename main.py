from pathlib import Path
import os
import sys
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import fire
import numpy as np
import torch
from numpy.typing import ArrayLike
from PIL import Image

IMAGE = np.ndarray
PSF = np.ndarray

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# jax.config.update("jax_log_compiles", True)

def open_image(path:str|Path)-> np.ndarray:
    with Image.open(path) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))

    return np.stack(frames, axis=0) # [n_frames, h, w]


def get_data(data_path:str|Path)-> tuple[IMAGE, PSF]:
    data_path = Path(data_path)
    image = open_image(data_path / "light_field_image.tif")
    psf = open_image(data_path / "measured_psf.tif")
    return image, psf

# v1 ===========================================================================

def run_v1_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
    irfft2_fn:Callable,
    rfft2_fn:Callable,
    fftshift_fn:Callable,
    sum_fn:Callable,
) -> ArrayLike:
    denominator = sum_fn(irfft2_fn(PSF_fft * rfft2_fn(data)))
    img_err = image / denominator
    return data * fftshift_fn(irfft2_fn(rfft2_fn(img_err) * PSFt_fft))

def run_numpy_v1_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:
    return run_v1_step(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=np.fft.irfft2,
        rfft2_fn=np.fft.rfft2,
        fftshift_fn=partial(np.fft.fftshift, axes=(-2, -1)),
        sum_fn=partial(np.sum, axis=0, keepdims=True)
    )

v1_jitted = jax.jit(run_v1_step, static_argnames=["irfft2_fn", "rfft2_fn", "fftshift_fn", "sum_fn"], donate_argnames=("data",))
jax_fftshift = jax.jit(partial(jnp.fft.fftshift, axes=(-2, -1)))
jax_sum = jax.jit(partial(jnp.sum, axis=0, keepdims=True))
def run_jax_v1_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:

    return v1_jitted(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=jnp.fft.irfft2,
        rfft2_fn=jnp.fft.rfft2,
        fftshift_fn=jax_fftshift,
        sum_fn=jax_sum
    )


def run_torch_v1_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:

    return run_v1_step(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=partial(torch.fft.irfft2, dim=(-2, -1)),
        rfft2_fn=torch.fft.rfft2,
        fftshift_fn=partial(torch.fft.fftshift, dim=(-2, -1)),
        sum_fn=partial(torch.sum, dim=0, keepdim=True)
    )
# ==============================================================================

# v2 ===========================================================================

def run_v2_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
    irfft2_fn:Callable,
    rfft2_fn:Callable,
    sum_fn:Callable,
) -> ArrayLike:
    freq_sum = sum_fn(PSF_fft * rfft2_fn(data))
    img_err = image / irfft2_fn(freq_sum)
    return data * irfft2_fn(rfft2_fn(img_err) * PSFt_fft)

def run_numpy_v2_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:
    return run_v2_step(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=np.fft.irfft2,
        rfft2_fn=np.fft.rfft2,
        sum_fn=partial(np.sum, axis=0, keepdims=True)
    )


v2_jitted = jax.jit(run_v2_step, static_argnames=["irfft2_fn", "rfft2_fn", "sum_fn"], donate_argnames=("data",))
def run_jax_v2_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:
    return v2_jitted(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=jnp.fft.irfft2,
        rfft2_fn=jnp.fft.rfft2,
        sum_fn=jax_sum
    )


def run_torch_v2_step(
    data:ArrayLike,
    image:ArrayLike,
    PSF_fft:ArrayLike,
    PSFt_fft:ArrayLike,
) -> ArrayLike:
     return run_v2_step(
        data=data,
        image=image,
        PSF_fft=PSF_fft,
        PSFt_fft=PSFt_fft,
        irfft2_fn=partial(torch.fft.irfft2, dim=(-2, -1)),
        rfft2_fn=torch.fft.rfft2,
        sum_fn=partial(torch.sum, dim=0, keepdim=True)
    )



# ==============================================================================
FILE_MSG = "{mean} \\pm {std}"
PRINT_MSG = "backend: {backend}, version: {version}, mean time: {mean}s, std time: {std}s"
def main(
    backend:str,
    use_openflr:bool = False,
    use_openflr_v2:bool = False,
    n_iters:int = 20,
):
    if use_openflr == use_openflr_v2:
        raise ValueError("Exactly one of use_openflr and use_openflr_v2 must be True.")

    img, psf = get_data("data/openflr")

    if backend == "jax":
        img = jnp.asarray(img).astype(jnp.float32)
        psf = jnp.asarray(psf).astype(jnp.float32)
        psf = psf / jnp.sum(psf)
        psf_fft = jnp.fft.rfft2(psf)
        psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))
        guess = jnp.ones_like(psf) * 0.5

        if use_openflr:
            guess = run_jax_v1_step(
                data=guess,
                image=img,
                PSF_fft=psf_fft,
                PSFt_fft=psft_fft,
            ).block_until_ready()

            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_jax_v1_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                ).block_until_ready()
                times.append(time.perf_counter() - start)
        else:
            psft_fft = jnp.fft.rfft2(jnp.fft.ifftshift(jnp.flip(psf, axis=(-2, -1)), axes=(-2, -1)))

            guess = run_jax_v2_step(
                data=guess,
                image=img,
                PSF_fft=psf_fft,
                PSFt_fft=psft_fft,
            ).block_until_ready()

            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_jax_v2_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                ).block_until_ready()
                times.append(time.perf_counter() - start)

        print(PRINT_MSG.format(backend=backend, version=1 if use_openflr else 2, mean=np.mean(times), std=np.std(times)), file=sys.stderr)
        print(FILE_MSG.format(mean=np.mean(times), std=np.std(times)))

    elif backend == "torch":
        img = torch.as_tensor(img.astype(np.float32)).to("cuda")
        psf = torch.as_tensor(psf.astype(np.float32))
        psf = psf / torch.sum(psf)
        psf_fft = torch.fft.rfft2(psf).to("cuda")
        psft_fft = torch.fft.rfft2(torch.flip(psf, dims=(-2, -1))).to("cuda")
        guess = torch.ones_like(psf).to("cuda") * 0.5

        if use_openflr:
            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_torch_v1_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                )
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        else:
            psft_fft = torch.fft.rfft2(torch.fft.ifftshift(torch.flip(psf, dims=(-2, -1)), dim=(-2, -1))).to("cuda")
            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_torch_v2_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                )
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        print(PRINT_MSG.format(backend=backend, version=1 if use_openflr else 2, mean=np.mean(times), std=np.std(times)), file=sys.stderr)
        print(FILE_MSG.format(mean=np.mean(times), std=np.std(times)))

    elif backend == "numpy":
        img = np.asarray(img).astype(np.float32)
        psf = np.asarray(psf).astype(np.float32)
        psf = psf / np.sum(psf)
        psf_fft = np.fft.rfft2(psf)
        psft_fft = np.fft.rfft2(np.flip(psf, axis=(-2, -1)))
        guess = np.ones_like(psf) * 0.5

        if use_openflr:
            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_numpy_v1_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                )
                times.append(time.perf_counter() - start)
        else:
            psft_fft = np.fft.rfft2(np.fft.ifftshift(np.flip(psf, axis=(-2, -1)), axes=(-2, -1)))

            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                guess = run_numpy_v2_step(
                    data=guess,
                    image=img,
                    PSF_fft=psf_fft,
                    PSFt_fft=psft_fft,
                )
                times.append(time.perf_counter() - start)

        print(PRINT_MSG.format(backend=backend, version=1 if use_openflr else 2, mean=np.mean(times), std=np.std(times)), file=sys.stderr)
        print(FILE_MSG.format(mean=np.mean(times), std=np.std(times)))
    else:
        raise ValueError("Invalid backend specified.")


if __name__ == "__main__":
    fire.Fire(main)
