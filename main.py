
import time
from pathlib import Path

import fire

def main(
    use_openflr:bool = False,
    use_openflr_v2:bool = False,
):
    if not use_openflr ^ use_openflr_v2:
        raise ValueError("At least one of use_olaf, use_openflr, or use_openflr_v2 must be True.")

    if use_openflr:
        run_openflr(Path(__file__).parent / "data" / "openflr")

    if use_openflr_v2:
        run_openflr_v2(Path(__file__).parent / "data" / "openflr")

def run_openflr(data_path:str|Path, backend:str)-> None:
    """Runs openflr reconstruction on the given data path using the specified backend.

    Set the timing variable to a truthy variable and set the backend to either "torch" or "jax"
    FLFM_TIME_RECONSTRUCTION=1 uv run --group openflr main.py --use_openflr
    """
    import flfm.io
    import flfm.restoration

    data_path = Path(data_path)
    image = flfm.io.open(data_path / "light_field_image.tif")
    psf = flfm.io.open(data_path / "measured_psf.tif")
    psf_norm = psf / flfm.restoration.sum(psf)
    reconstruction = flfm.restoration.reconstruct(image, psf_norm, recon_kwargs=dict())



if __name__ == "__main__":
    fire.Fire(main)
