"""One-time preprocessing: convert the OpenFLR TIFF stacks into raw float32
binary files that the Mojo benchmark can read directly (no TIFF decoder in
Mojo). Mirrors `open_image`/`get_data` in ../main.py.

Run with the sibling uv-managed venv, which already has Pillow/numpy:
    ../.venv/bin/python3 prepare_data.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).parent.parent / "data" / "openflr"
DST = Path(__file__).parent / "data"


def open_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))
    return np.stack(frames, axis=0)  # [n_frames, h, w]


def main() -> None:
    DST.mkdir(exist_ok=True)

    image = open_image(SRC / "light_field_image.tif").astype(np.float32)
    psf = open_image(SRC / "measured_psf.tif").astype(np.float32)

    print("image shape:", image.shape, "psf shape:", psf.shape)
    assert image.shape[0] == 1

    image[0].tofile(DST / "image.bin")
    psf.tofile(DST / "psf.bin")

    (DST / "shape.txt").write_text(
        f"{psf.shape[0]} {psf.shape[1]} {psf.shape[2]}\n"
    )
    print("wrote", DST / "image.bin", DST / "psf.bin", DST / "shape.txt")


if __name__ == "__main__":
    main()
