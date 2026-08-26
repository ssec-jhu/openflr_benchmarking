"""Cross-checks the Mojo GPU implementation against the numpy reference in
../main.py, at full scale (41, 2048, 2048).

Usage (from openflr-mojo/):
    ../.venv/bin/python3 verify_correctness.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import get_data, run_numpy_v1_step, run_numpy_v2_step  # noqa: E402

ROOT = Path(__file__).parent


def numpy_reference(version: str) -> np.ndarray:
    img, psf = get_data(ROOT.parent / "data" / "openflr")
    img = img.astype(np.float32)
    psf = psf.astype(np.float32)
    psf = psf / psf.sum()
    data = np.full(psf.shape, 0.5, dtype=np.float32)

    psf_fft = np.fft.rfft2(psf)
    if version == "v1":
        psft_fft = np.fft.rfft2(np.flip(psf, axis=(-2, -1)))
        return run_numpy_v1_step(data, img, psf_fft, psft_fft)
    else:
        psft_fft = np.fft.rfft2(
            np.fft.ifftshift(np.flip(psf, axis=(-2, -1)), axes=(-2, -1))
        )
        return run_numpy_v2_step(data, img, psf_fft, psft_fft)


def mojo_result(version: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bin", dir="/tmp") as f:
        subprocess.run(
            ["pixi", "run", "mojo", "run", "src/main.mojo", "--", version, "1", "--dump", f.name],
            cwd=ROOT, check=True,
        )
        return np.fromfile(f.name, dtype=np.float32)


def main() -> None:
    for version in ("v1", "v2"):
        expected = numpy_reference(version).ravel()
        actual = mojo_result(version)
        diff = np.abs(expected - actual)
        rel = diff / (np.abs(expected) + 1e-8)
        print(
            f"{version}: max abs diff = {diff.max():.3e}, "
            f"mean abs diff = {diff.mean():.3e}, max rel diff = {rel.max():.3e}"
        )
        assert diff.max() < 1e-3, f"{version} mismatch too large"
    print("OK: Mojo GPU implementation matches the numpy reference.")


if __name__ == "__main__":
    main()
