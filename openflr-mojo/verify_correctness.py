"""Cross-checks the Mojo GPU implementation against the numpy reference in
../main.py, at full scale (41, 2048, 2048), for every benchmark arm.

Two checks per version:
  * every arm agrees with numpy to better than 1e-3;
  * every *work-mapping* arm agrees with `base` **bit-exactly**. Those arms
    differ only in how work is mapped onto threads and blocks, never in the
    arithmetic or its order, so any difference at all is a bug.
    `t8w8c4Gtw` is the one exception and is held to the numpy tolerance
    only: it reads twiddles from a precomputed shared table instead of
    calling cos/sin per thread, which is a different rounding of the same
    quantity, not a different quantity.

Usage (from openflr-mojo/):
    pixi run verify
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


ARMS = ("base", "t4w8c4", "t8w8c4G", "t8w8c4Gi8", "t8w16c4G", "t8w8c4Gtw")
# Every arm but the twiddle-table one is a pure work-mapping change.
BIT_EXACT = tuple(a for a in ARMS if not a.endswith("tw"))


def mojo_result(version: str, arm: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bin", dir="/tmp") as f:
        subprocess.run(
            ["pixi", "run", "mojo", "run", "src/main.mojo", "--",
             version, "1", arm, "--dump", f.name],
            cwd=ROOT, check=True,
        )
        return np.fromfile(f.name, dtype=np.float32)


def main() -> None:
    for version in ("v1", "v2"):
        expected = numpy_reference(version).ravel()
        results = {}
        for arm in ARMS:
            actual = mojo_result(version, arm)
            results[arm] = actual
            diff = np.abs(expected - actual)
            rel = diff / (np.abs(expected) + 1e-8)
            print(
                f"{version} {arm:<8}: max abs diff = {diff.max():.3e}, "
                f"mean abs diff = {diff.mean():.3e}, max rel diff = {rel.max():.3e}"
            )
            assert diff.max() < 1e-3, f"{version} {arm} mismatch too large"

        ref = results[BIT_EXACT[0]]
        for arm in BIT_EXACT[1:]:
            assert np.array_equal(ref, results[arm]), (
                f"{version}: {arm} is not bit-identical to {BIT_EXACT[0]} "
                f"(max diff {np.abs(ref - results[arm]).max():.3e}). These arms "
                f"remap work across threads and must not change the arithmetic."
            )
        print(f"{version}: {' == '.join(BIT_EXACT)} bit-identical")
    print("OK: every arm matches the numpy reference; work-mapping arms are bit-exact.")


if __name__ == "__main__":
    main()
