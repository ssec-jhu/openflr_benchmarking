# openflr-mojo

A Mojo/MAX implementation of one OpenFLR Richardson-Lucy deconvolution
iteration (both the `v1` and `v2` formulations from `../main.py`), targeting
GPU (tested on AMD via ROCm; portable to NVIDIA since it's pure Mojo, no
vendor FFT library) with a CPU fallback for correctness checking.

## Why not just call MAX's `irfft` op?

MAX's `max.graph.ops.irfft` is GPU-only and is a thin wrapper around
**cuFFT** (see `max/kernels/src/_cufft/`), so it only runs on NVIDIA. There's
also no forward `rfft`/`fft2` op at all. Since this repo's benchmark target
is a real-input 2D FFT convolution and needs to run on AMD too, this project
implements its own radix-2 Cooley-Tukey FFT directly in Mojo, GPU and CPU.

`H = W = 2048` in the real dataset is a power of two (`2^11`), which radix-2
FFT handles natively.

**Real-input optimization**: `run_v1_step`/`run_v2_step` use a real-input
`rfft2`/`irfft2` (in `fft_cpu.mojo`/`fft_gpu.mojo`), matching
`numpy.fft.rfft2`/`irfft2` exactly rather than a full complex FFT with a
zeroed imaginary part. The forward/inverse real transform along the width
axis is computed via the standard "pack a length-`N` real signal into one
length-`N/2` complex FFT" trick (`rfft_1d`/`irfft_1d` on CPU,
`rfft_row_kernel`/`irfft_row_kernel` on GPU): even/odd samples are packed
into one complex signal, transformed with the existing radix-2 complex FFT,
and unpacked into (or repacked from) the non-redundant half spectrum
(`W/2+1` complex bins) via the even/odd DFT symmetry. The height axis is
still a full complex FFT (numpy's `rfft2` does the same: real FFT along the
last axis, complex FFT along the rest). This roughly halves both the
frequency-domain memory footprint and the FFT arithmetic relative to the
zero-padded full-complex approach, and measured close to 2x faster
end-to-end on the `(41, 2048, 2048)` dataset (v2: ~0.36-0.39s/iteration →
~0.14s/iteration).

## Layout

- `src/fft_cpu.mojo` / `src/fft_gpu.mojo` — radix-2 Cooley-Tukey 1D FFT,
  composed into a batched 2D FFT via row-transform + transpose + row-transform.
  The GPU row kernel keeps one whole row in shared memory per block (up to
  N=2048, i.e. 16KB/buffer, well under typical 48-164KB/block budgets) and
  does bit-reversal + all `log2(N)` butterfly stages without leaving shared
  memory. The transpose is a classic tiled shared-memory transpose
  (`TileTensor` + `stack_allocation`), batched over the depth axis via a 3D
  grid (`block_idx.z`). `rfft_1d`/`irfft_1d` (CPU) and `rfft_row_kernel`/
  `irfft_row_kernel` (GPU) build a real-input-optimized FFT out of the same
  complex butterfly stages; `rfft2_batched`/`irfft2_batched` compose that
  with a full complex row-transform over the other axis (row-transform +
  transpose + row-transform + transpose back) to match `numpy.fft.rfft2`/
  `irfft2`.
- `src/openflr_cpu.mojo` / `src/openflr_gpu.mojo` — the elementwise/reduction
  glue (complex multiply, depth-sum, divide, multiply, PSF flip/shift) that
  turns the FFT primitives into `run_v1_step`/`run_v2_step`.
- `src/main.mojo` — CLI entrypoint / benchmark harness, mirroring
  `../main.py`'s timing loop and stdout/stderr output format.
- `src/test_*.mojo` — correctness tests, each validated against a numpy
  reference embedded as literal data (small sizes only — embedding large
  literal arrays in Mojo source compiles very slowly, so bigger checks use
  `verify_correctness.py` instead, which compares against real data via
  files).

## Setup

```bash
pixi install
pixi run prepare-data   # converts ../data/openflr/*.tif into data/*.bin (once)
```

## Testing

```bash
pixi run test           # CPU + GPU correctness tests, small synthetic sizes
../.venv/bin/python3 verify_correctness.py   # full-scale (41, 2048, 2048) check against numpy
```

## Running / benchmarking

```bash
pixi run run-v1                       # v1, 20 iterations
pixi run run-v2                       # v2, 20 iterations
mojo run src/main.mojo -- v2 20       # equivalent, explicit
mojo run src/main.mojo -- v2 20 t4    # one arm only (what a profiler wants)
mojo run src/main.mojo -- v1 1 --dump out.bin   # single step, dump raw float32 output
pixi run verify                       # every arm vs the numpy reference, full scale
```

Each run benchmarks five **arms** — compile-time configurations threaded
from `main.mojo` down to the kernels, each giving some group of kernels a
different block size — and prints one line each: `base` (the reference
implementation), `t4w8`, `t4w8c4`, `t8w8c4` and `t4w8c4g`. `t4w8c4` is the
fastest measured on the A100; the last two are written and verified but not
yet measured there. All five arms are bit-identical to each other by
construction — they only remap work across threads — and `pixi run verify`
asserts it. See `optimizations.md` for what each knob sizes and which two
arms are there as measurement controls rather than as candidates.

Output mirrors `../main.py`: a human-readable line on stderr, and an
`arm=<name> mean \pm std` (seconds) line on stdout per arm — so
`make time-mojo-v1` / `make time-mojo-v2` from the repo root behave like the
other backends' targets, with one line per arm instead of one.

**Before optimising anything, read the `START HERE` section at the top of
[`optimizations.md`](optimizations.md).** It carries the current numbers, the
ordered list of what to do next, and — most importantly — why the A100 and
the AMD iGPU this was originally tuned on are bottlenecked on different
resources, which invalidates a lot of otherwise-reasonable intuition.

## Known limitations / future work

- ~~**Full complex FFT, not real FFT.**~~ Fixed: `run_v1_step`/`run_v2_step`
  (CPU and GPU) now use a real-input-optimized `rfft2`/`irfft2` (see "Real-input
  optimization" above) instead of a full complex FFT with a zeroed imaginary
  part. Measured ~2x faster end-to-end on the `(41, 2048, 2048)` dataset
  (v2: ~0.36-0.39s/iteration → ~0.14s/iteration) — the speedup this README
  had anticipated from this change.
- ~~**No scratch-buffer reuse across iterations.**~~ Fixed: `run_v1_step_gpu`/
  `run_v2_step_gpu` now take a caller-owned `OpenFlrScratch[D, H, W]`
  (`openflr_gpu.mojo`), allocated once outside the benchmark loop and reused
  every call; `fft2_batched_gpu`/`rfft2_batched_gpu`/`irfft2_batched_gpu`
  likewise take their transpose scratch as parameters instead of allocating
  internally. This did **not** produce a measurable speedup in practice on
  its own (benchmarked at parity with the allocate-every-call version) —
  MAX's `DeviceContext.enqueue_create_buffer` appears to already be backed by
  a caching allocator, so the win this README anticipated wasn't actually
  available at the allocator layer. The refactor is kept anyway since
  explicit scratch ownership is still better practice than repeated ad hoc
  allocation.
- **No warp-level primitives in the FFT butterfly stages** — the shared
  memory row-FFT kernel is a straightforward textbook implementation, not
  tuned with warp shuffles, bank-conflict avoidance, or mixed radix (4/8)
  stages the way production FFT libraries (cuFFT/rocFFT) are.
- Sizes are compile-time constants (`D=41, H=2048, W=2048`, hardcoded in
  `main.mojo`) rather than dynamic — matches this benchmark's fixed dataset,
  but means a different dataset shape needs a rebuild, not just new input
  files.
