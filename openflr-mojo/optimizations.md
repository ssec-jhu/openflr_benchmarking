# Optimization findings

Investigated why the Mojo GPU implementation is slower than JAX/PyTorch.
Tested on an AMD integrated GPU (no H100 available); absolute numbers won't
transfer, but the methodology and the unexplained bottleneck should.

## Kept

- **Transpose shared-memory padding** (`fft_gpu.mojo`, `transpose_kernel`):
  padded the tile allocation from `[TILE, TILE]` to `[TILE, TILE+1]`. The
  unpadded version has every thread in a warp land in the same shared-memory
  bank on the transposed read (stride == TILE == bank count), a 32-way bank
  conflict on every access. Verified correct, benchmarks at parity or
  slightly better. Only change left in the tree.

## Tried and reverted (no measured benefit)

- **Twiddle-factor table**: replaced per-thread `cos`/`sin` in the FFT
  butterfly stages with a precomputed table (first global memory, then
  staged through shared memory). Textbook-correct, verified bit-exact
  against numpy at full scale both times — but measured neutral-to-slightly
  *slower* on the test GPU. A kernel-by-kernel timing breakdown showed why:
  the FFT row kernels are a small slice of total time to begin with, so
  trimming their transcendental math couldn't move the needle much, and the
  extra shared memory likely cost more in occupancy than it saved.

- **W2-padding to fix transpose asymmetry**: profiling found one transpose
  direction (`(H, W2) -> (W2, H)`) running ~3x slower than the reverse on
  identical data volume — `W2 = W/2+1 = 1025` isn't a tile multiple, so
  hypothesized cache-line misalignment. Padded `W2` up to 1056 (exact
  `TILE`-multiple) throughout the frequency-domain buffers and pipeline.
  Correct (full-scale numpy match), but end-to-end got *worse* (0.150s vs
  0.137s baseline). Re-profiled the padded code directly: the ~3.5x
  asymmetry was still there, unchanged, even with alignment and boundary-tile
  waste both eliminated. This rules out both misalignment and divergent
  boundary tiles as the cause — the real mechanism is still unknown.

## Real, unresolved finding

One transpose direction in `rfft2`/`irfft2` is consistently ~3-3.5x slower
than the other despite identical data volume and block count, costing
roughly 20-25% of total runtime. Two plausible mechanisms were tested and
ruled out. Diagnosing further needs real profiling tools (`rocprof`/Nsight
Compute) against the actual target hardware — worth pointing `ncu` at this
specific kernel pair on the H100 directly rather than continuing to guess
from an unrepresentative GPU.

## Update: transpose asymmetry root-caused and fixed

Follow-up session, same AMD integrated GPU (gfx1151, RDNA3.5, Ryzen AI
MAX+ 395 — still not the H100 target, caveat still applies).

**Profiling tooling note first:** `rocprof`, `rocprofv2`, and
`rocprof-compute` are all installed on this machine, but every one of them
segfaults/aborts on this GPU before a single kernel runs — the crash is in
`HsaRsrcFactory`/`aqlprofile` agent enumeration at HSA init time, reproduces
with `--stats`, `--hip-trace`, `--hsa-trace`, and `--kernel-trace` alike, and
is independent of the workload (it aborts during ROCm's own resource-factory
setup, not inside any Mojo-generated kernel). `rocprof-compute` additionally
lacks its Python dependencies in this environment. This looks like a known
gap in ROCm's profiler support for RDNA3.5 APUs, not anything fixable from
this repo. Given that, root-causing below was done with host-side
`perf_counter_ns` + `ctx.synchronize()` microbenchmarks isolating individual
kernels/kernel pairs at production scale (D=41, H=W=2048) — less precise
than real hardware counters, but sufficient to get a clean, reproducible,
large (3-4.5x) signal. **This is the main thing worth re-running with actual
`rocprof`/Nsight Compute on the H100**: confirm the mechanism below with real
DRAM/L2 channel-conflict counters instead of wall-clock inference.

**Root cause: DRAM partition camping on power-of-two destination pitch.**
Isolating the two transpose directions from the rest of the pipeline (a
`transpose_kernel[TILE, H, W2]` call vs its reverse, identical D=41 data
volume and identical block count, only the grid-dimension order swapped)
reproduced the reported asymmetry cleanly: 29.3ms vs 9.7ms, a 2.9-3.0x ratio,
matching the original ~3-3.5x estimate. Sweeping controls narrowed it down:

- A square-transpose control (H=W=2048 both directions) showed *no*
  asymmetry (ratio 0.98-1.04) — both directions pay the same "bad" cost.
- A decisive ablation held `grid.x` fixed at 64 in both runs (eliminating
  grid-orientation as a variable) and varied *only* the destination
  tensor's fast (contiguous/last) axis length between 1024 and 1025:
  dest-pitch 1024 (power of two) -> 43.0ms; dest-pitch 1025 (not a power of
  two) -> 9.8ms. **4.4x**, with everything else held constant.

This is the classic GPU-transpose "partition camping" effect: when a
strided write's row pitch is an exact power of two, every concurrently
scheduled block's destination address collides on the same DRAM
channel/bank (the low address bits repeat identically row after row),
serializing what should be parallel memory traffic. In this pipeline, `H`
and `W` are both 2048 (required to be powers of two for the radix-2 FFT),
so *whichever* transpose direction writes to a destination whose fast axis
is `H` or `W` (always true for one of the two directions per call) pays this
cost; the direction writing to a destination with fast axis `W2 = W/2+1 =
1025` (not a power of two) does not. This also fully explains why the prior
session's "pad `W2` to 1056" experiment didn't change anything:
`ceildiv(1025, 32) == ceildiv(1056, 32) == 33`, so that padding never
changed the grid shape or which axis was power-of-two — it was padding a
dimension that was never the problem.

**Fix: diagonal block-index remap**, the classic
NVIDIA-transpose-sample technique for defeating partition camping without
touching any buffer layout. Inside `transpose_kernel`, before computing tile
coordinates:
```
comptime Gx = ceildiv(W, TILE)
comptime Gy = ceildiv(H, TILE)
var bid = block_idx.x + Gx * block_idx.y
var blk_y = bid % Gy
var blk_x = ((bid // Gy) + blk_y) % Gx
```
then use `blk_x`/`blk_y` everywhere `block_idx.x`/`block_idx.y` were used
before. This is a pure bijection over the same set of tiles (every tile
still gets processed exactly once) — it only changes which *hardware-
scheduled* block index handles which logical tile, decorrelating
consecutively-scheduled blocks' destination addresses from the power-of-two
stride. Verified bit-exact (`max|diff| = 0.0`) against the un-remapped
kernel on the isolated microbenchmark, then against the full numpy
reference via the existing test suite. Measured on the previously-slow
direction: 29.3ms -> 8.6ms (3.4x faster), with no regression on the
already-fast direction (9.8ms -> 9.4ms). **Kept** — applied unconditionally
in `transpose_kernel` (`fft_gpu.mojo`), since it helps the slow direction and
doesn't hurt the fast one.

**End-to-end impact of the fix alone** (before any of the fusion work
below): v1 0.1948s -> 0.1378s (29% faster), v2 0.1362s -> 0.0980s (28%
faster). This alone recovered most of the "20-25% of total runtime" the
prior session attributed to this asymmetry.

## Tried and reverted: eliminating the transpose via strided row-FFT

Hypothesis from this session's brief: `fft_row_kernel`'s bit-reversed load
already scatters global reads regardless of whether the row is contiguous
(post-transpose) or strided (pre-transpose), so maybe the transpose isn't
buying as much as it costs. Implemented `fft_row_kernel_strided` — runs the
height-axis FFT directly on the un-transposed `(D, H, W2)` buffer with a
`WStride`-strided column access, and `rfft2_batched_gpu_notranspose`/
`irfft2_batched_gpu_notranspose` that skip both `transpose_kernel` calls
around it entirely. Verified bit-exact against the numpy reference at small
scale. But benchmarked at production scale (D=41, H=W=2048) *after* the
diagonal-remap fix above: 32.1ms (transpose-based) vs 41.2ms
(notranspose), i.e. **28% slower**. Once the transpose itself is fast
(post-fix), the strided kernel's uncoalesced *writes* (previously
contiguous, now `W2`-strided — each thread's global address is 1025
elements from its neighbor's) cost more than the transpose saves. Reverted
in full (kernel and both entry points removed from `fft_gpu.mojo`); the
insight in the brief was reasonable but empirically wrong once the real
bottleneck (partition camping, not the transpose per se) was fixed.

## Kernel-fusion pass (kept)

Three fusions, each removing a full kernel launch and a round trip through
global memory of an intermediate buffer, implemented in `openflr_gpu.mojo`/
`fft_gpu.mojo`. All verified against the numpy-reference test suite
(`test_fft_gpu`, `test_openflr_gpu`) bit-exact/unchanged before and after
(same `max|diff|` as the pre-fusion baseline in every case).

- **`shift_hw_kernel` + `elementwise_mul_kernel` -> `shift_mul_kernel`**
  (v1 only, final two ops): reads `back` at the fftshift-computed index and
  multiplies against `data_buf` directly, eliminating the `shifted` scratch
  buffer and its buffer round trip. `shift_hw_kernel` and the `shifted`
  field on `OpenFlrScratch` were removed (no longer used anywhere). Small
  but consistent win: v1 0.1378s -> 0.1349s (~2%).

- **`elementwise_div_kernel` fused into `rfft_row_kernel`'s load stage ->
  `rfft_row_kernel_div` / `rfft2_batched_gpu_div`** (v1 and v2, the
  `image_buf / denom` step): divides at the point each element is first
  read into shared memory instead of materializing `img_err` first.
  Additive — the original `rfft_row_kernel`/`rfft2_batched_gpu` are
  untouched and still used elsewhere (e.g. PSF prep). The `img_err` scratch
  buffer and `elementwise_div_kernel` were removed (no remaining callers).
  This step operates on an `(H, W)` buffer, much smaller than the
  `(D, H, W)`-scale buffers dominating the rest of the pipeline, so the win
  is real but small in absolute terms: isolated microbenchmark (this stage
  alone, 50 iters) measured 1.66x faster (1.05ms -> 0.63ms, saving
  ~0.42ms/call) — correctly too small to distinguish from run-to-run noise
  (~1ms std) in the full ~135ms end-to-end benchmark. Kept because the
  isolated measurement is unambiguous and there is no downside.

- **`complex_mul_kernel` / `complex_mul_broadcast_depth_kernel` fused into
  the first transpose of the following `irfft2_batched_gpu` call ->
  `irfft2_batched_gpu_cmul` / `irfft2_batched_gpu_cmul_broadcast`** (new
  `transpose_kernel_cmul` / `transpose_kernel_cmul_broadcast` kernels in
  `fft_gpu.mojo` compute the complex product at the transpose's load stage
  instead of reading a pre-materialized product): this is the highest-value
  fusion, since the eliminated product buffer — `(D, H, W/2+1)` complex,
  D=41 — is the single largest buffer in the pipeline. Applied to both of
  v1's `cmul -> irfft2` pairs and v2's second one (v2's first `cmul` feeds a
  depth-reduction before its `irfft2`, so it doesn't fit this pattern and
  was left unfused). `complex_mul_broadcast_depth_kernel` and the separate
  `elementwise_mul_kernel`-driven `cmulb` call sites were removed (no
  remaining callers); `complex_mul_kernel` is kept (still used by v2's
  unfused first step). Verified bit-exact (`max|diff| = 0.0`) against the
  unfused baseline at small scale. Isolated production-scale microbenchmark
  (D=41, H=W=2048, 20 iters): 42.6ms -> 38.2ms, 1.12x, saving ~4.4ms per
  call — and unlike the div fusion, this shows up clearly at the full-
  pipeline level too (see below), since v1 has two such calls per iteration
  and v2 has one.

**Combined effect of all three fusions on top of the transpose fix**: v1
0.1378s -> 0.1268s (~8% further reduction), v2 0.0980s -> 0.0940s (~4%
further reduction).

## Overall before/after (this session)

Same benchmark harness (`pixi run run-v1`/`run-v2`, 20 iterations, mean over
the run):

| | v1 mean | v2 mean |
|---|---|---|
| Baseline (start of this session) | 0.1948s | 0.1362s |
| + transpose diagonal-remap fix | 0.1378s (-29%) | 0.0980s (-28%) |
| + all 3 kernel fusions | 0.1268s (-35% total) | 0.0940s (-31% total) |

## Recommended follow-up on real H100 hardware

- **Re-verify the partition-camping diagnosis with real counters.** The
  mechanism above (DRAM channel/bank collision on power-of-two pitch) is
  inferred from wall-clock ablations, not measured with hardware
  performance counters, because `rocprof`/`rocprofv2`/`rocprof-compute` all
  fail to attach on this gfx1151 APU. NVIDIA's memory-controller/DRAM
  partition-camping counters in Nsight Compute (`dram__throughput`,
  L2 sector hit rate, or the classic "partition camping" guidance in
  NVIDIA's own transpose sample) would confirm or refute this directly on
  the target GPU, and would also confirm whether the diagonal-remap fix is
  restoring full channel/bank utilization there the way it appears to here.
- **Check whether the diagonal remap is still a net win on H100.** H100's
  memory subsystem (HBM3, different channel/partition count and address
  hashing than this RDNA3.5 iGPU's shared DDR) may not exhibit the same
  magnitude of power-of-two-pitch penalty, or may exhibit it differently.
  The remap is cheap (a few extra integer ops per block, no extra memory
  traffic) so it's very unlikely to regress anything, but the *size* of the
  win (3.4x here) should not be assumed to transfer.
- **Re-check the `complex_mul` fusion's relative value.** It was the
  clearest win of the three fusions here because it touches the pipeline's
  largest buffer; that should hold on H100 too, but the *ratio* of compute-
  bound vs. memory-bound cost in the transpose+FFT kernels will differ on a
  GPU with much higher compute throughput relative to memory bandwidth, so
  the specific 1.12x/4.4ms number measured here should be re-measured
  rather than assumed.
- **The `elementwise_div` fusion is low-priority to re-verify** — it's a
  small, provably-correct, no-downside change; whether it's worth the added
  code surface on H100 depends on whether that stage is still proportionally
  small there, which is likely but not verified.
