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

## Follow-up session: what's already exploited, and one more fusion tried

Answering the standing questions directly, against the code as it stands
(same AMD gfx1151 iGPU, same caveat about transferability to H100):

- **Compile-time image size.** Already fully exploited. `D`, `H`, `W`,
  `TILE` are `comptime` parameters threaded through every kernel and host
  function (`OpenFlrScratch[D, H, W]`, `run_v1_step_gpu[D, H, W, TILE]`,
  `rfft2_batched_gpu[D, H, W, TILE]`, ...). `ilog2_ct`/`bit_reverse_ct` run
  at compile time, and `comptime for stage in range(log2n)` fully unrolls
  every FFT butterfly stage — there is no runtime loop over stages, no
  runtime-computed grid/block shape, and no shape branching anywhere in the
  hot path. There isn't more to gain here; this was the starting point, not
  an opportunity.
- **Real-valued data.** Already exploited via `rfft2_batched_gpu`/
  `irfft2_batched_gpu`, which mirror `numpy.fft.rfft2`/`irfft2`: each row's
  real-input FFT is computed as one length-`N/2` complex FFT (the standard
  "pack two reals into one complex FFT" trick, `rfft_row_kernel`/
  `irfft_row_kernel`) rather than a full length-`N` complex FFT with a
  zeroed imaginary part, and only the non-redundant `W/2+1`-wide half
  spectrum is carried through the whole pipeline. This roughly halves both
  the arithmetic and the memory footprint of every FFT-adjacent buffer
  relative to a naive complex implementation.
- **Fusing operations.** Three real fusions already landed (see above:
  `shift_mul_kernel`, `rfft_row_kernel_div`, `transpose_kernel_cmul[_broadcast]`).
  This session tried one more, described below, and reverted it based on
  measurement.

### Tried and reverted: atomic-accumulate fusion of the depth-sum reductions

Both `run_v1_step_gpu` and `run_v2_step_gpu` contain a materialize-then-
reduce pattern that the existing fusions hadn't touched: an elementwise op
writes a full `(D, H, W)`-or-`(D, H, W/2+1)` buffer, and the very next
kernel reads that whole buffer back just to sum it over the depth axis
into a much smaller `(H, W)`/`(H, W/2+1)` result, with no other consumer
in between. Concretely:

- v1: `irfft2_batched_gpu_cmul` writes `scratch.conv` (D, H, W real, one of
  the largest buffers in the pipeline), then `sum_over_depth_real_kernel`
  reads all of it back to produce `denom` (H, W).
- v2: `complex_mul_kernel` writes `scratch.prod_re`/`prod_im` (D, H, W/2+1
  complex — the single largest buffer in the pipeline, per the fusion pass
  above), then `sum_over_depth_kernel` reads it all back to produce
  `reduce_re`/`reduce_im` (H, W/2+1).

Implemented `irfft_row_kernel_accum` (a variant of `irfft_row_kernel` that,
instead of writing each row to its own slot in a `(D, H, W)` buffer,
`Atomic.fetch_add`s its output directly into a pre-zeroed `(H, W)`
accumulator at `row % H`) and `complex_mul_sum_depth_kernel` (elementwise
complex multiply that atomically accumulates the product directly into a
pre-zeroed `(H, W/2+1)` accumulator instead of writing a `(D, H, W/2+1)`
product buffer). Wired as `irfft2_batched_gpu_cmul_sumdepth` (v1) and a
direct replacement of the `complex_mul_kernel`+`sum_over_depth_kernel` pair
(v2). This would eliminate `scratch.conv` and the `(D, H, W/2+1)` product
buffer entirely — both the write *and* the read-back — rather than just
one side of the round trip as the earlier fusions did.

Correctness: bit-exact (`max|diff|` unchanged, `8.94e-08`) against the
unfused baseline on the full test suite (`test_fft_gpu`, `test_openflr_gpu`).

Performance: measured *worse* at production scale (D=41, H=W=2048, mean of
20 iterations, three repeated runs each to check for noise):

| | baseline | fused (atomic accumulate) |
|---|---|---|
| v1 | 0.1249s | 0.1291-0.1309s (+3-5%) |
| v2 | 0.0928-0.0944s (run-to-run range) | 0.0951s (+~1%, close to noise) |

v1's regression is clear and reproducible (the gap is 3-4x the run-to-run
std). v2's is small enough to be arguably within noise, but showed no
measured benefit either. **Reverted in full** (both new kernels and both
call-site changes) rather than kept in an ambiguous state.

**Why this plausibly loses despite eliminating a real memory round trip:**
each output element gets a `Atomic.fetch_add` instead of a plain coalesced
write, and there are `D=41` of these contending on the same address across
the depth axis. On this GPU's atomic hardware, that per-element atomic
overhead apparently costs more than the eliminated buffer's write+read
saved. This is exactly the kind of result that's plausibly hardware-
specific rather than fundamental: NVIDIA GPUs from Volta on have
progressively invested more in global-memory atomic throughput and L2-
resident atomic combining (recent architectures handle heavily-contended
`atomicAdd` far better than older/integrated parts), so the same fusion
could plausibly be a net win on H100 even though it measured as a loss
here. **Worth re-trying on the actual H100 target** if anyone picks this
back up — the code for both kernels is straightforward to reconstruct from
this description (or from git history if this change was ever committed),
and the correctness harness (`test_openflr_gpu`) will immediately confirm
it's still bit-exact before re-measuring performance.

### Other ideas not yet tried (untested, listed for future work)

None of these were implemented or measured this session — they're recorded
because they're the more consequential remaining ideas, in roughly
descending order of expected payoff, and are the natural things to reach
for after the depth-sum fusion above:

- **Interleaved (AoS) complex storage.** Every FFT/transpose buffer pair
  is currently two separate `re`/`im` `DeviceBuffer`s (struct-of-arrays).
  Every transpose and row-FFT load/store is therefore two separate 4-byte
  memory transactions per complex element instead of one 8-byte
  transaction for an interleaved `float2`-per-element layout. This is a
  bandwidth-bound-kernel classic and would touch every buffer and kernel
  signature in `fft_gpu.mojo`/`openflr_gpu.mojo` — high payoff, high
  blast radius, best attempted as its own isolated pass with the existing
  bit-exact test suite guarding every step.
- **Vectorized (`float4`) loads/stores in `transpose_kernel` and its
  `_cmul`/`_cmul_broadcast` variants.** Each thread currently moves one
  `float` per buffer per phase. `W2 = 1025` (not a multiple of 4) makes the
  boundary case for the width-2049-ish transpose direction fiddly to get
  right without a correctness regression — doable, but the boundary-tile
  logic needs care.
- **Warp-shuffle butterflies for the last few FFT stages.** Every stage of
  every FFT kernel goes through shared memory + a full-block `barrier()`,
  even for the final ~5 stages where the butterfly distance is ≤32 and
  the exchange could happen via `warp.shuffle_xor` within a single warp
  with no `barrier()` at all (both this GPU's RDNA3+ wavefront and an
  H100's warp are 32 lanes, so the mapping is direct). This trims
  `log2(32) = 5` block-wide barriers off the tail of every row-FFT
  kernel. Not attempted here; the twiddle-table experiment earlier in this
  log suggests transcendental/arithmetic tweaks inside these kernels have
  had limited impact, but *removing barriers* is a different lever
  (synchronization cost, not compute cost) that hasn't specifically been
  tested.
- **Higher-radix (radix-4/8) FFT stages** to cut the number of
  stages/barriers roughly in half for large `N` (e.g. `H = W = 2048`).
  Larger rewrite, more correctness surface, only worth it if the barrier-
  removal idea above shows the FFT kernels are actually
  synchronization-bound rather than bandwidth-bound at production scale.

## Follow-up session: warp-shuffle fast path, inspired by `gpu-fft`

Prompted by a reference GPU-FFT library dropped into `../gpu-fft/`
(Özcan's `GPU-FFT-Optimization`, https://eprint.iacr.org/2023/1410). Its
kernel design (`src/lib/fft.cu`) already matches this codebase's two
biggest structural choices: it keeps an entire chunk of the transform
resident in shared memory across many butterfly stages per kernel launch
(rather than one global-memory round trip per stage), and — the detail that
motivated this session — its last several butterfly stages inside a kernel
launch **skip `__syncthreads()` entirely**, relying on the stages being
warp-local once the butterfly distance drops below the warp width. That's
exactly the "Warp-shuffle butterflies" idea this log had previously listed
as untested (see above). This session implemented it properly (via Mojo's
`shuffle_xor`, not gpu-fft's bare unsynced-shared-memory trick, which is
undefined behavior on architectures with independent thread scheduling —
see "false start" below) and it is a **small, real, verified win**.

**Correction to the earlier untested write-up first:** that entry said the
warp-local stages were "the final ~5 stages" of each row-FFT kernel. They
are not — `stage_half` (butterfly distance) *increases* every iteration of
`comptime for stage in range(log2n)` (`size = 1 << (stage + 1)`), so the
stages with distance ≤ 32 are the **first** ~6 stages (right after the
bit-reversed load), not the last. This only matters for where in the
kernel the optimization applies; the underlying insight (small-distance
stages are warp-local and don't need a block-wide barrier) was correct.

**The technique:** immediately after the bit-reversed load, thread
`(warp, lane)` owns two array positions `pos_a = 64*warp + lane` and
`pos_b = pos_a + 32` (the two elements a warp collectively owns for every
`stage_half <= 32`; see the index algebra in `fft_row_kernel`'s comment —
`i0 XOR i1 == stage_half` always, and any group of ≤ 64 elements is
threaded by ≤ 32 consecutive `tid`s, i.e. one warp, since 64 = 2 ×
`WARP_SIZE`). Instead of writing the bit-reversed load to shared memory and
running `stage_half = 1, 2, 4, 8, 16, 32` through the normal
shared-memory-read + `barrier()` loop, each thread loads its two values
directly into registers and runs those same 6 stages as: 5 stages of
`shuffle_xor`-based butterflies (exchanging the partner's register value
across lanes, no shared memory, no barrier) on `pos_a`'s and `pos_b`'s
32-lane sub-networks independently, then one **shuffle-free** local combine
between the thread's own two registers for `stage_half = 32` (both operands
are already local once `pos_a`/`pos_b` differ only in bit 5). The result is
written to shared memory once, followed by one `barrier()`, and the
remaining (large-stride) stages continue exactly as before. Net effect for
the production `N = 2048` case (`log2n = 11`): 6 of 11 stages' worth of
`barrier()`s and shared-memory traffic replaced by register/shuffle work.
Applied to all four row kernels (`fft_row_kernel`, `rfft_row_kernel`,
`rfft_row_kernel_div`, `irfft_row_kernel`) — `rfft`/`irfft`'s pack/unpack
load steps differ in shape from `fft_row_kernel`'s, but the internal
length-`half` butterfly network and the fast-path index algebra are
identical, so the same pattern applies directly (`irfft_row_kernel`'s
"reconstruct" load is a scatter, `dst = bit_reverse(k)`, rather than a
gather, but bit-reversal is self-inverse so the value landing at
`pos_a`/`pos_b` is just the reconstruction of `k = bit_reverse(pos_a/b)` —
same gather shape as the others). Gated behind
`comptime if half >= 64 and WARP_SIZE == 32` with the original
shared-memory path kept verbatim as the `else` branch, so it's a no-op
(falls back to the unchanged path) for small transform sizes (the
`N = 8`/`N = 64` correctness tests) and for any future non-32-wide-warp
target.

**False start: naive `if`/`else` role dispatch was 2.4x *slower*, not
faster.** The first implementation used a literal `if is_lo: ... else:
...` to pick the i0-vs-i1 role per lane (mirroring how the math reads on
paper). Verified bit-exact, but an isolated microbenchmark of just
`fft_row_kernel` at production scale (a standalone script launching the
kernel alone 50x with `ctx.synchronize()` timing, D×H = 83968 rows,
N=2048) measured **39.8ms vs. a 16.7ms baseline** — 2.4x *slower* than the
plain shared-memory version it was meant to improve. Root cause: for any
given stage, exactly half the lanes in a warp take each branch of
`is_lo`— textbook intra-warp divergence, where the GPU serializes the two
branches (idle half the lanes, twice, every stage) rather than running
predicated/select instructions. This fully explains the ~2.4x cost and
matches why gpu-fft's own reference kernel doesn't need this concern: it
never introduces a data-dependent branch in the first place, addressing
shared memory directly by computed index instead.

**Fix: made the role dispatch branchless.** Replaced the `if`/`else` with
computing *both* candidate results (the i0-role formula and the i1-role
formula) unconditionally and blending them with a 0.0/1.0 float mask
(`lo_f = Float32(1 - ((lane >> stage) & 1))`, `hi_f = 1.0 - lo_f`,
`result = lo_f*lo_result + hi_f*hi_result`) — pure data-dependent select,
no branch, no divergence, at the cost of a handful of extra multiply-adds
per stage (computing the unused branch's arithmetic too). Re-measured on
the same isolated `fft_row_kernel` microbenchmark: **15.6ms**, i.e. faster
than *both* the divergent version (39.8ms) *and* the original
shared-memory baseline (16.7ms) — about a 7% win on this kernel in
isolation. Verified bit-exact via `pixi run test-fft-gpu`,
`pixi run test-openflr-gpu`, and `verify_correctness.py` (all three
identical to the pre-change baseline — same `max|diff|`/`max abs diff`
values in every case, including the full-scale numpy comparison:
`5.722e-06` before and after).

**End-to-end impact:** measured via several back-to-back
new/baseline/new `pixi run run-v1`/`run-v2` triplets (to control for
session-to-session drift on this shared iGPU, which was ~2% run-to-run on
its own) rather than isolated single runs. Consistently ~2% faster on both
formulations: v1 0.1269s -> 0.1246s (baseline/new pair), 0.1250s (new,
second pair); v2 0.0944s -> 0.0923s, 0.0922s (new, second pair). Smaller
than the isolated single-kernel win (7%) because the row-FFT kernels are
only one part of the pipeline (transposes and elementwise-fused kernels
make up the rest), consistent with this log's earlier finding that FFT row
kernels are "a small slice of total time to begin with."

**Worth re-verifying on the actual H100 target:** the core mechanism
(warp-local shuffle avoiding shared-memory barriers) is architecture-
generic and should transfer, but the *relative* payoff depends on how
barrier/shared-memory cost compares to compute throughput on H100 vs. this
RDNA3.5 iGPU — the same caveat that applies to every finding in this log.
The intra-warp-divergence lesson (branchless role dispatch, not `if`/
`else`) is architecture-independent and should hold regardless: divergent
branches cost real cycles on any SIMT GPU, NVIDIA included.

## Follow-up session: higher-radix (radix-4) FFT stages — kept, small win

Implemented the "Higher-radix (radix-4/8) FFT stages" idea this log had
previously listed as untested. Targeted the large-stride stages that the
warp-shuffle work above *doesn't* cover: after the first ~6 stages run
warp-locally via `shuffle_xor`, every remaining stage (`stage_half > 32`)
still goes through shared memory with one `barrier()` per stage. Radix-4
fusion rewrites two consecutive radix-2 stages as one radix-4 stage,
halving the barrier()/shared-memory round trips for that tail.

**Derivation.** For two chained radix-2 DIT stages (sizes `m` then `2m`),
combining 4 elements `a0..a3` at offsets `0, m/2, m, 3m/2` within a `2m`
group reduces algebraically to:

```
wA = W_m^k                                  (k in [0, m/2))
tA = a1*wA;  y0 = a0+tA;  y1 = a0-tA
tB = a3*wA;  y2 = a2+tB;  y3 = a2-tB

w0 = W_{2m}^k
t0 = y2*w0;               z0 = y0+t0;  z2 = y0-t0
tC = y3*w0;  t1 = tC * j*sign;  z1 = y1+t1;  z3 = y1-t1
```

where `j*sign` (the twiddle for `W_{2m}^{k+m/2}` relative to `W_{2m}^k`, a
quarter turn) is a free negate-and-swap: `t1 = (-sign*tC.im, sign*tC.re)`.
This is a pure algebraic regrouping of the existing two-stage math — same
butterfly, same twiddle factors, no new approximation — with only `N/4` of
the `N/2` launched threads doing work per fused stage (each now produces 4
outputs instead of 2); the rest idle until the shared `barrier()`. A
leftover odd stage (when the remaining stage count is odd) still runs as a
single plain radix-2 stage.

**Applied** to all four row kernels (`fft_row_kernel`, `rfft_row_kernel`,
`rfft_row_kernel_div`, `irfft_row_kernel`), in both the warp-shuffle
fast-path's tail loop (`stage in range(6, log2n)`) and the small-N `else`
branch's full loop (used when `N < 64`, and effectively dead code at
production scale but converted the same way for consistency and because
the existing `N=8` unit test already covers it).

**Correctness.** Added `test_fft_gpu_radix4.mojo` (wired into
`pixi run test-fft-gpu`): drives all four kernels at N in
{128, 256, 512, 1024, 2048} — chosen so `log2(N)`/`log2(N/2)` sweeps every
fused-pair-count/leftover-stage parity combination the production sizes
hit — against the arbitrary-length CPU reference in `fft_cpu.mojo` on
deterministic pseudo-random input, since the pre-existing small-N tests
(N=8, N=64) never exercised this tail loop at all (N=64's tail range is
empty; N=8 only hits the `else` branch). All pass at the same float32
precision as an unfused reference FFT. Also verified bit-close
(`max|diff| = 6.676e-06`) against the full numpy reference at production
scale (D=41, H=W=2048) via `verify_correctness.py`, and unchanged on the
existing `test_fft_gpu`/`test_fft2_gpu`/`test_openflr_gpu` suites.

**Performance.** Measured with the same isolate-and-revert methodology as
the transpose fix above: reverted just the four fast-path tail loops back
to plain per-stage radix-2 (leaving everything else, including the
warp-shuffle work, unchanged) for a same-session baseline, then restored
and re-measured, several `pixi run run-v1`/`run-v2` trials each way to
control for the ~1% run-to-run drift on this shared iGPU:

| | v1 mean (3 trials) | v2 mean (3 trials) |
|---|---|---|
| Baseline (warp-shuffle only, tail loops un-fused) | 0.1253s | 0.0935s |
| + radix-4 fusion of tail loops | 0.1240s (-1.0%) | 0.0929s (-0.6%) |

A small, real, reproducible win for v1 (radix-4 trials clustered
0.1239–0.1242s, baseline trials 0.1252–0.1254s — non-overlapping); v2's
improvement is smaller and closer to run-to-run noise. No regression
observed in either version across any trial. Consistent with this log's
repeated finding that the row-FFT kernels are a small slice of total
runtime — even fully removing half the remaining barriers in their
tail stages only moves the end-to-end needle by ~1%, similar in
magnitude to the warp-shuffle fast path's own end-to-end win. **Kept**,
since it's a verified-correct, no-downside change with a small positive
effect, but it's a good example of the previously-recorded caveat: this
was "only worth it if the barrier-removal idea shows the FFT kernels are
actually synchronization-bound" — they're only *mildly* so at this scale
on this GPU, so treat this as a modest win rather than the "cut stages
roughly in half" payoff the original untested note speculated about.

**Worth re-verifying on the actual H100 target:** same caveat as every
finding in this log — the algebraic fusion itself is architecture-generic,
but whether it's worth the added code surface depends on whether H100's
better barrier/shared-memory throughput (relative to its much higher
compute throughput) makes stage-count/barrier-count matter more or less
there than it does on this bandwidth-constrained iGPU.
