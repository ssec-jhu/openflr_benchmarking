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

---

# Session: rocFFT-guided investigation

Brief: compare against the ROCm rocFFT implementation (`../rocfft` ->
`rocm-libraries/projects/rocfft`) to find optimization opportunities. No
code was changed this session. Hardware: same AMD gfx1151 (Radeon 8060S /
Ryzen AI MAX+ PRO 395, 40 CU, wave32, LPDDR5X unified memory).

Two things changed the picture materially versus every prior session:

1. **`rocprofv3` works on this GPU.** Prior sessions concluded profiling
   was impossible here (`rocprof`/`rocprofv2`/`rocprof-compute` all abort
   during HSA agent enumeration). ROCm 7.2 ships `rocprofv3`, which
   attaches cleanly to the Mojo binary and costs ~1% runtime overhead. We
   now have real per-kernel dispatch data instead of host-side
   isolate-and-time inference.
2. **A measured achievable-bandwidth number for this GPU: ~232 GB/s.**
   Everything below is interpreted against that ceiling.

## 0. Headline: we are not actually behind PyTorch/JAX per unit of bandwidth

This reframes the whole task, so it goes first.

Measured achievable bandwidth on this GPU (HIP `float4` streaming
kernels, 1 GiB buffers, 20 iterations):

| test | GB/s |
|---|---|
| copy (read+write) | 232.5 |
| read-only | 237.9 |

(Theoretical peak for Strix Halo's 256-bit LPDDR5X-8000 is 256 GB/s, so
232 GB/s is ~91% of theoretical — a normal streaming ceiling.)

A byte-traffic model of the pipeline — every kernel's global reads plus
writes at D=41, H=W=2048, W2=1025, per the per-kernel `MB` column in
§2 — gives **20.15 GB per v1 iteration** and **16.12 GB per v2
iteration**.
Dividing the published times in `../results.md` by that same traffic
model, against each GPU's theoretical peak bandwidth:

| GPU (peak BW) | v1 backend/time | v1 % of peak | v2 backend/time | v2 % of peak |
|---|---|---|---|---|
| H100 SXM (3350 GB/s) | torch 0.0118 s | 51% | torch 0.00668 s | 72% |
| H100 SXM (3350 GB/s) | jax 0.00973 s | 62% | jax 0.00631 s | 76% |
| A100 SXM (2039 GB/s) | jax 0.0172 s | 57% | jax 0.0112 s | 71% |
| L40S (864 GB/s) | jax 0.0402 s | 58% | jax 0.0252 s | 74% |
| V100 PCIe (900 GB/s) | jax 0.0357 s | 63% | jax 0.0236 s | 76% |
| **gfx1151 (256 GB/s)** | **mojo 0.1197 s** | **66%** | **mojo 0.0900 s** | **70%** |

Every backend on every GPU lands in a 51-76% band, and the Mojo
implementation is at the **top** of the band for v1 and inside it for v2.
Against the *achievable* 232 GB/s rather than theoretical, Mojo is at 72%
(v1) and 77% (v2).

The interpretation: **this pipeline is memory-bandwidth-bound on every
GPU tested, all implementations run at a similar fraction of their
hardware's bandwidth, and the 10-14x absolute gap to H100 is almost
entirely the 13x bandwidth gap between LPDDR5X-8000 and HBM3.** The Mojo
kernels are not the problem.

Caveat: this assumes cuFFT/XLA move a similar number of bytes to our
model. That is an assumption, not a measurement — but the consistency of
the 51-76% band across four GPUs spanning a 3.9x bandwidth range is
strong evidence that all of them are bandwidth-bound with comparable
efficiency. **Worth confirming directly** by running `main.py` with a
ROCm build of torch on this machine (see §6) — the currently-installed
`.venv` torch is a CUDA build (`2.13.0+cu126`, `torch.version.hip is
None`), so there is no same-hardware reference today.

Consequence for prioritization: micro-optimizing kernels can win at most
~1.3x (72% -> 100% of achievable). **Moving fewer bytes is the only lever
with a big payoff**, and §3 shows there are ~33% of bytes available to
remove.

## 1. Profiling is unblocked: use `rocprofv3`

```
pixi run -- /opt/rocm/bin/rocprofv3 --kernel-trace --stats \
    -d <outdir> -o v1 --output-format csv \
    -- mojo run src/main.mojo -- v1 3
```

Produces `<outdir>/v1_kernel_trace.csv` (per-dispatch start/end
timestamps, grid/block dims, LDS bytes, VGPR/SGPR counts) and
`v1_kernel_stats.csv` (per-kernel aggregates). Overhead measured at ~1%
(v1 0.1216 s under the profiler vs 0.1197 s without), so the numbers are
trustworthy. `--stats` alone is enough for a breakdown; `rocprofv3-avail`
lists hardware counters if per-kernel DRAM/L2 counters are wanted next.

This retires the standing "needs real profiling tools on the target
hardware" caveat in the earlier sections — at least for kernel-level
timing on *this* GPU. It also means every prior host-side
`perf_counter_ns` isolation result in this log can now be re-checked
cheaply.

## 2. Measured per-kernel breakdown (one steady-state iteration)

Achieved GB/s = (modelled bytes) / (measured duration). Compare against
the 232 GB/s ceiling.

### v1 — 121.6 ms total (matches the 0.1197 s benchmark)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6226 | 1376 | **221** |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8511 | 1377 | 162 |
| 3 | `fft_row_kernel` (H, fwd) | 7079 | 1377 | **195** |
| 4 | `transpose_kernel` (W2,H)->(H,W2) | 9598 | 1377 | 143 |
| 5 | `transpose_kernel_cmul` | 14865 | 2066 | 139 |
| 6 | `fft_row_kernel` (H, inv) | 6832 | 1377 | **202** |
| 7 | `transpose_kernel` | 9291 | 1377 | 148 |
| 8 | `irfft_row_kernel` (W) | 6159 | 1376 | **223** |
| 9 | `sum_over_depth_real_kernel` | 3294 | 705 | **214** |
| 10 | `rfft2_batched_gpu_div` (4 kernels, (1,H,W)) | 499 | 151 | — |
| 11 | `transpose_kernel_cmul_broadcast` | 16304 | 1394* | **85*** |
| 12 | `fft_row_kernel` (H, inv) | 6850 | 1377 | 201 |
| 13 | `transpose_kernel` | 9494 | 1377 | 145 |
| 14 | `irfft_row_kernel` (W) | 6151 | 1376 | **224** |
| 15 | `shift_mul_kernel` | 10411 | 2064 | **198** |
| | **total** | **121564** | 20147 | 166 |

### v2 — 90.1 ms total (matches the 0.0900 s benchmark)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6272 | 1376 | **219** |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8413 | 1377 | 164 |
| 3 | `fft_row_kernel` (H, fwd) | 7168 | 1377 | **192** |
| 4 | `transpose_kernel` (W2,H)->(H,W2) | 9434 | 1377 | 146 |
| 5 | `complex_mul_kernel` | 9606 | 2066 | **215** |
| 6 | `sum_over_depth_kernel` | 3200 | 705 | **220** |
| 7 | `irfft2_batched_gpu` (4 kernels, (1,H,W)) | 440 | 101 | — |
| 8 | `rfft2_batched_gpu_div` (4 kernels) | 588 | 151 | — |
| 9 | `transpose_kernel_cmul_broadcast` | 13555 | 1394* | **103*** |
| 10 | `fft_row_kernel` (H, inv) | 6807 | 1377 | **202** |
| 11 | `transpose_kernel` | 9444 | 1377 | 146 |
| 12 | `irfft_row_kernel` (W) | 6155 | 1376 | **224** |
| 13 | `elementwise_mul_kernel` | 9009 | 2064 | **229** |
| | **total** | **90091** | 16118 | 179 |

\* The `_cmul_broadcast` MB figure counts the broadcast `(H, W2)` complex
operand once (16.8 MB), assuming it stays cache-resident across the 41
depth slices. Its apparent GB/s is well below every other kernel's,
which says the assumption is at least partly wrong: at the other
extreme, a full re-read per depth slice (41 x 16.8 = 689 MB, ~2066 MB
total) puts it at 127 GB/s (v1) / 152 GB/s (v2) — in the same band as
the plain transposes. The truth is somewhere between, so this kernel is
moving up to ~1.5x the bytes it needs to. See §4d.

### What this settles, decisively

- **The transposes are the bottleneck.** Full-size transposes (including
  the two `_cmul` variants, which are transposes) are
  8511+9598+14865+9291+16304+9494 = **68.1 ms of v1's 121.6 ms (56%)**
  and 8413+9434+13555+9444 = **40.8 ms of v2's 90.1 ms (45%)**. They run
  at 139-164 GB/s (60-71% of achievable), and the two `_cmul` variants
  are worse still.
- **The FFT row kernels are essentially optimal: 192-224 GB/s, i.e.
  83-96% of the 232 GB/s ceiling.** Every remaining FFT-kernel idea in
  this log (radix-16, twiddle tables, `half_lds`, barrier removal) is
  competing for at most 4-17% of a group of kernels that is only ~24%
  of runtime — a ceiling of ~2-4% end-to-end. This confirms, with
  hardware data rather than inference, the conclusion this log reached
  three separate times by other means. **Stop optimizing the butterfly
  math.**
- **The elementwise kernels are also already optimal** (198-229 GB/s).
  There is nothing to gain by making them faster — only by making them
  not exist (§3).
- Register usage is tiny everywhere (VGPR 8-56), LDS is 8.2-16.4 KB, so
  nothing is occupancy-limited by resources. The transposes' problem is
  their access pattern and their tiny per-thread work, not resources.

## 3. What rocFFT says about our top-level structure (and what it doesn't)

Read `library/src/tree_node_2D.cpp`, `tree_node_real.cpp`,
`node_factory.cpp`, `fuse_shim.cpp`, `assignment_policy.cpp`,
`rtc_transpose_gen.cpp`, `rtc_transpose_kernel.cpp`,
`device/generator/stockham_gen_{base,rr,cc,rc}.h`, and
`device/kernels/configs/config_{sbrr,sbcc}.py`.

### Our RTRT structure is the same choice rocFFT makes for this size

rocFFT's 2D scheme preference (`NodeFactory::Decide2DScheme`,
`Real2DEvenNode::BuildTree_internal`) is:

1. `CS_KERNEL_2D_SINGLE` — whole 2D transform in LDS. Requires
   `length0*length1*transforms_per_block*8 <= ldsSize/1.5`; at
   2048x2048 that is 32 MB of LDS. Not applicable.
2. `CS_2D_RC` / `INPLACE_SBCC` — row FFT then a **Stockham Block Column
   Common (SBCC)** kernel for the column dimension, with **no transposes
   at all** (2 kernels). Requires `has_SBCC_kernel(length[1])`.
3. `CS_2D_RTRT` / `TR_PAIR` — row FFT, transpose, row FFT, transpose.
   Explicitly the "last resort".

**`config_sbcc.py` tops out at length 512.** So for H=2048,
`has_SBCC_kernel(2048)` is false and rocFFT falls all the way through to
RTRT — exactly what `rfft2_batched_gpu`/`irfft2_batched_gpu` already do.

rocFFT also has a fuse shim that *would* absorb the transpose into the
row-FFT kernel by simply making it write with the transposed output
stride (`RTFuseShim`, `FT_STOCKHAM_WITH_TRANS` — no extra LDS, no extra
kernel). It is gated by `canOptimizeWithStride`, which demands
`transforms_per_block >= 8` for single precision ("ensure we are doing
enough rows to coalesce properly"). For length 2048,
`transforms_per_block == 1`, so **rocFFT does not fuse the transpose
either.**

This is worth stating plainly because it retires a family of ideas:
- The earlier "eliminate the transpose via strided row-FFT" experiment
  (reverted, 28% slower) failed for the right reason. Its
  uncoalesced `W2`-strided writes are precisely what SBCC avoids by
  giving each block many adjacent columns — and the reason rocFFT can't
  use SBCC at 2048 is that `transforms_per_block * length * 8 bytes`
  won't fit in LDS with enough columns to coalesce (16 columns x 2048 x
  8 B = 256 KB vs 64 KB available). **A transpose-free single-pass
  column FFT is not possible at H=2048.** Don't retry it.
- A two-pass SBCC decomposition (H = 2048 = 64 x 32, rocFFT's `CS_L1D_CC`
  pattern) *is* possible and transpose-free, but costs two full passes
  over the (D,H,W2) buffer — 2754 MB — which is the same traffic as the
  one transpose + one column-FFT pass that §3's cancellation idea leaves
  behind. Same bytes, far more code. **Don't do it.**
- Fusing the transpose into the row-FFT's store is not available at this
  size for the same coalescing reason rocFFT rejects it.

So the structural win has to come from something rocFFT *can't* do,
because rocFFT is a general FFT library and doesn't get to see the
surrounding pipeline. That is exactly where the remaining 33% is.

### The big win: cancel the redundant transposes across the rfft2/irfft2 boundary

The kernel sequences are:

```
rfft2_batched_gpu:            rfft_row(W) -> T_A -> fft_row(H) -> T_B
irfft2_batched_gpu[_cmul]:    T_A' -> fft_row(H,inv) -> T_B' -> irfft_row(W)
```

`T_B` maps `(D,H,W2) -> (D,W2,H)`... and `T_A'` maps it straight back.
**Every `rfft2 -> (elementwise) -> irfft2` chain in the pipeline
performs a transpose immediately followed by its own inverse.** The
elementwise multiply between them commutes with the transpose, so both
can be dropped if the frequency-domain buffers are kept in the
**transposed `(D, W2, H)` layout as the canonical layout** and
`psf_fft`/`psft_fft` are precomputed transposed (free — PSF prep is
outside the timed loop; just drop the final `transpose_kernel` call in
the `rfft2_batched_gpu` used for PSF setup in `main.mojo`).

This is measured, not modelled: dispatches 4 and 5 in the v1 table
(`transpose_kernel` 9598 µs + `transpose_kernel_cmul` 14865 µs = **24.5
ms, 20% of v1's runtime**) are a transpose immediately followed by its
inverse-with-a-multiply-folded-in.

### Four concrete restructurings, with measured payoff

Ordered by measured value. All four preserve bit-exact arithmetic (no
reassociation, no reordering of any float operation — only the *layout*
of intermediate buffers and which kernel does the load/store changes),
so `test_openflr_gpu` / `verify_correctness.py` should report identical
`max|diff|` values, which makes each one cheap to validate.

**(a) Transposed canonical spectrum + fuse forward column FFT, complex
multiply, and inverse column FFT into a single kernel.** (v1's first
chain.) Both column FFTs operate on the *same* `(D*W2, H)` rows. In one
kernel: load the row into LDS, forward FFT, multiply by
`psf_fft_T[d,w,:]`, inverse FFT, store. One LDS residency instead of two
kernels plus two transposes plus an intermediate global buffer.

Replaces v1 dispatches 3,4,5,6 (7079+9598+14865+6832 = **38.4 ms**) with
one kernel moving 2066 MB, which at the 195-202 GB/s the existing column
kernels already achieve is **~10.3 ms**. **Saves ~28.0 ms (23% of v1).**

**(b) Fuse the broadcast complex multiply into the inverse column FFT's
load** instead of into a transpose. With `psft_fft` stored as
`(D,W2,H)` and `rfft2_batched_gpu_div` emitting `(1,W2,H)` (drop its
final transpose — a ~34 MB kernel), the leading transpose disappears
entirely and the multiply happens at the point the inverse column FFT
first reads each element.

Replaces v1 dispatches 11,12 (16304+6850 = **23.2 ms**) with one kernel
moving ~1394 MB at ~200 GB/s = **~7.0 ms**. **Saves ~16.2 ms (13% of
v1).** Same change in v2 replaces dispatches 9,10 (13555+6807 = 20.4 ms)
with ~7.0 ms, **saving ~13.4 ms (15% of v2)**.

One implementation detail that decides whether this actually hits
~1394 MB: the fused kernel's grid is `D*W2` rows, and the broadcast
operand row `err_fft_T[w, :]` is shared by all `D` blocks with the same
`w`. Launched naively (`row = block_idx.x`, i.e. `w` fastest over the
`(D, W2, H)` layout) those `D` blocks are scheduled `W2` apart and the
16.8 MB operand gets swept `D` times — reintroducing exactly the ~689 MB
of avoidable traffic §4d diagnoses in the kernel this replaces. Map the
block index so `d` is fastest instead (`d = block_idx.x % D`,
`w = block_idx.x / D`, buffer row `d*W2 + w`); that keeps the 16 KB
shared row hot in L1 across all 41 blocks and needs no layout change.

**(c) Fuse the final elementwise multiply into `irfft_row_kernel`'s
store.** v1's `shift_mul_kernel` (10411 µs) and v2's
`elementwise_mul_kernel` (9009 µs) each read a full `(D,H,W)` buffer
that the immediately preceding `irfft_row_kernel` just wrote. Fold it
in: the kernel already holds the finished row in LDS. For v1 the
fftshift is a pure index rotation — the row rotation picks the
destination row (`r = (rr + H/2) % H`) and the column rotation makes the
store two contiguous half-row segments, so both the `data` read and the
`out` store stay coalesced.

Replaces v1 dispatches 14,15 (6151+10411 = **16.6 ms**) with one kernel
moving 2064 MB at ~220 GB/s = **~9.4 ms**. **Saves ~7.2 ms (6% of v1)**;
v2 dispatches 12,13 (6155+9009 = 15.2 ms) -> ~9.2 ms, **saving ~6.0 ms
(7% of v2)**.

This is the same trick as the already-kept `shift_mul_kernel` fusion,
pushed one kernel further up the chain.

**(d) Fuse the complex multiply into `sum_over_depth_kernel`** (v2
only). `complex_mul_kernel` writes the pipeline's largest buffer
`(D,H,W2)` complex and `sum_over_depth_kernel` reads all of it back just
to reduce over depth. **Crucially, this needs no atomics**: the existing
`sum_over_depth_kernel` already walks `b in range(D)` reading
`x[b*hw + i]` in a per-thread register accumulator, so folding the
multiply into that loop costs nothing extra and the whole product buffer
disappears — write *and* read-back.

This is the fusion the earlier "atomic-accumulate fusion of the
depth-sum reductions" experiment was reaching for, but it got there the
expensive way: that attempt put the reduction in the *producer*
(`complex_mul_sum_depth_kernel` atomically accumulating into a pre-zeroed
accumulator), which needed `D=41` atomics contending per output element
and measured 1-5% *slower*. Doing it in the *consumer* instead needs no
atomics at all. The earlier session's note that "the same fusion could
plausibly be a net win on H100" is beside the point — it can be a win
here, in a form that never touches an atomic.

Replaces v2 dispatches 5,6 (9606+3200 = **12.8 ms**) with one kernel
moving ~1394 MB at ~220 GB/s = **~6.3 ms**. **Saves ~6.5 ms (7% of v2).**

### Projected result

Summing the measured per-dispatch numbers with (a)-(d) applied:

| | current | projected | speedup |
|---|---|---|---|
| v1 | 0.1216 s | **~0.070 s** | 1.7x |
| v2 | 0.0900 s | **~0.055 s** | 1.6x |

Traffic drops from 20.15 -> ~13.4 GB (v1) and 16.12 -> ~10.7 GB (v2),
about **-33% each**, and kernel count per iteration drops from 19 to 14
(v1) and 19 to 15 (v2). At that point the pipeline would be running at a
*higher* fraction of its hardware's bandwidth than any torch/jax number
in §0's table.

Suggested order of attack: (b) and (d) are the best value-per-line and
are independent of each other; (c) is straightforward; (a) is the
biggest single win but the most involved (it needs a new fused kernel and
the transposed-canonical-layout change threaded through
`OpenFlrScratch` and both `run_v*_step_gpu`). All four share the
transposed-layout prerequisite except (c) and (d).

## 4. rocFFT's transpose kernel vs ours

Even after §3, three full-size transposes remain in v1 (~27.3 ms at
143-162 GB/s) and two in v2 (~17.9 ms). rocFFT's transpose
(`rtc_transpose_gen.cpp`, `rtc_transpose_kernel.cpp`) differs from
`transpose_kernel` in four ways.

**(a) Tile shape and elements per thread — the top micro-optimization.**
rocFFT single precision uses `tileX = 64, tileY = 16`, block `(64,16)` =
1024 threads, LDS tile `64 x 64`, and therefore
`elems_per_thread = tileX/tileY = 4`, with `#pragma unroll` on the read,
LDS-transpose, and write loops. Ours is `TILE=32`, block `(32,32)` =
1024 threads, **1 element per thread per plane**.

Our threads each move 8 bytes (one float from `re`, one from `im`) per
phase. That is far too little to build up memory-level parallelism: the
kernel spends its time on index arithmetic and one round of load
latency it can't hide. rocFFT's threads have 4 independent loads in
flight, and its 64-wide tile rows are 64 contiguous elements (256 B per
plane) instead of 32 (128 B). Given that our transposes sit at 60-71%
of achievable while every kernel that does more work per thread sits at
83-96%, this is the most likely single explanation. **Highest-value
micro-optimization; try `TILE=64` with block `(64,16)` and 4 elements
per thread first.**

**(b) Interleaved (AoS) complex storage.** rocFFT's internal temp buffers
are always `rocfft_complex<T>` (interleaved); planar layout is only a
user-facing option, implemented by a late `make_planar` source
transform. Our buffers are all separate `re`/`im` `DeviceBuffer`s.
Interleaving halves the number of memory instructions and address
computations and the number of concurrently-open DRAM streams. Note the
correction to the earlier untested write-up in this log: SoA does *not*
halve the achievable bandwidth — both `re` and `im` rows are separately
coalesced — so the upside is instruction count and stream count, not
bytes. Bounded by the ~1.4x headroom on these kernels. **Do it after
(a), and only if (a) doesn't already close the gap** — it touches every
buffer and kernel signature in `fft_gpu.mojo`/`openflr_gpu.mojo`.

**(c) Row-pitch padding — and the earlier padding experiment padded the
wrong thing.** rocFFT pads intermediate buffer *strides*
(`AssignmentPolicy::PadStride`, `assignment_policy.cpp:948`):

```
needsPadding = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0))
               && (biggerDim >= 512);
static const size_t padding = 64;   // elements, added to the highest dim's stride
```

i.e. for a `(W2, H)` temp buffer it would allocate row pitch `H + 64 =
2112` rather than `2048`. The earlier "W2-padding to fix transpose
asymmetry" experiment (reverted, made things worse) padded `W2`
1025 -> 1056 — a *length*, on the dimension that was already not a power
of two. rocFFT's rule pads the *stride* on the dimension that *is* a
power of two. That is a different change and was never tried.

It is likely redundant with the diagonal remap already in the tree —
note that rocFFT enables its own diagonal reordering only when
`(fastOut % 256) == 0 && (node.outStride[0] % 256 == 0)`
(`rtc_transpose_kernel.cpp:88`), and a padded pitch of 2112 fails that
test (2112 % 256 = 64), so in rocFFT the two are **alternatives, not
complements**. Independent confirmation of the partition-camping
diagnosis this log root-caused: rocFFT's threshold is exactly "the
destination's fast axis and row stride are multiples of 256". Low
priority — try it only as an A/B against the diagonal remap, not as an
addition.

Also worth a one-line change: our diagonal remap is applied
unconditionally, including on the direction whose destination fast axis
is `W2 = 1025` (not a power of two), where rocFFT would skip it. Gating
it on a comptime `is_pow2(dest_fast_axis)` costs nothing and removes
dead integer work from half the transpose calls.

**(d) `transpose_kernel_cmul_broadcast` is the worst kernel in the
pipeline (85-103 GB/s) and its grid ordering is why.** It launches
`grid = (ceildiv(W2,TILE), ceildiv(H,TILE), D)` with depth as the
*slowest*-varying dimension, and the broadcast `(H,W2)` operand is
indexed `a_re[y, x]` — independent of `block_idx.z`. So all 41 depth
slices read the same 16.8 MB of `a`, but consecutively-scheduled blocks
share a `z` and sweep the whole of `a` before `z` advances. 16.8 MB
would fit this APU's 32 MB MALL/Infinity Cache on its own — but the
kernel simultaneously streams ~1.4 GB through that cache (`b` in, `dst`
out), so `a` is repeatedly evicted between one `z` and the next.
Charging a full re-read per slice (+689 MB) puts the kernel at
127 GB/s (v1) / 152 GB/s (v2), i.e. right in the plain-transpose band —
so the deficit is extra *traffic*, not a slow kernel. Direct evidence
that this is cache-state-dependent: the identical kernel on identical
data takes 16.3 ms in v1 and 13.6 ms in v2, a 20% spread no other
kernel in either trace shows.

**Making depth the fastest-varying grid dimension** (swap `z` into `x`,
or fold `D` into the tile index so the 41 blocks sharing an `a` tile are
scheduled together) would let the 16.8 MB operand be read once instead
of up to 41 times — worth up to ~4 ms in v1 and ~3 ms in v2 on its own. It is a small change to a
kernel §3(b) proposes to delete outright, so its main value is as a
cheap standalone win to bank first, or as a fallback if §3(b) stalls.
Either way the same broadcast-reuse hazard applies to §3(b)'s
replacement fused column-FFT kernel, which reads the same broadcast
operand — **get the grid ordering right there from the start**. Note the
diagonal remap in `transpose_kernel` only permutes `x`/`y`, so it does
not conflict with folding `D` into the fastest-varying axis.

One place we are *better* than rocFFT: rocFFT's LDS tile is unpadded
`tileX x tileX`, so its LDS *write* (`lds[threadIdx.x][...]`, stride
`tileX` complex) is bank-conflicted; ours pads to `TILE+1` and is
conflict-free on both the write and the transposed read. Keep the
padding when moving to a 64x64 tile (`row_major[64, 65]()` → 2 x 64 x 65
x 4 = 33.3 KB, which fits, but check occupancy; if it hurts, the
alternative is a 64x64 tile of interleaved complex with a different
padding).

## 5. rocFFT row-kernel techniques — recorded, but deprioritized

These are the substantive differences between rocFFT's Stockham kernels
and our row kernels. §2 shows our row kernels are at 83-96% of
achievable bandwidth, so **the total available win from this entire
section is ~2-4% end-to-end.** Recorded for completeness and in case the
work is ever ported to a compute-bound GPU, but none of it should be
attempted before §3.

- **Stockham autosort: no bit-reversal permutation, anywhere.** This is
  the most interesting one. rocFFT's global load
  (`stockham_gen_rr.h:load_from_global`) is linear and fully coalesced —
  `idx = thread + h*width`, `LoadGlobal{buf, offset + idx*stride0}`.
  The self-sorting is achieved purely by asymmetric LDS indexing between
  passes: store at
  `(tid/cumheight)*(width*cumheight) + tid%cumheight + w*cumheight`,
  load at `tid + w*length/width`. Our kernels open with a **bit-reversed
  global gather** (`src_a = bit_reverse_ct(pos_a, log2n)`), where
  consecutive lanes in a warp read addresses 64 elements (256 B) apart —
  32 distinct cache lines per instruction. The row (8 KB/plane) fits in
  L0/L1 so the *bytes* are not amplified, but the address-coalescing
  hardware issues 32 requests where it could issue 1-2. Measured at
  219-224 GB/s, the ceiling here is ~4%; still, this is the cleanest
  structural improvement available in the row kernels if anyone wants
  it, and it would also delete `bit_reverse_ct` from the hot path.
- **Mixed radix, very few passes.** For length 2048,
  `config_sbrr.py:456` specifies `workgroup_size=256`,
  `threads_per_transform=256`, `factors=(16, 16, 8)` — **3 passes**,
  8 elements per thread held in registers, radix-16 butterflies done
  entirely in registers with hardcoded internal twiddles
  (`rocfft_butterfly_template.h:FwdRad16B1`). Ours: 1024 threads, **2
  elements per thread**, 11 radix-2 stages (6 warp-shuffle + a radix-4
  fused tail). rocFFT does roughly 3x fewer twiddle multiplies. Note
  rocFFT's choice of a **256-thread block with 8 elements per thread**
  is the same "more work per thread" principle as §4(a); our 1024-thread,
  2-element-per-thread configuration is the opposite everywhere.
- **`half_lds`** (default `True` for `CS_KERNEL_STOCKHAM`): between
  passes, store/load only the real component, then only the imaginary,
  halving LDS per transform (16 KB -> 8 KB for N=2048) at the cost of
  ~2 extra barriers per pass boundary. Doubles occupancy headroom.
- **`direct_to_from_reg`** (default `True`): the first pass loads
  global -> registers directly and the last pass stores registers ->
  global directly, skipping two LDS round trips. Our warp-shuffle fast
  path already does the load half of this; worth checking whether the
  radix-4 tail's final stage writes to LDS only to be re-read for the
  global store, which would be a free removal.
- **Precomputed per-radix twiddle tables** (`twiddles.cpp`), sized
  `sum over passes of (radix-1)*cumheight` rather than the full `N`.
  This log's earlier twiddle-table experiment used a full-length table
  and measured neutral-to-slower; the per-radix table is far smaller and
  LDS-friendly. Still bounded by the ~4% ceiling above.
- **Buffer-load/store intrinsics** (`IntrinsicLoad`/`IntrinsicStore`,
  `llvm.amdgcn.raw.buffer.*`) let rocFFT use hardware bounds-checking
  instead of `if` guards, removing branches from the load/store path.
  AMD-specific; unclear whether Mojo exposes an equivalent.
- **Callbacks / `load_store_ops`.** rocFFT lets callers fuse arbitrary
  elementwise work into any kernel's load and store via callbacks
  (`callback_h`, `LoadGlobal`/`StoreGlobal`), plus a built-in
  `scale_factor` store op. Our equivalent is hand-written kernel variants
  (`rfft_row_kernel_div`, `transpose_kernel_cmul`,
  `transpose_kernel_cmul_broadcast`, ...), and §3 would add three more.
  **Worth considering a comptime load-op/store-op closure parameter on
  the row and transpose kernels** before writing those three by hand —
  same performance, far less duplicated kernel body. This is a
  maintainability argument, not a performance one, but §3 is about to
  make the duplication noticeably worse.

## 6. Measurement gaps worth closing

- **No same-hardware torch/JAX baseline.** `../results.md` compares Mojo
  on gfx1151 against torch/jax on NVIDIA parts; the installed
  `.venv` torch is a CUDA build. Installing a ROCm torch (or
  jax-rocm) and running `main.py` on this machine would turn §0's
  bandwidth-normalized argument from an inference into a direct
  measurement — and is the single most informative thing left to
  measure.
- **No same-hardware expert-FFT reference.** rocFFT 7.2.0 is installed
  (`/opt/rocm/lib/librocfft.so`), `hipcc` is present, and the source
  checkout includes `clients/bench`. Timing a batched
  `2048 x 2048` real-to-complex transform at batch 41 through rocFFT
  directly would give the hardware-achievable floor for the FFT stages
  specifically, and `ROCFFT_LAYER` plan printing would confirm on this
  exact problem that rocFFT picks RTRT (as §3 predicts from the source).
  That turns "our structure matches rocFFT's choice" from a code reading
  into a verified fact.
- **Hardware counters, not just timings.** `rocprofv3` works; §4's
  claims about the `_cmul_broadcast` cache behaviour and the transposes'
  memory-level parallelism are inferred from achieved-bandwidth
  arithmetic and could be confirmed directly with DRAM/L2/MALL counters
  (`rocprofv3-avail` lists what gfx1151 exposes).

## 7. Aside: v1 and v2 are mathematically the same computation

Not an optimization proposal — the two formulations are the benchmark's
subject, so this should not be "fixed" — but worth recording. v1
computes `denominator = sum_d(irfft2(PSF_fft * rfft2(data)))` and v2
computes `denominator = irfft2(sum_d(PSF_fft * rfft2(data)))`. Because
`irfft2` is linear, these are identical up to float rounding; v2 just
does the depth reduction while the data is still `(D,H,W2)` complex,
performing one inverse 2D transform instead of `D=41` of them. That is
the whole reason v2 is ~26% faster, and it is the same class of
optimization as §3 — moving work to where the buffer is smallest.

---

# Session: sec. 3(a) implemented — transposed canonical spectrum + fused forward-FFT / complex-multiply / inverse-FFT column kernel

Brief: implement the highest-value item from the rocFFT-guided session's
sec. 3 — restructuring (a). Hardware unchanged: AMD gfx1151 (Radeon 8060S /
Ryzen AI MAX+ PRO 395, 40 CU, wave32, LPDDR5X unified memory, ~232 GB/s
achievable).

## Result

| | before | after | speedup |
|---|---|---|---|
| v1 | 0.1213 s | **0.0948 s** | **1.28x** |
| v2 | 0.0906 s | **0.0812 s** | **1.12x** |

Both baselines were re-measured on this machine rather than taken from the
log (they agree with the recorded 0.1197 / 0.0900 to within the run-to-run
spread). 20 iterations per run; the "after" column is the median of four v1
runs (0.0937, 0.0948, 0.0964, 0.0974) and three v2 runs (0.0802, 0.0818,
0.0820). This APU drifts a few percent between runs — enough to matter when
reading a single number, not enough to touch a 20-28% change. v1's
per-dispatch model in sec. 3(a) predicted 0.0936 s, the low end of what was
measured.

Correctness is unchanged and bit-identical to every previous revision:
`verify_correctness.py` still reports `max abs diff = 6.676e-06`,
`mean 1.108e-06`, `max rel 5.269e-07` for both v1 and v2 against the full
numpy reference at (41, 2048, 2048) — the same numbers recorded earlier in
this log. `test_openflr_gpu` reports `8.940697e-08` for both, and the whole
CPU/GPU FFT suite passes. This is expected: nothing about the arithmetic or
its order changed, only which buffer layout intermediates live in and which
kernel performs each load/store.

## What was built

**The canonical frequency-domain layout for the forward chain is now the
transposed `(D, W2, H)` one.** Both `rfft2 -> elementwise -> irfft2` chains
used to run `transpose (D,H,W2)->(D,W2,H)` immediately followed by its own
inverse, with only an elementwise multiply — which commutes with a
transpose — in between. Keeping the spectrum transposed deletes both.

**`fft_row_cmul_ifft_kernel`** (`fft_gpu.mojo`) then collapses what is left
of v1's first chain. Forward height-axis FFT, complex multiply by
`psf_fft_T`, and inverse height-axis FFT all happen in one shared-memory
residency, so the intermediate spectrum never reaches global memory. The
body is a `comptime for` over the two passes; the passes share the entire
stage schedule (warp-shuffle head + fused radix-4 tail, unchanged from
`fft_row_kernel`) and differ only at the two ends:

- the forward pass opens with the usual bit-reversed gather from global,
  and closes by multiplying the finished spectrum by `p` and leaving it in
  shared memory;
- the inverse pass opens with the same bit-reversed gather out of *shared*
  memory, and closes with the usual `1/N`-scaled global store.

New host-side drivers, all in `fft_gpu.mojo`: `rfft_w_transposed_gpu`,
`irfft_w_from_transposed_gpu`, `fft_col_cmul_ifft_gpu`,
`rfft2_batched_gpu_t`, `irfft2_batched_gpu_t`. `psf_fft` is precomputed
transposed by `prepare_psf` (free — PSF prep is outside the timed loop);
`psft_fft` is untouched, since the back-projection chain still runs in the
`(D, H, W2)` layout.

v2 got the transposed canonical layout too, though not the fusion: its
frequency-domain step is a multiply plus a depth reduction, not an inverse
column FFT, so there is nothing to fuse the forward FFT into. What v2 gains
is the same pair of cancelled transposes — the final transpose of `rfft2`
(9.4 ms) and the leading transpose of the small `(1, H, W)` `irfft2`.
`complex_mul_kernel` and `sum_over_depth_kernel` needed no changes at all:
both are flat over the buffer and are blind to which of the last two axes
is fastest.

### The one non-obvious detail: a shared-memory swizzle for the hand-off

The inverse pass's opening gather is bit-reversed, and reading that pattern
straight out of shared memory is a **32-way bank conflict**. For `N = 2048`
(`log2n = 11`), warp `w` lane `l` wants the element at
`bit_reverse(64w + l) == 64*rev5(l) + rev5(w)`, so every lane in the warp
lands on bank `rev5(w)` — the same one. Four such loads (`a_re, a_im, b_re,
b_im`) per thread, serialized 32 ways, is roughly 1.7 ms of pure LDS stall
per iteration by a rough cycle count — an eighth of the win, thrown away at
the seam.

`lds_swizzle(i) = i ^ ((i >> 6) & 31)` fixes it. It only moves bits 0-4, so
an element never leaves its own 32-element aligned block (the permutation
is a bijection, and an involution); the gather's bank becomes
`rev5(w) ^ rev5(l)`, distinct for all 32 lanes. The multiply at the end of
the forward pass writes in swizzled order, the inverse pass's gather reads
in swizzled order, and everything in between is untouched natural-order
indexing. For `N < 64` it is the identity, so the small-`N` (non-warp) path
gets it for free and the `D=3, H=8, W=8` test still exercises that branch.

Both the multiply and the inverse pass's gather are permutations of shared
memory performed in place, so each gathers into registers, hits a
`barrier()`, and only then stores — two extra barriers per row, against
four global round trips removed.

## Measured per-kernel breakdown after the change (rocprofv3, one
steady-state iteration)

### v1 — 94.6 ms total, 15 dispatches (was 121.6 ms, 19 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6271 | 1377 | 220 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8422 | 1377 | 164 |
| 3 | **`fft_row_cmul_ifft_kernel`** | **10855** | **2066** | **190** |
| 4 | `transpose_kernel` (W2,H)->(H,W2) | 9407 | 1377 | 146 |
| 5 | `irfft_row_kernel` (W) | 6424 | 1377 | 214 |
| 6 | `sum_over_depth_real_kernel` | 3336 | 705 | 211 |
| 7 | `rfft2_batched_gpu_div` (4 kernels) | 516 | 151 | — |
| 8 | `transpose_kernel_cmul_broadcast` | 17105 | 1394* | 81* |
| 9 | `fft_row_kernel` (H, inv) | 6526 | 1377 | 211 |
| 10 | `transpose_kernel` | 9287 | 1377 | 148 |
| 11 | `irfft_row_kernel` (W) | 6242 | 1377 | 221 |
| 12 | `shift_mul_kernel` | 10180 | 2064 | 203 |
| | **total** | **94571** | **16018** | **169** |

Dispatches 3-6 of the old table (`fft_row(H,fwd)` 7079 + `transpose` 9598 +
`transpose_cmul` 14865 + `fft_row(H,inv)` 6832 = 38.4 ms) are now the
single 10.9 ms kernel on row 3. The model predicted 10.3 ms at
195-202 GB/s; the kernel achieves 190 GB/s, a hair under the plain column
FFTs, which is the cost of the two extra barriers and the swizzle.

### v2 — 80.7 ms total, 17 dispatches (was 90.1 ms, 19 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6300 | 1377 | 219 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8712 | 1377 | 158 |
| 3 | `fft_row_kernel` (H, fwd) | 6459 | 1377 | 213 |
| 4 | `complex_mul_kernel` | 9686 | 2066 | 213 |
| 5 | `sum_over_depth_kernel` | 3219 | 705 | 219 |
| 6 | `irfft2_batched_gpu_t` (3 kernels) | 419 | 118 | — |
| 7 | `rfft2_batched_gpu_div` (4 kernels) | 596 | 151 | — |
| 8 | `transpose_kernel_cmul_broadcast` | 14046 | 1394* | 99* |
| 9 | `fft_row_kernel` (H, inv) | 6618 | 1377 | 208 |
| 10 | `transpose_kernel` | 9350 | 1377 | 147 |
| 11 | `irfft_row_kernel` (W) | 6215 | 1377 | 222 |
| 12 | `elementwise_mul_kernel` | 9057 | 2064 | 228 |
| | **total** | **80677** | **14760** | **183** |

\* Same caveat as the original table: the `_cmul_broadcast` MB figure counts
the broadcast operand once.

Traffic: 20.15 -> 16.02 GB (v1, -20%) and 16.12 -> 14.76 GB (v2, -8%).
Against sec. 0's bandwidth-normalized table, the pipeline now runs at 67%
(v1) and 72% (v2) of this GPU's *theoretical* 256 GB/s — 74% and 79% of the
measured 232 GB/s achievable — putting both at or above the top of the
51-76% band every torch/JAX number on every NVIDIA part occupies.

## What this changes about the remaining plan in sec. 3

- **(b) — fuse the broadcast complex multiply into the inverse column FFT's
  load — is now unambiguously the top remaining item, and it got *bigger*.**
  `transpose_kernel_cmul_broadcast` is still the worst kernel in the
  pipeline by a wide margin (81 GB/s in v1, 99 GB/s in v2) and now accounts
  for **18% of v1** and **17% of v2** all by itself. Dispatches 8+9 are
  23.6 ms (v1) and 20.7 ms (v2); one fused kernel moving ~1394 MB at
  ~200 GB/s is ~7.0 ms, so **~16.6 ms (18% of v1) and ~13.7 ms (15% of
  v2)**. The infrastructure this session built makes it much cheaper than
  the original estimate: the transposed-layout prerequisite is done, and
  `fft_row_cmul_ifft_kernel` is a working template for "gather a row, do
  elementwise work against a second operand, transform, store" — the
  broadcast variant is the same kernel with the multiply moved to the
  *load* and only the inverse pass kept. Note that v1's dispatch 8 got
  ~5% *slower* than the pre-change trace (17.1 ms vs 16.3 ms) even though
  nothing about it changed, which is more evidence for sec. 4(d)'s
  cache-eviction diagnosis: the kernels around it now stream different
  buffers, so the broadcast operand's residency shifted. **Get the grid
  ordering right (`d` fastest) in the replacement, per sec. 3(b).**
- **(c) — fuse the final elementwise multiply into `irfft_row_kernel`'s
  store — is unchanged and still worth ~7 ms (v1) / ~6 ms (v2).** It is
  independent of everything above.
- **(d) — fuse the complex multiply into `sum_over_depth_kernel` (v2 only)
  — is unchanged and still worth ~6.5 ms.** Dispatches 4+5 are 12.9 ms;
  one kernel moving ~1394 MB at ~220 GB/s is ~6.3 ms.
- **The remaining plain transposes are now a larger share of what is left**:
  17.8 ms of v1 (19%) and 18.1 ms of v2 (22%), still at 146-164 GB/s
  against a 232 GB/s ceiling. After (b)/(c)/(d), sec. 4(a) — `TILE=64`,
  block `(64,16)`, 4 elements per thread — becomes the top item, worth up
  to ~1.4x on those two kernels, i.e. ~5 ms each version.
- **sec. 5 is unaffected**: the row kernels are still 208-222 GB/s, and the
  new fused kernel at 190 GB/s is the closest thing in the pipeline to a
  compute-limited kernel. If anything, the extra barrier pressure inside it
  makes `half_lds` and the barrier-count ideas slightly more attractive
  than before — but still bounded by a few percent.

Projected if (b), (c) and (d) all land on top of this:
v1 ~0.070 s, v2 ~0.055 s — the same endpoint sec. 3 projected, now with
the largest and riskiest piece already banked.

**Same standing caveat as every finding in this log:** measured on a
bandwidth-constrained iGPU. The *structural* part of this change (fewer
kernels, fewer bytes, no redundant transposes) is architecture-generic and
should help anywhere. The *fusion* part trades global traffic for shared
memory pressure and barriers, which is a strictly better trade on this GPU
and very likely on H100 too, but the margin there is unmeasured.

# Session: sec. 3(b) implemented — broadcast complex multiply fused into the inverse column FFT's load

Brief: implement the top remaining item from the rocFFT-guided session's
sec. 3 — restructuring (b), which the previous session's closing notes flagged
as "unambiguously the top remaining item, and it got *bigger*". Hardware
unchanged: AMD gfx1151 (Radeon 8060S / Ryzen AI MAX+ PRO 395, 40 CU, wave32,
LPDDR5X unified memory, ~232 GB/s achievable).

## Result

| | before | after | speedup |
|---|---|---|---|
| v1 | 0.0950 s | **0.0798 s** | **1.19x** |
| v2 | 0.0817 s | **0.0686 s** | **1.19x** |

Both baselines were re-measured on this machine before touching anything and
agree with the previous session's recorded 0.0948 / 0.0812. 20 iterations per
run; both columns are medians of five runs (v1 after: 0.0795, 0.0796, 0.0798,
0.0800, 0.0803; v2 after: 0.0685, 0.0685, 0.0686, 0.0687, 0.0688). The spread
is well inside the few-percent drift this APU shows between runs.

Cumulative over the last two sessions: **v1 0.1213 -> 0.0798 (1.52x), v2
0.0906 -> 0.0686 (1.32x)**.

## What was built

**`ifft_row_cmul_broadcast_kernel`** (`fft_gpu.mojo`) replaces the
`transpose_kernel_cmul_broadcast -> fft_row_kernel(H, inv)` pair outright. The
transpose existed only to get the two operands from the `(D, H, W/2+1)` layout
into the `(D, W/2+1, H)` one the column FFT needs; with both operands already
transposed there is nothing left for it to do, and the broadcast complex
multiply it was carrying costs nothing folded into the inverse FFT's opening
bit-reversed gather — a load the kernel was performing anyway. Two dispatches
moving ~2066 MB become one moving ~1394 MB.

Getting the operands transposed took two supporting changes, both free:

- `psft_fft` is now precomputed by `rfft2_batched_gpu_t` instead of
  `rfft2_batched_gpu`, so the back-projection PSF spectrum joins `psf_fft` in
  the transposed canonical layout. PSF prep is outside the timed loop, so this
  is a pure drop of one transpose from setup.
- `rfft2_batched_gpu_div_t` (with `rfft_w_div_transposed_gpu` underneath) is
  `rfft2_batched_gpu_div` stopping one transpose early, exactly as
  `rfft2_batched_gpu_t` relates to `rfft2_batched_gpu`. That deletes one
  `(1, H, W/2+1)` transpose per iteration from the small four-kernel group.

With this, **every frequency-domain buffer in the pipeline is now in the
transposed `(D, W/2+1, H)` layout**; sec. 3(a)'s note that "only the forward
chain has been moved to the transposed layout" no longer applies, and the
`_t`-suffixed drivers are the only ones the pipeline uses.

New host driver `irfft2_batched_gpu_cmul_broadcast_t`: three dispatches
(fused inverse column FFT, transpose, `irfft_row_kernel`) where
`irfft2_batched_gpu_cmul_broadcast` had four.

### The grid ordering is worth as much as the fusion's last third

Sec. 3(b) predicted that mapping `row = block_idx.x` naively over the
`(D, W2, H)` layout — `w` fastest — would schedule the `D` blocks sharing a
broadcast row `W2` apart and sweep the 16.8 MB operand `D` times. That
prediction is now measured, by building the kernel both ways:

| block-index mapping | v1 | v2 |
|---|---|---|
| `d` fastest (`d = block_idx.x % D`) | **0.0798 s** | **0.0686 s** |
| `w` fastest (`w = block_idx.x % W2`) | 0.0828 s | 0.0718 s |

The `w`-fastest ordering costs 2.9 ms (v1) and 3.2 ms (v2) end-to-end. A full
re-sweep of the broadcast operand is `41 x 16.8 = 689` MB, which at the
232 GB/s ceiling is 3.0 ms — the model and the measurement agree closely
enough to consider sec. 4(d)'s cache-eviction diagnosis confirmed. **Two lines
of index arithmetic are worth a third of this restructuring's total win.**

### Cleanup: the FFT stage schedule is now written once

Adding a third kernel that needed the full radix-2 stage schedule would have
made three near-identical 250-line copies of it. Instead the schedule is now
two helpers, used by all three:

- **`fft_warp_head[invert]`** — the six register-resident stages
  (`stage_half = 1..32`, five `shuffle_xor` stages plus the purely local
  stage-32 combine). Independent of `N`: it only ever touches one warp's own
  64 elements.
- **`fft_lds_stages[N, first_stage, invert]`** — the fused radix-4 pairs and
  the leftover odd stage, over a row already in shared memory. `first_stage`
  is 6 after the warp head, 0 on the `N < 64` fallback path.

`fft_gpu.mojo` went from 2272 to 2275 lines while gaining a kernel and three
drivers. The refactor was verified to produce **bit-identical** full-scale
output, so it is a pure restructuring; benchmarks before and after it are the
same to within run-to-run noise (no inlining regression).

## Correctness

**Bit-identical to the previous revision.** The decisive check is Mojo against
Mojo: dumping the full `(41, 2048, 2048)` single-step output before and after
the change and comparing the raw `uint32` bit patterns gives an exact match
for both v1 and v2. Since sec. 3(b) only moves which kernel performs each
load/store — the same products in the same order, then the same stage
schedule — that is the expected result, and it is a stronger statement than
any tolerance against a reference. `test_openflr_gpu` reports `8.940697e-08`
for both versions and the whole CPU/GPU FFT suite passes, all unchanged.

**Caveat on `verify_correctness.py`: it could not be run this session.** The
repository's `.venv` is mid-migration from a CUDA to a ROCm torch/jax stack
(uncommitted edits to `pyproject.toml`/`.python-version`, and
`.venv/lib/python3.14/site-packages` is empty), so `main.py`'s module-level
`import jax` / `import torch` cannot resolve. A numpy-only standalone
equivalent — same reference arithmetic, reading the already-prepared
`data/*.bin` instead of importing `main.py` — reports `max abs diff 8.18e-06`
against numpy 1.26.4 and `8.58e-06` against numpy 2.5.2, versus the
`6.676e-06` recorded earlier in this log. **That spread is the numpy version,
not the Mojo code**: two numpy builds disagree with each other by more than
either disagrees with the record, while the Mojo output is bit-for-bit
unchanged. Re-run `verify_correctness.py` once the venv is rebuilt to restore
the like-for-like number.

## Measured per-kernel breakdown after the change (rocprofv3, mean of two steady-state iterations)

### v1 — 81.7 ms total, 13 dispatches (was 94.6 ms, 15 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6410 | 1377 | 215 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8860 | 1377 | 155 |
| 3 | `fft_row_cmul_ifft_kernel` | 11491 | 2066 | 180 |
| 4 | `transpose_kernel` (W2,H)->(H,W2) | 9413 | 1377 | 146 |
| 5 | `irfft_row_kernel` (W) | 6453 | 1377 | 213 |
| 6 | `sum_over_depth_real_kernel` | 3385 | 705 | 208 |
| 7-9 | `rfft2_batched_gpu_div_t` (3 kernels, (1,H,W)) | 465 | 118 | — |
| 10 | **`ifft_row_cmul_broadcast_kernel`** | **8412** | **1394** | **166** |
| 11 | `transpose_kernel` | 9482 | 1377 | 145 |
| 12 | `irfft_row_kernel` (W) | 6710 | 1377 | 205 |
| 13 | `shift_mul_kernel` | 10596 | 2064 | 195 |
| | **total** | **81678** | **14609** | **179** |

Dispatches 8+9 of the old table (`transpose_kernel_cmul_broadcast` 17105 +
`fft_row_kernel(H,inv)` 6526 = 23.6 ms) are now the single 8.4 ms kernel on
row 10, and the old four-kernel `rfft2_batched_gpu_div` group is down to three.

### v2 — 69.0 ms total, 15 dispatches (was 80.7 ms, 17 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6324 | 1377 | 218 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8651 | 1377 | 159 |
| 3 | `fft_row_kernel` (H, fwd) | 7186 | 1377 | 192 |
| 4 | `complex_mul_kernel` | 9769 | 2066 | 211 |
| 5 | `sum_over_depth_kernel` | 3248 | 705 | 217 |
| 6-8 | `irfft2_batched_gpu_t` (3 kernels) | 466 | 118 | — |
| 9-11 | `rfft2_batched_gpu_div_t` (3 kernels) | 664 | 118 | — |
| 12 | **`ifft_row_cmul_broadcast_kernel`** | **7839** | **1394** | **178** |
| 13 | `transpose_kernel` | 9241 | 1377 | 149 |
| 14 | `irfft_row_kernel` (W) | 6464 | 1377 | 213 |
| 15 | `elementwise_mul_kernel` | 9133 | 2064 | 226 |
| | **total** | **68985** | **13350** | **194** |

Traffic: 16.02 -> 14.61 GB (v1) and 14.76 -> 13.35 GB (v2). Note both "before"
figures were themselves optimistic — they counted the broadcast operand once
in a kernel that was demonstrably re-sweeping it (the 81/99 GB/s anomaly), so
the real byte reduction is larger than the -9% the model shows. Against
sec. 0's normalization the pipeline now runs at 71% (v1) and 76% (v2) of this
GPU's *theoretical* 256 GB/s, i.e. 79% and 84% of the measured 232 GB/s
achievable — both now clear of the 51-76% band every torch/JAX number on
every NVIDIA part occupies.

## What this changes about the remaining plan in sec. 3

- **The plain transposes are now the largest single category, and in v1 they
  are the obvious next target.** v1 has three full-size transposes left
  (dispatches 2, 4, 11) totalling **27.8 ms — 34% of v1's runtime** — at
  145-155 GB/s; v2 has two totalling 17.9 ms (26%) at 149-159 GB/s. Sec. 4(a)
  (`TILE=64`, block `(64,16)`, 4 elements per thread with unrolled read /
  LDS-transpose / write loops, as rocFFT does it) is worth up to ~1.4x on
  these, i.e. **~8 ms on v1 and ~5 ms on v2**. This is now a bigger item than
  (c) for v1 and comparable for v2, and it is the last idea in this log with a
  large payoff that does not require restructuring the pipeline.
- **(c) — fuse the final elementwise multiply into `irfft_row_kernel`'s store
  — is unchanged and now slightly larger.** v1 dispatches 12+13 are 17.3 ms;
  one kernel moving 2064 MB at ~220 GB/s is ~9.4 ms, **saving ~7.9 ms (10% of
  v1)**. v2 dispatches 14+15 are 15.6 ms -> ~9.4 ms, **saving ~6.2 ms (9% of
  v2)**. Still independent of everything else, still the cheapest thing on the
  list to write.
- **(d) — fuse the complex multiply into `sum_over_depth_kernel` (v2 only) —
  is unchanged and worth ~6.7 ms.** Dispatches 4+5 are 13.0 ms; one kernel
  moving ~1394 MB at ~220 GB/s is ~6.3 ms. Note the caution below about what
  rate to expect.
- **`ifft_row_cmul_broadcast_kernel` itself came in at 166 GB/s (v1) /
  178 GB/s (v2), under the ~200 GB/s the model assumed** (8.4 ms rather than
  the projected 7.0 ms). The kernel it replaced ran at 81/99 GB/s, so this is
  still a large win, but the shortfall is real and has a plausible cause: the
  `d`-fastest ordering that keeps the broadcast operand cache-resident
  deliberately makes consecutive blocks read and write the main `(D, W2, H)`
  stream 8.4 MB apart, trading DRAM locality on the big stream for cache
  locality on the small one. The measurement above says that trade is worth
  ~3 ms net, so it is the right call — but **a middle ordering that keeps `d`
  fast in groups while walking `w` locally (e.g. swizzling within tiles of a
  few `w` values) might recover part of the remaining ~1.4 ms** and is cheap
  to try. It also means the ~220 GB/s assumed for (c) and (d) should be
  treated as optimistic where the fused kernel's access pattern is not purely
  streaming.
- **Sec. 5 and the "stop optimizing the butterfly math" conclusion are
  unaffected**: the row kernels are still 192-218 GB/s.

Projected if (c), (d) and sec. 4(a) all land on top of this: **v1 ~0.064 s,
v2 ~0.051 s** — past the 0.070 / 0.055 endpoint sec. 3 originally projected,
with the two largest structural pieces now banked.

**Same standing caveat as every finding in this log:** measured on a
bandwidth-constrained iGPU. The structural part (two fewer dispatches per
iteration, ~1.4 GB less traffic, no redundant transposes anywhere in the
frequency domain) is architecture-generic. The grid-ordering result is the
part most specific to this hardware — it is a statement about this GPU's cache
capacity relative to a 16.8 MB operand, and the right ordering on a part with
a 50 MB L2 could differ.

# Session: sec. 4(a) tried and rejected; sec. 3(c) implemented — final elementwise multiply fused into the width inverse FFT's store

Brief: the previous session's closing notes named sec. 4(a) (rocFFT's
`TILE=64` / block `(64,16)` / 4-elements-per-thread transpose) the top
remaining item — "the last idea in this log with a large payoff that does not
require restructuring the pipeline" — worth ~8 ms on v1 and ~5 ms on v2. It
was implemented, measured across the whole tile/block space, and **rejected:
it is a loss on this GPU at every configuration.** Sec. 3(c) was implemented
instead and is now the largest banked win left. Hardware unchanged: AMD
gfx1151 (Radeon 8060S / Ryzen AI MAX+ PRO 395, 40 CU, wave32, LPDDR5X unified
memory, ~232 GB/s achievable).

## Result

| | before | after | speedup |
|---|---|---|---|
| v1 | 0.0793 s | **0.0722 s** | **1.10x** |
| v2 | 0.0683 s | **0.0628 s** | **1.09x** |

Both baselines were re-measured on this machine before touching anything
(0.07927 / 0.06825) and agree with the previous session's recorded 0.0798 /
0.0686. 20 iterations per measurement. Correctness is unchanged and exact:
`test_openflr_gpu` reports the same `max|diff| = 8.940697e-08` as before, and
`verify_correctness.py` at full scale (41, 2048, 2048) reports
`max abs diff = 8.583e-06, max rel diff = 6.760e-07` against the numpy
reference for both versions — identical to the pre-change values, as expected
for a change that only moves where a load and a store happen.

Cumulative from the start of the rocFFT-guided investigation: v1 0.1197 ->
0.0722 (**1.66x**), v2 0.0900 -> 0.0628 (**1.43x**).

## Part 1: sec. 4(a) is wrong for this GPU — a negative result worth recording

The claim in sec. 4(a) was that our transposes sit at 145-162 GB/s (vs
192-226 for every other kernel) because each thread moves only 8 bytes per
phase and therefore "spends its time on index arithmetic and one round of
load latency it can't hide", and that rocFFT's 4-elements-per-thread shape
would fix it.

**Implemented in full**, as `TROWS` elements per thread with block
`(TILE, TILE/TROWS)`, `comptime for`-unrolled read and write loops, and a
fast path for interior tiles so the unrolled global loads issue back to back
with no branch between them. Two details differ from sec. 4(a)'s sketch and
both are improvements on it:

- **An XOR swizzle (`tile[r, c ^ r]`) instead of `TILE+1` column padding.**
  Both fix the same bank conflict, but at `TILE=64` the padded tile costs
  `2 * 64 * 65 * 4 = 33.3 KB` of LDS — over half this GPU's 64 KB per-CU
  budget, so only one 1024-thread block would be resident where two fit
  today. Unpadded, `re` and `im` together are exactly 32 KB and occupancy is
  unchanged. (Sec. 4(a) flagged the LDS problem and left it open; this is the
  answer.) The swizzle is conflict-free in both phases — `c ^ r` is a
  bijection over a row, so write lanes (consecutive `tx`, one `r`) hit
  distinct banks, and read lanes (row `tx`, column `c ^ tx`) have bank index
  `(tx * TILE + (c ^ tx)) % 32 = (c ^ tx) % 32`, also distinct.
- Applied to all three transpose variants, so the tile shape is one constant.

Measured, v1, 15 iterations per point:

| | TROWS=1 | TROWS=2 | TROWS=4 | TROWS=8 | TROWS=16 |
|---|---|---|---|---|---|
| **TILE=32** | 0.0796 | 0.0793 | 0.0794 | 0.0800 | — |
| **TILE=64** | — | n/a | 0.0863 | 0.0842 | 0.0833 |

(The two "n/a"s are configurations whose block would exceed the 1024-thread
limit: `TILE=64, TROWS=2` is `(64,32)` and every `TILE=128` point is at least
`(128,16)`. Baseline for reference: 0.0793.)

Two things fall out, and they are more informative than the win would have
been:

1. **Elements-per-thread does nothing.** At `TILE=32`, going from 1 to 2 to 4
   elements per thread — i.e. from 256 to 1024 threads' worth of work per
   thread and 1 to 4 independent loads in flight — moves the total by less
   than the 0.0006-0.0008 s run-to-run std. **The transposes are not
   latency-limited, so the memory-level-parallelism diagnosis in sec. 4(a) is
   simply wrong for this GPU.** Every conclusion in this log that rests on
   "more work per thread is why the row kernels are faster than the
   transposes" should be treated as retracted.
2. **A bigger tile is actively worse, by 5-9%,** monotonically improving as
   the block gets *smaller* within `TILE=64` (16 elems/thread, 256 threads,
   beats 4 elems/thread, 1024 threads) but never recovering `TILE=32`. The
   destination footprint is the difference: a `TILE=64` block writes 64 rows
   of 256 B spread over 64 * 8 KB = 512 KB of destination, twice `TILE=32`'s.
   The transposes are limited by *how much address space a block touches at
   once* — open DRAM pages / channel spread — not by per-thread work.

That reframes what is left. **The remaining transposes' 145-155 GB/s is a
partition-camping residue, not a latency problem**, and the specific residue
the diagonal remap does not address is *intra-block*: the remap decorrelates
consecutively-scheduled blocks from each other, but the 32 destination rows
*within* one block are still exactly 8192 B apart (`H * 4`, a power of two),
so they can all land on the same channel. That points at sec. 4(c)
(**row-pitch padding**, which rocFFT applies to strides and which this log has
never actually tried — the earlier reverted experiment padded a *length* on
the non-power-of-two axis, a different change) as the right next idea for the
transposes, and demotes sec. 4(b) (AoS interleaving), whose stated benefit was
instruction count.

The change was reverted in full; `transpose_kernel`,
`transpose_kernel_cmul` and `transpose_kernel_cmul_broadcast` are byte-identical
to their previous form and `TILE` is back to 32.

## Part 2: sec. 3(c) implemented

`shift_mul_kernel` (v1) and `elementwise_mul_kernel` (v2) each read back a
full `(D, H, W)` buffer that the immediately preceding `irfft_row_kernel` had
just written, only to multiply it by `data_buf`. But `irfft_row_kernel`
already holds the finished row in shared memory at the moment it stores it, so
that whole round trip existed for nothing.

`irfft_row_mul_kernel` does the store instead: it reads the corresponding row
of `data_buf`, multiplies, and writes the product. The `(D, H, W)`
backprojection buffer is never written and never read — 1377 MB of traffic per
iteration removed, one dispatch removed, and `OpenFlrScratch.back` (688 MB of
device memory, which matters on a unified-memory part) deleted outright.

**v1's fftshift comes along free.** fftshift over the last two axes is a pure
index rotation and is its own inverse for even `H`/`W`, so rotating the
*destination* index is equivalent and keeps everything coalesced: the row
rotation picks a different destination row (`(rr + H/2) % H`) and the column
rotation turns each block's contiguous run of columns into two contiguous
half-row segments. Both the `data_buf` read and the `out_buf` store stay fully
coalesced, and the `SHIFT` flag is a comptime parameter, so v2 pays nothing
for code it does not use.

### What was built

- **`irfft_row_into_lds`** (`fft_gpu.mojo`) — the body of the row-wise real
  inverse FFT, extracted verbatim from `irfft_row_kernel`: reads a row of the
  `(num_rows, N/2+1)` half spectrum and leaves the unscaled length-`N` real
  result in shared memory. Everything after it is a store, and the store is
  the only thing the callers disagree about. This keeps the radix-4 stage
  schedule and its warp-shuffle head in exactly one place — the same
  factoring `fft_warp_head` / `fft_lds_stages` already established for the
  forward direction.
- **`irfft_row_kernel`** — now `irfft_row_into_lds` plus the original scaled
  store. Behaviour unchanged; `test_fft_gpu` and `test_fft_gpu_radix4` pass
  with identical error figures.
- **`irfft_row_mul_kernel[N, SHIFT, H, ...]`** — `irfft_row_into_lds` plus the
  fused store: `out = mul * irfft(in)`, or `out = mul * fftshift(irfft(in))`
  when `SHIFT`.
- **`irfft2_batched_gpu_cmul_broadcast_t` -> `..._cmul_broadcast_mul_t`**,
  gaining a `SHIFT` parameter and a `mul_buf` argument. Both of the function's
  call sites — the tails of `run_v1_step_gpu` and `run_v2_step_gpu` — wanted
  the trailing multiply, so this is a change to the existing function rather
  than a variant beside it. It now computes the entire tail of a
  Richardson-Lucy update, `data * [fftshift](irfft2(err_fft * psft_fft))`, in
  three dispatches where the pre-3(b) code took five.
- **Deleted**: `shift_mul_kernel`, `elementwise_mul_kernel`,
  `OpenFlrScratch.back`, and the now-unused `thread_idx` / `block_idx` imports
  in `openflr_gpu.mojo`.

## Measured per-kernel breakdown after the change (rocprofv3, mean of two steady-state iterations)

### v1 — 74.2 ms total, 12 dispatches (was 81.7 ms, 13 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6515 | 1377 | 211 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 9077 | 1377 | 152 |
| 3 | `fft_row_cmul_ifft_kernel` | 11708 | 2066 | 176 |
| 4 | `transpose_kernel` (W2,H)->(H,W2) | 9428 | 1377 | 146 |
| 5 | `irfft_row_kernel` (W) | 6448 | 1377 | 214 |
| 6 | `sum_over_depth_real_kernel` | 3352 | 705 | 210 |
| 7-9 | `rfft2_batched_gpu_div_t` (3 kernels, (1,H,W)) | 438 | 118 | — |
| 10 | `ifft_row_cmul_broadcast_kernel` | 8625 | 1394 | 162 |
| 11 | `transpose_kernel` | 9437 | 1377 | 146 |
| 12 | **`irfft_row_mul_kernel`** | **9200** | **2064** | **224** |
| | **total** | **74228** | **13232** | **178** |

### v2 — 63.1 ms total, 14 dispatches (was 69.0 ms, 15 dispatches)

| # | kernel | µs | MB | GB/s |
|---|---|---|---|---|
| 1 | `rfft_row_kernel` (W) | 6407 | 1377 | 215 |
| 2 | `transpose_kernel` (H,W2)->(W2,H) | 8993 | 1377 | 153 |
| 3 | `fft_row_kernel` (H, fwd) | 7019 | 1377 | 196 |
| 4 | `complex_mul_kernel` | 9707 | 2066 | 213 |
| 5 | `sum_over_depth_kernel` | 3211 | 705 | 220 |
| 6-8 | `irfft2_batched_gpu_t` (3 kernels) | 439 | 118 | — |
| 9-11 | `rfft2_batched_gpu_div_t` (3 kernels) | 546 | 118 | — |
| 12 | `ifft_row_cmul_broadcast_kernel` | 8146 | 1394 | 171 |
| 13 | `transpose_kernel` | 9475 | 1377 | 145 |
| 14 | **`irfft_row_mul_kernel`** | **9179** | **2064** | **225** |
| | **total** | **63122** | **11973** | **190** |

Dispatches 12+13 of the old v1 table (`irfft_row_kernel` 6710 +
`shift_mul_kernel` 10596 = 17.3 ms, 3441 MB) are now the single 9.2 ms,
2064 MB kernel on row 12 — **saving 8.1 ms**, against the 7.9 ms sec. 3(c)
projected. v2's 14+15 (6464 + 9133 = 15.6 ms) become 9.2 ms, **saving
6.4 ms** against a projected 6.2 ms. The new kernel runs at **224-225 GB/s,
97% of this GPU's measured achievable bandwidth** — the fastest kernel in
either pipeline, and above the ~220 GB/s sec. 3(c) assumed despite the
previous session's warning that that figure might be optimistic. The fftshift
rotation costs nothing measurable: v1's and v2's copies of the kernel are
within 0.2% of each other.

Traffic: 14.61 -> 13.23 GB (v1) and 13.35 -> 11.97 GB (v2), -9.4% and -10.3%.
Note that the pipeline's *achieved* bandwidth is essentially unchanged
(183 -> 183 GB/s in v1, 195 -> 191 GB/s in v2): the pair of kernels this
replaced was already running at 199 GB/s, so **the entire win is bytes not
moved, not a faster kernel** — which is exactly what sec. 0 predicted would be
the only lever with a large payoff.

## What is left

- **The plain transposes are now 29-38% of the remaining runtime** — 27.9 ms
  of v1's 74.2 (three dispatches) and 18.5 ms of v2's 63.1 (two) — at
  145-153 GB/s. Part 1 rules out the micro-optimization this log had queued
  for them and re-points at **sec. 4(c) row-pitch padding** (pad the `(D, W2,
  H)` intermediate's row stride from `H = 2048` to `H + 64 = 2112` floats,
  which the layouts here can express by declaring the buffer `row_major[D, W2,
  H + 64]` and having every kernel touch only the first `H` columns) as the
  one untried idea aimed at the actual mechanism. Per rocFFT this is an
  **alternative to** the diagonal remap, not a complement — A/B it, do not
  stack it. Upside if it lands cleanly is the same ~1.4x on those kernels the
  transpose band suggests: ~8 ms (v1), ~5 ms (v2).
- **(d) — fuse the complex multiply into `sum_over_depth_kernel` (v2 only) —
  is unchanged and is now the largest remaining fusion.** Dispatches 4+5 are
  12.9 ms moving 2771 MB; one kernel moving ~1394 MB at the 224 GB/s row 14
  just demonstrated is ~6.2 ms, **saving ~6.7 ms (11% of v2)**. Row 14 is
  direct evidence that a fused kernel here can hit that rate, so this estimate
  is no longer optimistic. Still no atomics needed — see sec. 3(d).
- **`ifft_row_cmul_broadcast_kernel` remains the slowest large kernel**
  (162 GB/s v1, 171 GB/s v2) for the grid-ordering reason sec. 3(b) diagnosed.
  The "middle ordering" idea there (keep `d` fast in groups while walking `w`
  locally) is still cheap and untried, worth maybe ~1.5 ms.
- **`fft_row_cmul_ifft_kernel` at 176 GB/s** is the other kernel below the
  band, and unlike the transposes it genuinely does the most work per thread
  in the pipeline — it is the closest thing here to a compute-limited kernel,
  so sec. 5's barrier-count ideas apply to it if anything does.

Projected if (d) and sec. 4(c) both land: v1 ~0.064 s, v2 ~0.049 s.

**Same standing caveat as every finding in this log:** measured on a
bandwidth-constrained iGPU. Sec. 3(c) is architecture-generic — it removes a
full-size write and its matching read on any hardware, and the arithmetic is
bit-identical. Part 1's negative result is the opposite: it is a statement
about this GPU's DRAM organization and cache capacity, and rocFFT's tile shape
may well be right on the parts rocFFT is tuned for. Anyone porting this to
H100 should re-run the sweep rather than inherit the `TILE=32` conclusion.
