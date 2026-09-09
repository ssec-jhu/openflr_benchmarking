#!/usr/bin/env bash
# Which resource actually binds the FFT kernels on this card.
#
#   ./ncu_card.sh v2 t8w8c4G
#
# Answers three questions the nsys pass cannot:
#   1. Is the byte model in optimizations.md true?  Every GB/s figure in that
#      file is *intended* traffic and none has ever been checked against
#      dram__bytes.  Group A checks it.
#   2. Are these kernels near the DRAM roof, or nowhere near it?
#   3. If nowhere near it, which per-SM pipe is saturated -- shared memory,
#      LSU/L1, or the SFU/XU that the per-thread cos/sin twiddles issue on?
#      Group B ranks the pipes against each other.
#
# Two invocations with small metric sets rather than one big one: kernel
# replay has to save and restore the 1.4-2 GB each of these kernels writes,
# once per pass, so pass count is the whole cost.  `--set full` is hopeless
# here for exactly that reason.
#
# ncu locks clocks (nsys does not), so absolute times here will not match
# the nsys pass.  Read the percentages, not the durations.
set -u

VER="${1:-v2}"
ARM="${2:-t8w8c4G}"
COUNT="${3:-20}"        # ~one full v2 iteration is 15 dispatches
# `prepare_psf` calls `rfft2_batched_gpu_t` three times before any arm runs,
# and that overload takes the *default* parameters (TDIV=2, WDIV=4, CG=0) --
# i.e. `base`, not the arm.  Each call is 3 kernels (rfft_row_kernel,
# transpose_kernel, fft_row_kernel), so the first 9 launches of the process
# must be skipped or half the report describes the wrong configuration.
SKIP="${4:-9}"

cd "$(dirname "$0")" || exit 1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '_' | tr -d '(),')
OUT="${GPU}_${VER}_${ARM}.ncu.txt"

# --target-processes all is mandatory: without it ncu attaches to `pixi` and
# waits forever for launches that never come.  --launch-count must stay in
# the low tens; these grids are 42k-84k blocks and sampling overhead scales
# with warp-cycles.
COMMON=(ncu --target-processes all -s "$SKIP" -c "$COUNT" --csv)
APP=(pixi run mojo run src/main.mojo -- "$VER" 1 "$ARM")

echo "=== card ===" > "$OUT"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv -i 0 >> "$OUT"
ncu --version 2>&1 | head -3 >> "$OUT"

# Compile outside the profiler. `mojo run` builds the kernels in-process,
# which takes minutes, and neither nsys nor ncu should be tracing that --
# nsys would trace a multi-minute CPU-only prologue into the report, and ncu
# would attach through it. This also leaves the compile cache warm.
echo "" >> "$OUT"
echo "=== pre-compile (outside the profiler) ===" >> "$OUT"
pixi run mojo run src/main.mojo -- "$VER" 1 "$ARM" >> "$OUT" 2>&1 \
  || { echo "pre-compile FAILED -- see $OUT, not profiling"; exit 1; }

echo "" >> "$OUT"
echo "=== group A: is the byte model right, and where is the DRAM roof ===" >> "$OUT"
echo "(skipping the first $SKIP launches: PSF prep, which runs base parameters)" >> "$OUT"
"${COMMON[@]}" --metrics \
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
lts__t_sectors.sum \
    "${APP[@]}" >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "=== group B: which per-SM pipe is saturated ===" >> "$OUT"
# sm__throughput is the ceiling of all the SM pipes; the four
# smsp__inst_executed_pipe_* lines say which one sets it.  `xu` is the
# SFU/transcendental pipe -- that is the twiddle cos/sin, and it is the one
# metric that would vindicate the retired `TW` twiddle-table arm on a card
# where DRAM got 1.9x faster and the SFU only 1.55x.
"${COMMON[@]}" --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    "${APP[@]}" >> "$OUT" 2>&1

echo "wrote $OUT"
