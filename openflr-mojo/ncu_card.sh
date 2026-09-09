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
# ncu locks clocks (nsys does not), so absolute times here will not match
# the nsys pass.  Read the percentages, not the durations.
#
# ---------------------------------------------------------------------------
# Why this profiles four kernels and not an iteration
#
# Kernel replay has to save and restore the 1.4-2 GB each of these kernels
# writes, once per pass per launch, so cost is launches x passes x 2 GB.
# optimizations.md's trap note is explicit that --launch-count has to stay
# in the low single digits and that ~33 launches x ~10 passes runs for over
# half an hour; the first version of this script asked for 20 and took
# 20-30 minutes, which is that warning being ignored.
#
# Four launches is enough, because the question is about two *populations*
# rather than about every dispatch.  The A100 profile split the pipeline
# cleanly: every kernel with an FFT butterfly in it sat at 22-37% of peak
# DRAM bandwidth, every kernel without one at 69-87%.  So take the two
# largest of the first group and the two best of the second:
#
#   ifft_row_cmul_broadcast_kernel  22.4% of the iteration, 22% of peak
#   irfft_row_mul_kernel            21.3% of the iteration, 35% of peak
#   complex_mul_kernel               8.5% of the iteration, 87% of peak
#   sum_over_depth_kernel            3.2% of the iteration, 79% of peak
#
# If the two FFT kernels are far from the DRAM roof while the two controls
# sit on it, and one SM pipe is pegged on the FFT pair and not on the
# controls, that pipe is the answer.  Adding the two forward FFT kernels
# would confirm rather than discriminate.
#
# The filter also removes the other trap in one stroke.  `prepare_psf` calls
# `rfft2_batched_gpu_t` three times before any arm runs, at that overload's
# *default* parameters (TDIV=2, WDIV=4, CG=0, i.e. `base`), so the process's
# first 9 launches are the wrong configuration -- but all three of them are
# `rfft_row_kernel`, `transpose_kernel` and `fft_row_kernel`, and none of
# the four kernels above is dispatched by the forward transform at all.  So
# no launch-skip is needed, and none of the `--launch-skip` semantics that
# vary between ncu versions is being relied on.
#
# To widen it later, pass a regex as the 4th argument.  Include
# `fft_row_kernel` or `rfft_row_kernel` there and the prep launches come
# back, so add `-s 9` by hand if you do.
# ---------------------------------------------------------------------------
set -u

VER="${1:-v2}"
ARM="${2:-t8w8c4G}"
KERNELS="${3:-regex:(ifft_row_cmul_broadcast|irfft_row_mul_kernel|complex_mul_kernel|sum_over_depth)}"
COUNT="${4:-4}"

cd "$(dirname "$0")" || exit 1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '_' | tr -d '(),')
OUT="${GPU}_${VER}_${ARM}.ncu.txt"

# --target-processes all is mandatory: without it ncu attaches to `pixi` and
# waits forever for launches that never come.
COMMON=(ncu --target-processes all -k "$KERNELS" -c "$COUNT" --csv)
APP=(pixi run mojo run src/main.mojo -- "$VER" 1 "$ARM")

echo "=== card ===" > "$OUT"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv -i 0 >> "$OUT"
ncu --version 2>&1 | head -3 >> "$OUT"
echo "kernel-filter=$KERNELS launch-count=$COUNT" >> "$OUT"

# Compile outside the profiler. `mojo run` builds the kernels in-process,
# which takes minutes, and ncu should not be attached through that. This
# also leaves the compile cache warm.
echo "" >> "$OUT"
echo "=== pre-compile (outside the profiler) ===" >> "$OUT"
pixi run mojo run src/main.mojo -- "$VER" 1 "$ARM" >> "$OUT" 2>&1 \
  || { echo "pre-compile FAILED -- see $OUT, not profiling"; exit 1; }

echo "" >> "$OUT"
echo "=== group A: is the byte model right, and where is the DRAM roof ===" >> "$OUT"
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
