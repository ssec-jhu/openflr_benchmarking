#!/usr/bin/env bash
# Per-kernel attribution for one arm on one card, plus the sustained clocks
# that attribution has to be interpreted against.
#
#   ./profile_card.sh v2 t8w8c4G
#
# Writes <gpu>_<version>_<arm>.prof.txt next to itself. Run it on every card
# being compared, with the same version/arm, and the per-kernel A100->H100
# ratios fall straight out of the two files.
#
# Requires the warmup-follows-the-arm fix in main.mojo: with a hard-coded
# `base` warmup, a single-arm trace is ~97% `base` and merges the two
# wherever a kernel name is unchanged.
set -u

VER="${1:-v2}"
ARM="${2:-t8w8c4G}"
ITERS="${3:-3}"

cd "$(dirname "$0")" || exit 1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '_' | tr -d '(),')
OUT="${GPU}_${VER}_${ARM}.prof.txt"
TAG="prof_${VER}_${ARM}"

echo "=== card ===" > "$OUT"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit \
    --format=csv -i 0 >> "$OUT"

# Sustained clocks matter more than the boost spec: the whole point of the
# comparison is a ratio of SM clocks, and a power- or thermally-capped card
# does not run at its boost number. Sample through the run and report the
# distribution, not one instantaneous reading.
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv,noheader,nounits -i 0 -lms 200 > /tmp/clk.$$.csv 2>/dev/null &
CLKPID=$!

# Compile outside the profiler. `mojo run` builds the kernels in-process,
# which takes minutes, and neither nsys nor ncu should be tracing that --
# nsys would trace a multi-minute CPU-only prologue into the report, and ncu
# would attach through it. This also leaves the compile cache warm.
echo "" >> "$OUT"
echo "=== pre-compile (outside the profiler) ===" >> "$OUT"
pixi run mojo run src/main.mojo -- "$VER" 1 "$ARM" >> "$OUT" 2>&1 \
  || { echo "pre-compile FAILED -- see $OUT, not profiling"; exit 1; }

echo "" >> "$OUT"
echo "=== nsys profile: $VER $ARM ($ITERS timed iters after 2s warmup) ===" >> "$OUT"
nsys profile --trace=cuda --force-overwrite=true -o "$TAG" \
    pixi run mojo run src/main.mojo -- "$VER" "$ITERS" "$ARM" >> "$OUT" 2>&1

kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null

echo "" >> "$OUT"
echo "=== sustained clocks during the run (sm_mhz, mem_mhz, W, degC, util%) ===" >> "$OUT"
awk -F', *' '$5+0 > 50 { n++; s+=$1+0;
       if (mx==0 || ($1+0)>mx) mx=$1+0; if (mn==0 || ($1+0)<mn) mn=$1+0;
       if (($3+0)>pmax) pmax=$3+0; if (($4+0)>tmax) tmax=$4+0 }
     END { if (!n) { print "no busy samples captured"; exit }
           printf "samples=%d  sm_clock: min=%d max=%d mean=%.0f MHz\n", n, mn, mx, s/n
           printf "peak power=%.0f W  peak temp=%d C\n", pmax, tmax }' /tmp/clk.$$.csv >> "$OUT"
rm -f /tmp/clk.$$.csv

nsys export --type sqlite --force-overwrite true -o "${TAG}.sqlite" "${TAG}.nsys-rep" >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "=== attribution (see prof_report.py for what each table is for) ===" >> "$OUT"
# stdlib Python, not the sqlite3 CLI: that CLI is not installed on every
# cluster node and this needs to run wherever the GPU is.
python3 prof_report.py "${TAG}.sqlite" >> "$OUT" 2>&1

echo "wrote $OUT"
