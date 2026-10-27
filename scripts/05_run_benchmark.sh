#!/usr/bin/env bash
# =====================================================================
# 05_run_benchmark.sh -- Track C1: how long one network evaluation takes,
# measured WITHOUT RenderMan (src/benchmark/bench_eval.cpp).
#
# Builds bench_eval.exe with the exact command in bench_eval.cpp's header
# (this one DOES use /O2), runs it BENCH_RUNS times (default 5) on the
# alum-bronze weights with 10,000,000 evaluations each, and reports the
# median of each timing line. The paper reports the median of 5 runs.
#
# Usage:   bash scripts/05_run_benchmark.sh
# Takes:   about 1-2 minutes.
# Output:  results/reproduced/bench_eval_runs.txt and bench_eval_median.txt
# Tip:     close other programs first; timings depend on what else is running.
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
is_windows || die "This script builds with MSVC on Windows. (On Linux/macOS: g++ -std=c++17 -O2 src/benchmark/bench_eval.cpp -o bench_eval)"

mkdir -p "$WORK_DIR"
cp "${REPO_ROOT}/src/benchmark/bench_eval.cpp" "${WORK_DIR}/"
cp "${WEIGHTS_DIR}/nbrdf_weights.bin" "${WORK_DIR}/"
rm -f "${WORK_DIR}/bench_eval.obj" "${WORK_DIR}/bench_eval.exe"

say "Compiling bench_eval.cpp (release, /O2)"
run_with_msvc "cl /std:c++17 /O2 /EHsc /D_USE_MATH_DEFINES bench_eval.cpp
if errorlevel 1 exit /b 1" 2>&1 | tee "${LOG_DIR}/build_benchmark.txt"
[ -f "${WORK_DIR}/bench_eval.exe" ] || die "bench_eval.exe was not produced; see the messages above."

RUNS_FILE="${OUT_DIR}/bench_eval_runs.txt"
: > "$RUNS_FILE"
for run_number in $(seq 1 "$BENCH_RUNS"); do
    say "Benchmark run $run_number of $BENCH_RUNS"
    (cd "$WORK_DIR" && ./bench_eval.exe nbrdf_weights.bin) | tee -a "$RUNS_FILE"
done

say "Median over $BENCH_RUNS runs"
"$(venv_python)" - "$RUNS_FILE" <<'PYCODE' | tee "${OUT_DIR}/bench_eval_median.txt"
import re, statistics, sys
values = {}
for line in open(sys.argv[1]):
    m = re.match(r"\s*(.+?)\s*:\s*([\d.]+) ns/eval", line)
    if m:
        values.setdefault(m.group(1), []).append(float(m.group(2)))
for name, v in values.items():
    print(f"{name:<20} median {statistics.median(v):9.2f} ns/eval  (runs: {', '.join(f'{x:.2f}' for x in v)})")
PYCODE
echo
echo "Published values (Intel Core Ultra 9 288V): EvalReflectance 1292.96 ns,"
echo "EvalNetwork only 1211.24 ns, StandardToHalfDiff 77.38 ns. Your numbers will"
echo "differ on different hardware; the ratios between the three lines are what"
echo "should look similar."
