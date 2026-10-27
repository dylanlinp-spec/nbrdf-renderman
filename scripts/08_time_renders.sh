#!/usr/bin/env bash
# =====================================================================
# 08_time_renders.sh -- Track C2: whole-render time of the same scene
# with three materials (NBRDF, PxrSurface, PxrDiffuse).
#
# Uses the published settings: prman -t:8, 1 discarded warm-up, then 6
# timed runs per scene, median reported (src/evaluation/time_render.py).
#
# Usage:   bash scripts/08_time_renders.sh
# Takes:   about 2-5 minutes.
# Output:  results/reproduced/timing.csv
# Tip:     plug in your laptop, close other programs, and do not touch
#          the computer while this runs.
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_setting RMANTREE "$RMANTREE" "It should point at your RenderManProServer-27.x folder."
[ -f "${WORK_DIR}/NBrdf.dll" ] || die "NBrdf.dll not found in ${WORK_DIR}. Run: bash scripts/04_build_plugin.sh"
PY="$(venv_python)"

WORK_DIR_MIXED="$(mixed_path "$WORK_DIR")"
cp "${WEIGHTS_DIR}/nbrdf_weights.bin" "${WORK_DIR}/"
for scene in c2_nbrdf c2_pxrsurface c2_pxrdiffuse; do
    sed "s#C:/NBrdfBuild#${WORK_DIR_MIXED}#g" "${REPO_ROOT}/scenes/${scene}.rib" > "${WORK_DIR}/${scene}.rib"
done

set_plugin_path
PRMAN="$(prman_exe)"
TIMING_CSV="${OUT_DIR}/timing.csv"
rm -f "$TIMING_CSV"
cd "$WORK_DIR"
for pair in "NBRDF:c2_nbrdf" "PxrSurface:c2_pxrsurface" "PxrDiffuse:c2_pxrdiffuse"; do
    label="${pair%%:*}"; scene="${pair#*:}"
    say "Timing $label ($scene.rib)"
    "$PY" "${REPO_ROOT}/src/evaluation/time_render.py" --rib "${scene}.rib" \
        --runs "$TIMING_RUNS" --label "$label" --threads-note "t:${TIMING_THREADS}" \
        --csv "$TIMING_CSV" -- "$PRMAN" "-t:${TIMING_THREADS}" "{rib}" | tee "${LOG_DIR}/timing_${scene}.txt"
done

say "Ratios"
"$PY" - "$TIMING_CSV" <<'PYCODE'
import csv, sys
rows = {r["label"]: float(r["median_s"]) for r in csv.DictReader(open(sys.argv[1]))}
print(f"NBRDF / PxrDiffuse = {rows['NBRDF'] / rows['PxrDiffuse']:.2f}x   (published: 5.20x)")
print(f"NBRDF / PxrSurface = {rows['NBRDF'] / rows['PxrSurface']:.2f}x   (published: 3.89x)")
PYCODE
