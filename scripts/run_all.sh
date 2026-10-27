#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- run every step in order (except optional retraining).
#
# Usage:   bash scripts/run_all.sh
# Takes:   about 30-90 minutes in total.
# Stops at the first step that fails and tells you which one.
# To also retrain the networks:  RETRAIN=yes bash scripts/run_all.sh
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=(01_check_setup 02_setup_python)
[ "${RETRAIN:-no}" = "yes" ] && STEPS+=(03_train_networks)
STEPS+=(04_build_plugin 05_run_benchmark 06_render_scenes 07_run_analysis 08_time_renders)
for step in "${STEPS[@]}"; do
    printf '\n######## %s ########\n' "$step"
    bash "${HERE}/${step}.sh" || { printf '\nStopped: %s.sh failed. Fix the error above, then re-run that step:\n    bash scripts/%s.sh\n' "$step" "$step"; exit 1; }
done
printf '\nAll steps finished. Results: results/reproduced/  Summary: results/reproduced/comparison_summary.txt\n'
