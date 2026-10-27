#!/usr/bin/env bash
# =====================================================================
# 03_train_networks.sh -- OPTIONAL. Retrain the two networks from the
# MERL data, using exactly the original training code and settings.
#
# You do NOT need this to reproduce the published numbers: the trained
# weights that produced them are already in results/weights/. Run this to
# check the training method itself.
#
# The original training never fixed PyTorch's random seed, so your new
# networks will be close to, but not bit-identical with, the shipped ones.
# They are written to results/reproduced/retrained/ and never overwrite
# results/weights/.
#
# Usage:   bash scripts/03_train_networks.sh
# Takes:   about 5-20 minutes per material on a laptop CPU.
# Needs:   data/merl/alum-bronze.binary and data/merl/chrome.binary
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
PY="$(venv_python)"
TRAIN_OUT="${OUT_DIR}/retrained"
mkdir -p "$TRAIN_OUT"

for material in alum-bronze chrome; do
    merl_file="${MERL_DIR}/${material}.binary"
    [ -f "$merl_file" ] || die "$merl_file not found. See data/README.md for the download."
    say "Training $material (20,000 steps)"
    "$PY" "${REPO_ROOT}/src/training/train_nbrdf.py" \
        --material "$merl_file" \
        --weights-out "${TRAIN_OUT}/${material}.pt" \
        --renderman-out "${TRAIN_OUT}/${material}.bin" \
        2>&1 | tee "${LOG_DIR}/train_${material}.txt"
done

echo
echo "Done. New weights and plots are in results/reproduced/retrained/."
echo "To render with them instead of the shipped weights, copy a .bin over the"
echo "matching file in ${WORK_DIR} after running scripts/05_render_scenes.sh once."
