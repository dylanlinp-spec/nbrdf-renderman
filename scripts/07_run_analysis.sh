#!/usr/bin/env bash
# =====================================================================
# 07_run_analysis.sh -- turn the renders and the weights into every
# results table in the paper (Tracks A and B).
#
# Each step prints what it is computing and which published file it
# should match. Every result goes to results/reproduced/ so the original
# numbers in results/tables/ are never overwritten. At the end,
# compare_with_published.py lists which reproduced tables match.
#
# Usage:   bash scripts/07_run_analysis.sh
# Takes:   about 5-10 minutes (mostly the no-renderer variance integral).
# Needs:   02_setup_python.sh, and 06_render_scenes.sh for the render-based
#          checks. Steps whose inputs are missing are skipped with a note.
# Optional: data/merl/*.binary enables the ground-truth (GT) columns.
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
PY="$(venv_python)"
EVAL="${REPO_ROOT}/src/evaluation"
mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"

have() { [ -f "$1" ] || { printf '    SKIPPED: %s not found\n' "$1"; return 1; }; }
log()  { tee "${LOG_DIR}/$1.txt"; }

# --- Make a .pt that holds EXACTLY the rendered alum-bronze network ----
say "Rebuilding a .pt from the rendered alum-bronze .bin (see doc/known_issues.md)"
"$PY" "${REPO_ROOT}/src/training/bin_to_pt.py" \
    --bin "${WEIGHTS_DIR}/nbrdf_weights.bin" --out "${WORK_DIR}/nbrdf_weights_from_bin.pt"

# --- Table 1: directional albedo / energy conservation (no renderer) ---
say "Table 1, alum-bronze, as published (uses nbrdf_weights_v2.pt)"
"$PY" "${EVAL}/compute_albedo.py" --pt "${WEIGHTS_DIR}/nbrdf_weights_v2.pt" \
    --csv "${OUT_DIR}/albedo_alum.csv" | log albedo_alum

say "Table 1, alum-bronze, for the network that was actually rendered"
ALUM_MERL_ARGS=()
[ -f "${MERL_DIR}/alum-bronze.binary" ] && ALUM_MERL_ARGS=(--merl "${MERL_DIR}/alum-bronze.binary")
"$PY" "${EVAL}/compute_albedo.py" "${ALUM_MERL_ARGS[@]}" --pt "${WORK_DIR}/nbrdf_weights_from_bin.pt" \
    --csv "${OUT_DIR}/albedo_alum_rendered_network.csv" | log albedo_alum_rendered_network

say "Table 1, chrome (GT + NBRDF)"
if have "${MERL_DIR}/chrome.binary"; then
    "$PY" "${EVAL}/compute_albedo.py" --merl "${MERL_DIR}/chrome.binary" \
        --pt "${WEIGHTS_DIR}/nbrdf_weights_v2_chrome.pt" --csv "${OUT_DIR}/albedo_chrome.csv" | log albedo_chrome
fi

# --- Track A1: white furnace -------------------------------------------
ALUM_REF="${OUT_DIR}/albedo_alum_rendered_network.csv"
furnace() {  # furnace <exr name without .exr> <albedo csv> <output name>
    if have "${WORK_DIR}/$1.exr"; then
        "$PY" "${EVAL}/furnace_check.py" --exr "$1.exr" --albedo-csv "$2" \
            --out "${OUT_DIR}/$3.csv" | log "$3"
    fi
}
say "Track A1: furnace renders vs the no-renderer albedo curve"
furnace furnace_alum             "$ALUM_REF" furnace_alum_compare
furnace furnace_alum_spherelight "$ALUM_REF" furnace_alum_spherelight_compare
[ -f "${OUT_DIR}/albedo_chrome.csv" ] && furnace furnace_chrome "${OUT_DIR}/albedo_chrome.csv" furnace_chrome_compare

# --- Track A2: analytic distant-light match ----------------------------
say "Track A2: analytic per-pixel reference (render_gt_comparison.py)"
if have "${MERL_DIR}/alum-bronze.binary" && have "${MERL_DIR}/chrome.binary"; then
    "$PY" "${EVAL}/render_gt_comparison.py" | log render_gt_comparison
    cp gt_vs_nbrdf.png "${OUT_DIR}/"
    if have "${WORK_DIR}/a2_distant_alum.exr"; then
        say "Track A2, as published (reference from nbrdf_weights_v2.pt)"
        "$PY" "${EVAL}/a2_check.py" --exr a2_distant_alum.exr --label alum-bronze \
            --out "${OUT_DIR}/a2_alum_compare.csv" | log a2_alum_compare
        for mode in cosine ggx; do
            have "b_distant_alum_${mode}.exr" && "$PY" "${EVAL}/a2_check.py" \
                --exr "b_distant_alum_${mode}.exr" --label alum-bronze \
                --out "${OUT_DIR}/b_distant_alum_${mode}_compare.csv" | log "b_distant_alum_${mode}_compare"
        done
        say "Track A2, reference rebuilt from the rendered network"
        mkdir -p rendered_network && (
            cd rendered_network
            ALUM_BRONZE_PT="${WORK_DIR}/nbrdf_weights_from_bin.pt" WEIGHTS_DIR="${WEIGHTS_DIR}" \
                "$PY" "${EVAL}/render_gt_comparison.py" > "${LOG_DIR}/render_gt_comparison_rendered_network.txt"
            "$PY" "${EVAL}/a2_check.py" --exr ../a2_distant_alum.exr --label alum-bronze \
                --out "${OUT_DIR}/a2_alum_compare_rendered_network.csv" | log a2_alum_compare_rendered_network
        )
    fi
else
    echo "    (needs both MERL files; see data/README.md)"
fi

# --- Track B: sampling-mode ablation through the renderer --------------
say "Track B (rendered): 512 spp furnace, one sampling technique at a time"
furnace furnace_alum_cosine "$ALUM_REF" furnace_alum_cosine_compare
furnace furnace_alum_ggx    "$ALUM_REF" furnace_alum_ggx_compare
say "Track B (rendered): 16 spp furnace, bin means and per-pixel noise"
for mode in mixture cosine ggx; do
    furnace "b16_furnace_alum_${mode}" "$ALUM_REF" "b16_furnace_alum_${mode}_compare"
    [ -f "b16_furnace_alum_${mode}.exr" ] && "$PY" "${EVAL}/track_b_noise.py" \
        --exr "b16_furnace_alum_${mode}.exr" --label "$mode" | log "b16_noise_${mode}"
done

# --- Track B: analytic estimator variance (Tables 3 and 3b) ------------
say "Table 3: single-sample variance per sampling technique (alpha = 0.05)"
"$PY" "${EVAL}/sampler_variance.py" --weights "${WEIGHTS_DIR}/nbrdf_weights.bin" \
    --label alum-bronze --csv "${OUT_DIR}/table3_variance.csv" | log table3_variance
say "Table 3b: the same, sweeping the GGX roughness alpha"
"$PY" "${EVAL}/sampler_variance.py" --weights "${WEIGHTS_DIR}/nbrdf_weights.bin" \
    --label alum-bronze --alphas 0.02 0.05 0.1 0.2 \
    --csv "${OUT_DIR}/table3_alpha_sweep.csv" | log table3_alpha_sweep

# --- Qualitative figure -------------------------------------------------
say "Tone-mapping the two-sphere test render"
if have "${WORK_DIR}/test_render.exr"; then
    "$PY" "${EVAL}/expose_render.py" | log expose_render
    cp test_render_exposed.png "${OUT_DIR}/"
fi

say "Comparing reproduced tables with the published ones"
"$PY" "${EVAL}/compare_with_published.py" | tee "${OUT_DIR}/comparison_summary.txt"
