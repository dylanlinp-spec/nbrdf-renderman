#!/usr/bin/env bash
# =====================================================================
# 06_render_scenes.sh -- render every RenderMan scene used in the paper.
#
# Copies the scenes (scenes/*.rib) and the shipped weights into WORK_DIR,
# then renders each scene with prman. The scene files are used exactly as
# they are in the repo, with ONE automatic edit: the original weights path
# C:/NBrdfBuild/ is replaced by your WORK_DIR, because that was the author's
# own folder. Nothing else in any scene changes.
#
# Usage:   bash scripts/06_render_scenes.sh            # all scenes
#          bash scripts/06_render_scenes.sh furnace_alum a2_distant_alum   # just some
# Takes:   roughly 15-60 minutes in total, depending on your CPU. The
#          512-samples-per-pixel correctness scenes are the slow ones.
# Output:  one .exr image per scene, in WORK_DIR.
# Needs:   scripts/04_build_plugin.sh to have been run first.
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_setting RMANTREE "$RMANTREE" "It should point at your RenderManProServer-27.x folder."
[ -f "${WORK_DIR}/NBrdf.dll" ] || die "NBrdf.dll not found in ${WORK_DIR}. Run: bash scripts/04_build_plugin.sh"

ALL_SCENES=(
    # Track A1: white furnace (energy conservation through the renderer)
    furnace_alum furnace_chrome furnace_alum_spherelight
    # Track A2: analytic distant-light match
    a2_distant_alum
    # Track B: sampling-mode ablation (512 spp and 16 spp)
    furnace_alum_cosine furnace_alum_ggx
    b_distant_alum_cosine b_distant_alum_ggx
    b16_furnace_alum_mixture b16_furnace_alum_cosine b16_furnace_alum_ggx
    # Qualitative two-sphere figure
    test_scene
)
if [ "$#" -gt 0 ]; then SCENES=("$@"); else SCENES=("${ALL_SCENES[@]}"); fi

WORK_DIR_MIXED="$(mixed_path "$WORK_DIR")"
cp "${WEIGHTS_DIR}/nbrdf_weights.bin" "${WEIGHTS_DIR}/nbrdf_weights_chrome.bin" "${WORK_DIR}/"
for rib in "${REPO_ROOT}"/scenes/*.rib; do
    sed "s#C:/NBrdfBuild#${WORK_DIR_MIXED}#g" "$rib" > "${WORK_DIR}/$(basename "$rib")"
done

set_plugin_path
PRMAN="$(prman_exe)"
for scene in "${SCENES[@]}"; do
    [ -f "${WORK_DIR}/${scene}.rib" ] || die "No scene called ${scene}.rib in scenes/."
    say "Rendering ${scene}.rib"
    start_seconds=$SECONDS
    (cd "$WORK_DIR" && "$PRMAN" "-t:${RENDER_THREADS}" "${scene}.rib") 2>&1 | tee "${LOG_DIR}/render_${scene}.txt"
    echo "    finished in $((SECONDS - start_seconds)) s"
    grep -q "NBrdf: loaded" "${LOG_DIR}/render_${scene}.txt" || \
        printf '    NOTE: no "NBrdf: loaded" line in the log. If this scene uses NBrdf, the\n          plugin or the weights file was not found; see doc/troubleshooting.md.\n'
done

echo
echo "Done. Images are in ${WORK_DIR}. Next: bash scripts/07_run_analysis.sh"
