#!/usr/bin/env bash
# =====================================================================
# 04_build_plugin.sh -- compile the NBrdf RenderMan plugin (NBrdf.dll).
#
# Uses exactly the compiler command written in the header of
# src/plugin/NBrdf.cpp. Note: that command has NO /O2 (optimisation)
# flag, and the evidence in the original build folder says the plugin
# used for the published timings was built this way. Keep it, or the
# timing numbers will not be comparable. See doc/known_issues.md.
#
# Usage:   bash scripts/04_build_plugin.sh
# Takes:   under a minute.
# Output:  ${WORK_DIR}/NBrdf.dll
# Needs:   Windows, RenderMan 27.x, Visual Studio Build Tools (MSVC).
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
is_windows || die "The plugin can only be built on Windows (RenderMan's libraries are MSVC-built)."
require_setting RMANTREE "$RMANTREE" "It should point at your RenderManProServer-27.x folder."

mkdir -p "${WORK_DIR}/compat"
cp "${REPO_ROOT}/src/plugin/NBrdf.cpp"  "${WORK_DIR}/"
cp "${REPO_ROOT}/src/plugin/NBrdf.args" "${WORK_DIR}/"
cp "${REPO_ROOT}/src/plugin/compat/"*.h "${WORK_DIR}/compat/"
rm -f "${WORK_DIR}/NBrdf.obj" "${WORK_DIR}/NBrdf.dll"   # never link a stale object file

RMANTREE_WIN="$(win_path "$RMANTREE")"
say "Compiling NBrdf.cpp"
run_with_msvc "cl /std:c++17 /EHsc /D_USE_MATH_DEFINES /FI\"compat\\gcc_builtins.h\" /I\"compat\" /I\"${RMANTREE_WIN}\\include\" /c NBrdf.cpp
if errorlevel 1 exit /b 1
link /DLL /OUT:NBrdf.dll NBrdf.obj \"${RMANTREE_WIN}\\lib\\libpxrcore.lib\"
if errorlevel 1 exit /b 1" 2>&1 | tee "${LOG_DIR}/build_plugin.txt"

[ -f "${WORK_DIR}/NBrdf.dll" ] || die "NBrdf.dll was not produced. Read the compiler messages above (also saved in results/reproduced/logs/build_plugin.txt)."
echo
echo "Done. Built ${WORK_DIR}/NBrdf.dll"
echo "Next: bash scripts/05_render_scenes.sh"
