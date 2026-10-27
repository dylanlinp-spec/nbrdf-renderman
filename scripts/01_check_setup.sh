#!/usr/bin/env bash
# =====================================================================
# 01_check_setup.sh -- check that everything this project needs is
# installed and findable, BEFORE you spend time on anything else.
#
# Usage:   bash scripts/01_check_setup.sh
# Changes nothing on your computer. Safe to run as often as you like.
# =====================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

problems=0
ok()   { printf '  [ OK ]  %s\n' "$*"; }
bad()  { printf '  [FAIL]  %s\n' "$*"; problems=$((problems + 1)); }
warn() { printf '  [WARN]  %s\n' "$*"; }

say "Operating system"
if is_windows; then ok "Windows (Git Bash) detected"
else warn "Not Windows. Training and the Python analyses work anywhere, but building the plugin and rendering need Windows + RenderMan, exactly as in the paper."
fi

say "Python"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    ok "$("$PYTHON_BIN" --version 2>&1) found as '$PYTHON_BIN'"
else
    bad "'$PYTHON_BIN' not found. Install Python 3.11 from python.org and tick 'Add python.exe to PATH', then reopen Git Bash."
fi
if [ -x "${VENV_DIR}/Scripts/python.exe" ] || [ -x "${VENV_DIR}/bin/python" ]; then
    ok "project Python environment exists (.venv)"
else
    warn "project Python environment not created yet. Next: bash scripts/02_setup_python.sh"
fi

say "MERL data (needed for training, and for the ground-truth comparisons)"
for material in alum-bronze chrome; do
    file="${MERL_DIR}/${material}.binary"
    if [ -f "$file" ]; then
        size=$(wc -c < "$file" | tr -d ' ')
        if [ "$size" = "34992012" ]; then ok "$material.binary ($size bytes)"
        else bad "$material.binary is $size bytes, expected 34992012. Re-download it."
        fi
    else
        warn "$file not found. See data/README.md. (Only needed for training and the GT comparisons.)"
    fi
done

say "Shipped trained weights (the exact files behind every published number)"
if command -v sha256sum >/dev/null 2>&1; then
    if (cd "$WEIGHTS_DIR" && sha256sum -c --quiet SHA256SUMS); then ok "all weight files match results/weights/SHA256SUMS"
    else bad "a weight file does not match its checksum. Re-download the repository."
    fi
else
    warn "sha256sum not available; skipping checksum test"
fi

say "RenderMan"
case "$RMANTREE" in
    path_to_your_*|"") bad "RMANTREE is not set. Edit scripts/config.sh (see README, step 4)." ;;
    *)
        if [ -f "${RMANTREE}/bin/prman.exe" ] || [ -f "${RMANTREE}/bin/prman" ]; then ok "prman found in ${RMANTREE}/bin"
        else bad "prman not found in ${RMANTREE}/bin. Is RMANTREE the RenderManProServer-27.x folder?"
        fi
        if [ -f "${RMANTREE}/include/RixBxdf.h" ]; then ok "RenderMan C++ headers found"
        else bad "RixBxdf.h not found in ${RMANTREE}/include"
        fi
        if [ -f "${RMANTREE}/lib/libpxrcore.lib" ]; then ok "libpxrcore.lib found"
        else bad "libpxrcore.lib not found in ${RMANTREE}/lib"
        fi
        ;;
esac

say "Visual Studio C++ compiler (MSVC)"
if is_windows; then
    if vcvars="$(find_vcvars64)"; then ok "vcvars64.bat: $vcvars"
    else bad "vcvars64.bat not found. Install 'Build Tools for Visual Studio 2022' with 'Desktop development with C++'."
    fi
else
    warn "skipped (Windows only)"
fi

say "Work folder"
case "$WORK_DIR" in
    *" "*) bad "WORK_DIR contains a space: '$WORK_DIR'. Pick a path without spaces." ;;
    *) ok "builds and renders will go to $WORK_DIR" ;;
esac

echo
if [ "$problems" -eq 0 ]; then
    echo "All required checks passed (warnings are fine). Next: bash scripts/02_setup_python.sh"
else
    echo "$problems problem(s) found. Fix the [FAIL] lines above, then run this script again."
    exit 1
fi
