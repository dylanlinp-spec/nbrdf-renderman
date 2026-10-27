#!/usr/bin/env bash
# =====================================================================
# lib.sh -- small helper functions shared by the numbered scripts.
# You never run this file directly; the other scripts load it.
# =====================================================================

# shellcheck source=scripts/config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

# Print a step heading.
say() { printf '\n==> %s\n' "$*"; }

# Print an error in plain words and stop.
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# True when running in Git Bash / MSYS / Cygwin on Windows.
is_windows() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

# Stop if a setting still holds its lower_case placeholder value.
require_setting() {
    local setting_name="$1" setting_value="$2" hint="$3"
    case "$setting_value" in
        path_to_your_*|"")
            die "$setting_name is not set. $hint
Open scripts/config.sh and replace the placeholder, or run:
    $setting_name=\"your_value_here\" bash $0" ;;
    esac
}

# Convert a path to the C:\style\path Windows programs expect.
win_path() {
    if is_windows; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

# Convert a path to the C:/style/path RenderMan scene files use.
mixed_path() {
    if is_windows; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

# Path of the Python inside this project's virtual environment.
venv_python() {
    if [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
        printf '%s' "${VENV_DIR}/Scripts/python.exe"
    elif [ -x "${VENV_DIR}/bin/python" ]; then
        printf '%s' "${VENV_DIR}/bin/python"
    else
        die "The Python environment is missing. Run: bash scripts/02_setup_python.sh"
    fi
}

# Find vcvars64.bat if the user left the placeholder.
find_vcvars64() {
    if [ -n "${VCVARS64_BAT}" ] && [ "${VCVARS64_BAT#path_to_your_}" = "${VCVARS64_BAT}" ]; then
        printf '%s' "${VCVARS64_BAT}"; return 0
    fi
    local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    if [ -x "$vswhere" ]; then
        local install_path
        install_path="$("$vswhere" -latest -products '*' \
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
            -property installationPath 2>/dev/null | tr -d '\r')"
        if [ -n "$install_path" ]; then
            local candidate
            candidate="$(cygpath -u "$install_path")/VC/Auxiliary/Build/vcvars64.bat"
            if [ -f "$candidate" ]; then printf '%s' "$candidate"; return 0; fi
        fi
    fi
    return 1
}

# Run Windows batch-file text with the Visual Studio compiler available.
# Usage: run_with_msvc "<lines of cmd.exe commands>"
run_with_msvc() {
    local commands="$1" vcvars bat_file line
    vcvars="$(find_vcvars64)" || die "Could not find vcvars64.bat (Visual Studio Build Tools).
Install 'Build Tools for Visual Studio 2022' with the 'Desktop development
with C++' workload, or set VCVARS64_BAT in scripts/config.sh."
    bat_file="${WORK_DIR}/run_with_msvc.bat"
    {
        printf '@echo off\r\n'
        printf 'call "%s" >nul\r\n' "$(win_path "$vcvars")"
        printf 'if errorlevel 1 exit /b 1\r\n'
        printf 'cd /d "%s"\r\n' "$(win_path "$WORK_DIR")"
        while IFS= read -r line; do
            printf '%s\r\n' "$line"
        done <<< "$commands"
    } > "$bat_file"
    cmd.exe //c "$(win_path "$bat_file")"
}

# Tell RenderMan where to find NBrdf.dll (the work folder) without losing
# its own built-in plugins.
set_plugin_path() {
    local work_win plugins_win
    work_win="$(win_path "$WORK_DIR")"
    plugins_win="$(win_path "${RMANTREE}/lib/plugins")"
    export RMAN_RIXPLUGINPATH="${work_win};${plugins_win}"
}

# Path of prman.exe.
prman_exe() {
    local exe="${RMANTREE}/bin/prman.exe"
    [ -f "$exe" ] || exe="${RMANTREE}/bin/prman"
    [ -f "$exe" ] || die "prman was not found in ${RMANTREE}/bin. Check RMANTREE in scripts/config.sh."
    printf '%s' "$exe"
}

mkdir -p "$LOG_DIR" 2>/dev/null || true
