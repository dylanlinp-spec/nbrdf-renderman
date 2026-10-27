#!/usr/bin/env bash
# =====================================================================
# config.sh -- the ONLY file you should need to edit.
#
# Every other script in this folder reads its settings from here.
#
# Naming rules used throughout these scripts:
#   UPPER_CASE_WITH_UNDERSCORES  = a setting (variable) name. Do not rename.
#   lower_case_with_underscores  = a placeholder value you must replace
#                                  with something from YOUR computer.
#
# Example: if RenderMan is installed in
#   C:\Program Files\Pixar\RenderManProServer-27.3
# change
#   RMANTREE_DEFAULT="path_to_your_renderman_install"
# to
#   RMANTREE_DEFAULT="C:/Program Files/Pixar/RenderManProServer-27.3"
# (Use forward slashes. Keep the double quotes.)
#
# Any setting can also be given on the command line instead, e.g.
#   WORK_DIR="D:/my_render_folder" bash scripts/run_all.sh
# =====================================================================

# --- 1. Where RenderMan is installed --------------------------------
# The RenderMan installer normally sets RMANTREE for you. If it did, the
# placeholder below is ignored. Find yours by opening a Command Prompt and
# typing:  echo %RMANTREE%
RMANTREE_DEFAULT="<path_to_renderman_install>"

# --- 2. Where Visual Studio's vcvars64.bat lives ----------------------
# Leave the placeholder alone if you installed Visual Studio Build Tools
# 2022 in the default place; scripts/lib.sh finds it automatically.
# Otherwise search your C: drive for vcvars64.bat and paste its path.
VCVARS64_BAT_DEFAULT="path_to_your_vcvars64_bat"

# --- 3. A scratch folder for building and rendering -------------------
# The plugin, weights, scenes and every rendered image are put here.
# It is created for you. Use a path WITHOUT spaces.
WORK_DIR_DEFAULT="C:/NBrdfRepro"

# --- 4. Where you put the MERL .binary files --------------------------
# See data/README.md. Default: the data/merl folder inside this repo.
MERL_DIR_DEFAULT="__repo__/data/merl"

# --- 5. Python launcher -----------------------------------------------
# "python" on most Windows installs; "py" if you installed from python.org
# without ticking "Add to PATH"; "python3" on macOS/Linux.
PYTHON_BIN_DEFAULT="python"

# --- 6. Measurement settings used for the published numbers -----------
# Change these only if you know you want different measurements.
TIMING_THREADS_DEFAULT="8"    # prman -t:8 for the whole-render timings
TIMING_RUNS_DEFAULT="6"       # timed runs per scene (after 1 warm-up)
BENCH_RUNS_DEFAULT="5"        # repeats of the per-evaluation benchmark
RENDER_THREADS_DEFAULT="0"    # prman -t:0 (all cores) for correctness renders

# =====================================================================
# Nothing below this line needs editing.
# =====================================================================
# "pwd -W" (Git Bash) gives C:/style paths, which both bash and Windows
# programs such as python.exe and prman.exe understand.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && { pwd -W 2>/dev/null || pwd; })"

RMANTREE="${RMANTREE:-$RMANTREE_DEFAULT}"
VCVARS64_BAT="${VCVARS64_BAT:-$VCVARS64_BAT_DEFAULT}"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"
MERL_DIR="${MERL_DIR:-${MERL_DIR_DEFAULT/__repo__/$REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
TIMING_THREADS="${TIMING_THREADS:-$TIMING_THREADS_DEFAULT}"
TIMING_RUNS="${TIMING_RUNS:-$TIMING_RUNS_DEFAULT}"
BENCH_RUNS="${BENCH_RUNS:-$BENCH_RUNS_DEFAULT}"
RENDER_THREADS="${RENDER_THREADS:-$RENDER_THREADS_DEFAULT}"

VENV_DIR="${REPO_ROOT}/.venv"
WEIGHTS_DIR="${REPO_ROOT}/results/weights"
OUT_DIR="${REPO_ROOT}/results/reproduced"
LOG_DIR="${OUT_DIR}/logs"

export RMANTREE VCVARS64_BAT WORK_DIR MERL_DIR PYTHON_BIN
export TIMING_THREADS TIMING_RUNS BENCH_RUNS RENDER_THREADS
export REPO_ROOT VENV_DIR WEIGHTS_DIR OUT_DIR LOG_DIR
export OPENCV_IO_ENABLE_OPENEXR=1
