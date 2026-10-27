#!/usr/bin/env bash
# =====================================================================
# 02_setup_python.sh -- create a private Python environment (.venv) for
# this project and install the packages listed in requirements.txt.
#
# Usage:   bash scripts/02_setup_python.sh
# Takes:   2-10 minutes (PyTorch is a large download, about 200 MB+).
# Safe to re-run: it reuses the environment if it already exists.
# =====================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "'$PYTHON_BIN' not found. Install Python 3.11 (python.org) and tick 'Add python.exe to PATH'."

if [ ! -d "$VENV_DIR" ]; then
    say "Creating the project environment in .venv"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$(venv_python)"
say "Upgrading pip"
"$PY" -m pip install --upgrade pip

say "Installing packages from requirements.txt"
"$PY" -m pip install -r "${REPO_ROOT}/requirements.txt"

say "Checking the packages import"
"$PY" - <<'PYCODE'
import numpy, torch, matplotlib, cv2, tifffile
print("numpy", numpy.__version__, "| torch", torch.__version__,
      "| matplotlib", matplotlib.__version__, "| opencv", cv2.__version__,
      "| tifffile", tifffile.__version__)
PYCODE

say "Running the built-in self-tests (no data files needed)"
"$PY" "${REPO_ROOT}/src/evaluation/compute_albedo.py" --self-test
"$PY" "${REPO_ROOT}/src/evaluation/furnace_check.py" --self-test

"$PY" -m pip freeze > "${LOG_DIR}/pip_freeze.txt"
echo
echo "Done. Exact package versions were saved to results/reproduced/logs/pip_freeze.txt."
echo "Next: bash scripts/03_train_networks.sh  (optional)  or  bash scripts/04_build_plugin.sh"
