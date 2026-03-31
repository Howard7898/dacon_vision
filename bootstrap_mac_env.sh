#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SCRIPT_DIR"

if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$SCRIPT_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$SCRIPT_DIR/requirements-mac.txt"
