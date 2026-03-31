#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ZIP="${1:-$HOME/Desktop/physics_solution.zip}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BUNDLE_DIR="$TMP_DIR/physics_solution"
mkdir -p "$BUNDLE_DIR"

FILES=(
  ".gitignore"
  "PhysicsSolution_Colab_OneClick.ipynb"
  "README.md"
  "README_COLAB.md"
  "bootstrap_mac_env.sh"
  "checkerboard_eval_summary.md"
  "checkerboard_rectification.py"
  "full_physics_solution.py"
  "full_pipeline.command"
  "geometry_reasoning.py"
  "infer.command"
  "make_colab_bundle.sh"
  "requirements-mac.txt"
  "run_colab_oneclick.py"
  "run_colab_oneclick.sh"
  "train.command"
)

for name in "${FILES[@]}"; do
  if [ -e "$SCRIPT_DIR/$name" ]; then
    cp -R "$SCRIPT_DIR/$name" "$BUNDLE_DIR/$name"
  fi
done

mkdir -p "$(dirname "$OUT_ZIP")"
rm -f "$OUT_ZIP"
(cd "$TMP_DIR" && zip -qr "$OUT_ZIP" physics_solution)

echo "Created Colab bundle: $OUT_ZIP"
