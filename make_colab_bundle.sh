#!/usr/bin/env bash
# make_colab_bundle.sh — Colab 업로드용 프로젝트 zip 생성
# 사용법: bash make_colab_bundle.sh [출력경로.zip]
# 기본 출력: ~/Desktop/physics_solution.zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ZIP="${1:-$HOME/Desktop/physics_solution.zip}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BUNDLE_DIR="$TMP_DIR/physics_solution"
mkdir -p "$BUNDLE_DIR"

# archive/ 및 불필요 파일은 제외하고 핵심 파일만 번들링
FILES=(
  ".gitignore"
  "README.md"
  "PIPELINE_ANALYSIS.md"
  "bootstrap_mac_env.sh"
  "checkerboard_rectification.py"
  "full_physics_solution.py"
  "full_pipeline.command"
  "geometry_reasoning.py"
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

echo "✅ Created Colab bundle: $OUT_ZIP"
