#!/usr/bin/env bash
# full_pipeline.command — macOS 더블클릭 실행용 (전체 파이프라인)
# 실제 제출 재현 파라미터:
#   image-size 336, backbone-lr 1e-5, weight-decay 1e-4
#   batch-size 16, epochs 30, tta-passes 4
#   --use-domain-head, --enable-geometry-reasoning, --refresh-motion
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

python full_physics_solution.py full-run \
    --data-root "${PHYSICS_DATA_ROOT:-./data}" \
    --out-dir ./runs/final \
    --backbone dinov2_vits14_reg \
    --image-size 336 \
    --batch-size 16 \
    --epochs 30 \
    --num-folds 5 \
    --num-workers 4 \
    --tta-passes 4 \
    --backbone-lr 1e-5 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --pretrained \
    --use-domain-head \
    --enable-geometry-reasoning \
    --refresh-motion

echo "✅ 파이프라인 완료"
read -p "Press Enter to close..."
