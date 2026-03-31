#!/usr/bin/env bash
# train.command — macOS 더블클릭 실행용 (학습만)
# v0.05 추천 파라미터 적용: batch-size 32, image-size 336, backbone-lr 2e-5,
#                           weight-decay 5e-5, num-workers 4
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

python full_physics_solution.py train \
    --data-root "${PHYSICS_DATA_ROOT:-./data}" \
    --out-dir ./runs/train \
    --backbone dinov2_vits14_reg \
    --image-size 336 \
    --batch-size 32 \
    --epochs 20 \
    --num-folds 5 \
    --num-workers 4 \
    --backbone-lr 2e-5 \
    --warmup-epochs 3 \
    --weight-decay 5e-5 \
    --pretrained

echo "✅ 학습 완료"
read -p "Press Enter to close..."
