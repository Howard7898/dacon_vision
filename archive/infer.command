#!/usr/bin/env bash
# infer.command — macOS 더블클릭 실행용 (추론만)
# v0.05 추천 파라미터 적용: batch-size 32, image-size 336,
#                           tta-passes 8, num-workers 4
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

python full_physics_solution.py infer \
    --data-root "${PHYSICS_DATA_ROOT:-./data}" \
    --out-dir ./runs/infer \
    --backbone dinov2_vits14_reg \
    --image-size 336 \
    --batch-size 32 \
    --num-workers 4 \
    --tta-passes 8 \
    --pretrained

echo "✅ 추론 완료"
read -p "Press Enter to close..."
