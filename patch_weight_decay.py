#!/usr/bin/env python3
"""
patch_weight_decay.py
full_physics_solution.py 에 --weight-decay 인자를 자동으로 추가하는 패치 스크립트.
같은 폴더에 full_physics_solution.py 가 있어야 합니다.

사용법:
    python patch_weight_decay.py
"""
import re, sys
from pathlib import Path

TARGET = Path(__file__).parent / "full_physics_solution.py"

if not TARGET.exists():
    sys.exit(f"❌ 파일 없음: {TARGET}")

src = TARGET.read_text(encoding="utf-8")

# ── 1. argparse에 --weight-decay 추가 ─────────────────────────────────────
# --warmup-epochs 정의 줄 바로 뒤에 삽입
WD_ARG = (
    '    parser.add_argument("--weight-decay", type=float, default=5e-5,\n'
    '                        help="AdamW weight decay (default: 5e-5)")\n'
)
ANCHOR_ARG = re.compile(
    r'(parser\.add_argument\(["\']--warmup-epochs["\'].*?\)[ \t]*\n)',
    re.DOTALL,
)
if "--weight-decay" in src:
    print("ℹ️  --weight-decay 인자가 이미 존재합니다. argparse 패치를 건너뜁니다.")
else:
    m = ANCHOR_ARG.search(src)
    if not m:
        sys.exit("❌ --warmup-epochs 정의 줄을 찾을 수 없습니다. 수동으로 추가해 주세요.")
    src = src[: m.end()] + WD_ARG + src[m.end():]
    print("✅ argparse에 --weight-decay 추가 완료")

# ── 2. Namespace → optimizer 전달 부분 패치 ────────────────────────────────
# AdamW(...) 호출에 weight_decay=args.weight_decay 추가
# 패턴: AdamW( ... backbone_lr 또는 lr=... 가 있는 블록
ANCHOR_OPT = re.compile(
    r'(optim\.AdamW\([^)]*?)(weight_decay\s*=\s*[\d.e\-]+)',
    re.DOTALL,
)
if ANCHOR_OPT.search(src):
    # weight_decay 하드코딩 값을 args.weight_decay 로 교체
    src = ANCHOR_OPT.sub(
        lambda mo: mo.group(1) + "weight_decay=args.weight_decay",
        src,
    )
    print("✅ AdamW weight_decay 하드코딩 → args.weight_decay 교체 완료")
else:
    # weight_decay 자체가 없으면: lr= 뒤에 삽입
    ANCHOR_LR = re.compile(
        r'(optim\.AdamW\([^)]*?lr\s*=\s*[^,\)]+)',
        re.DOTALL,
    )
    m2 = ANCHOR_LR.search(src)
    if m2:
        src = src[: m2.end()] + ", weight_decay=args.weight_decay" + src[m2.end():]
        print("✅ AdamW 호출에 weight_decay=args.weight_decay 삽입 완료")
    else:
        print("⚠️  AdamW 호출을 찾지 못했습니다. optimizer 부분은 수동으로 확인해 주세요.")

TARGET.write_text(src, encoding="utf-8")
print(f"\n✅ 패치 완료: {TARGET}")
