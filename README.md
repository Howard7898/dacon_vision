# Visual Structure Stability Prediction — Dacon Monthly AI Competition

> **[월간 데이콘] 시각 기반 구조물 안정성 예측 AI 모델 개발**
> 평가 지표: Log Loss (낮을수록 우수) · Best Score: **0.041**

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Definition](#problem-definition)
3. [Qualitative Examples](#qualitative-examples)
4. [Dataset](#dataset)
5. [Solution Architecture](#solution-architecture)
6. [Project Structure](#project-structure)
7. [Pipeline Flow](#pipeline-flow)
8. [Key Implementation Details](#key-implementation-details)
9. [Training Configuration](#training-configuration)
10. [How to Run](#how-to-run)
11. [Results](#results)
12. [Dependencies](#dependencies)

---

## Overview

본 프로젝트는 **두 시점(정면 / 상단) 구조물 이미지**로부터 시뮬레이션 10초 이내의 붕괴 가능성(`unstable_prob`)과 안정 가능성(`stable_prob`)을 예측하는 AI 모델을 개발합니다.

핵심 도전 과제는 **도메인 갭(Domain Gap)** 입니다.

- **Train**: 광원·카메라가 고정된 실험실 환경 (1,000개)
- **Dev / Test**: 광원·카메라가 무작위로 변동하는 실제 평가 환경 (100 / 1,000개)

단순 이미지 분류를 넘어 **물리적 인과관계 추론** 능력이 요구되므로, DINOv2 시각 인코더에 물리 기반 기하학 특징 및 도메인 적응(GRL)을 결합한 멀티태스크 학습 방식으로 접근했습니다.

---

## Problem Definition

| 항목 | 내용 |
|---|---|
| **주제** | 시각 기반 구조물 안정성 예측 |
| **입력** | 구조물 정면(front) + 상단(top) 이미지 (2-view) |
| **출력** | `unstable_prob`, `stable_prob` (합산 = 1.0) |
| **레이블** | `stable`: 10초간 의미 있는 이동 없음 / `unstable`: 누적 이동 ≥ 1.5 cm 또는 붕괴 |
| **평가 지표** | Log Loss |
| **목표 범위** | 0.015 ≤ LogLoss ≤ 0.030 |

## Qualitative Examples

라벨 의미와 입력 형태를 한 번에 보여주기 위해, 대표 샘플 2개(`stable` / `unstable`)를 정성 예시로 배치했습니다. 각 샘플은 `front.png`, `top.png`, 그리고 10초 시뮬레이션 GIF로 구성됩니다.

<table>
  <thead>
    <tr>
      <th>Label</th>
      <th>Front View</th>
      <th>Top View</th>
      <th>Simulation GIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Stable</strong></td>
      <td><img src="assets/readme/qualitative/stable_front.png" width="180" alt="Stable sample front view"></td>
      <td><img src="assets/readme/qualitative/stable_top.png" width="180" alt="Stable sample top view"></td>
      <td><img src="assets/readme/qualitative/stable_simulation.gif" width="180" alt="Stable sample simulation GIF"></td>
    </tr>
    <tr>
      <td><strong>Unstable</strong></td>
      <td><img src="assets/readme/qualitative/unstable_front.png" width="180" alt="Unstable sample front view"></td>
      <td><img src="assets/readme/qualitative/unstable_top.png" width="180" alt="Unstable sample top view"></td>
      <td><img src="assets/readme/qualitative/unstable_simulation.gif" width="180" alt="Unstable sample simulation GIF"></td>
    </tr>
  </tbody>
</table>

> 위 예시는 설명용 정성 샘플이며, 실제 학습 데이터에는 동일한 쌍의 `front.png`, `top.png`, `simulation.mp4`가 제공됩니다.

---

## Dataset

```
open/
├── train.csv                    # 1,000개 샘플 (label 포함)
├── dev.csv                      # 100개 샘플 (label 포함, 랜덤 환경)
├── sample_submission.csv        # 1,000개 테스트 ID
├── train/
│   └── {TRAIN_XXXX}/
│       ├── front.png            # 정면 이미지
│       ├── top.png              # 상단 이미지
│       └── simulation.mp4       # 10초 물리 시뮬레이션 영상 (학습 전용)
├── dev/
│   └── {DEV_XXXX}/
│       ├── front.png
│       └── top.png
└── test/
    └── {TEST_XXXX}/
        ├── front.png
        └── top.png
```

> `simulation.mp4` (train 전용): 프레임 간 픽셀 차이 분석으로 `max_diff`, `mean_diff`, 이동 발생 시점(onset), 심각도(severity) 등의 **보조 지도 신호(auxiliary supervision)**를 추출하는 데 사용됩니다.

---

## Solution Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       DualViewPhysicsModel                       │
│                                                                  │
│  front.png ──► ViewEncoder ──►┐                                 │
│               (DINOv2 ViT-S)  ├──► TransformerFusion ──►        │
│  top.png ───► ViewEncoder ──►┘     (2-layer, 8-head)            │
│               (DINOv2 ViT-S)                                     │
│                                    image_feat  (1536-d)          │
│                                         │                        │
│  [Optional] GeometryFeatures (14-d) ──►┤                         │
│                                         ▼                        │
│                               ┌───────────────────┐             │
│                               │  classifier       │──► logit    │
│                               │  motion_reg       │──► motion   │
│                               │  onset_head       │──► onset    │
│                               │  severity_head    │──► severity │
│                               │  domain_head(GRL) │──► domain   │
│                               └───────────────────┘             │
│                                         │                        │
│                               TemperatureScaler (LBFGS)         │
└──────────────────────────────────────────────────────────────────┘
```

### ViewEncoder

| 구성 요소 | 내용 |
|---|---|
| **Backbone** | `DINOv2 ViT-S/14 with Registers` (`dinov2_vits14_reg`) |
| **특징 추출** | CLS 토큰(384-d) + Patch 토큰 평균(384-d) → concat → **768-d** |
| **Projection** | `Linear(768 → 512)` + GELU + Dropout(0.3) |
| **Pooling** | GeM (Generalized Mean Pooling, 학습 가능한 파라미터 `p`) |

### Transformer Fusion

- 두 ViewEncoder 출력에 학습 가능한 **View Embedding** 추가 후 Transformer 입력
- `TransformerEncoderLayer`: `d_model=512`, `nhead=8`, `ffn=2048`, `dropout=0.35`, Pre-LayerNorm
- `[front_feat ‖ top_feat ‖ fused_mean]` concat → **image_feat (1536-d)**

### Auxiliary Heads

| 헤드 | 출력 | 목적 |
|---|---|---|
| `classifier` | logit (1) | 메인 안정성 이진 분류 |
| `motion_reg` | 2-d | `max_diff` / `mean_diff` 회귀 |
| `onset_head` | 4-class | 이동 발생 시점 분류 |
| `severity_head` | 4-class | 이동 심각도 분류 |
| `domain_head` | 2-class via **GRL** | train↔dev 도메인 분류 (역전파로 도메인 불변 표현 학습) |

### Geometry Reasoning Module (`geometry_reasoning.py`)

전경 마스크 기반으로 물리적 구조 안정성과 직결되는 **14개 기하학 특징** 자동 추출:

| 뷰 | 특징명 |
|---|---|
| 상단 (top) | `top_area_frac`, `top_support_width_frac`, `top_support_height_frac`, `top_fill_ratio`, `top_centroid_dx`, `top_centroid_dy` |
| 정면 (front) | `front_height_frac`, `front_width_frac`, `front_slenderness`, `front_base_width_frac`, `front_top_width_frac`, `front_centroid_dx`, `front_tilt`, `front_top_heaviness` |

추출된 특징으로 `collapse_margin` 스코어를 계산해 보조 회귀 타깃으로도 활용합니다.

### Top-View Normalization (`checkerboard_rectification.py`)

- **문제**: Dev/Test 환경에서 카메라 각도가 무작위로 변동 → 상단 이미지 방향 불일치
- **해결**: 배경 선분 방향 추정 기반 회전 정규화
  - Scharr 엣지 → Canny → HoughLinesP → 주 방향각 추정 → 역회전 보정
  - 신뢰도 점수(conf) 계산 후 `conf < 0.20`이면 자동 스킵
- 결과를 `_angle_cache`에 캐시하여 동일 이미지 중복 연산 방지

---

## Project Structure

```
physics_solution/
│
├── full_physics_solution.py       # 핵심 파이프라인 (학습·추론·제출 생성)
│   ├── extract-motion             #   서브커맨드: 영상 → motion_targets.csv 추출
│   ├── train-design               #   서브커맨드: train→dev 단일 폴드 설계 검증
│   ├── cv-train                   #   서브커맨드: Pooled Grouped K-Fold 학습
│   ├── make-submission            #   서브커맨드: 저장된 모델로 제출 파일 생성
│   └── full-run                   #   서브커맨드: 위 전체 자동 실행 (권장)
│
├── run_colab_oneclick.py          # Google Colab 원클릭 실행기
│                                  # (Drive 마운트 · 패키지 설치 · zip 복사 ·
│                                  #  압축 해제 · 파이프라인 호출 · 결과 백업)
│
├── run_colab_oneclick.sh          # run_colab_oneclick.py 의 bash 래퍼
│
├── checkerboard_rectification.py  # 상단뷰 회전 정규화 모듈
│
├── geometry_reasoning.py          # 물리 기반 기하학 특징 추출 모듈
│
├── patch_weight_decay.py          # weight-decay 인자 하위 호환 패치 유틸
│
└── 0329_0.05_DINO_re (1).ipynb    # 최종 Colab 실행 노트북
```

---

## Pipeline Flow

```
[Step 1] extract-motion
  train/{id}/simulation.mp4
       │  프레임 간 픽셀 차이 분석 (64×64 리사이즈)
       ▼
  motion_targets.csv
  (max_diff_first · mean_diff_prev · severity_bucket · onset_bucket · soft_target)

[Step 2] build_geometry_clusters
  train + dev 이미지 중심 크롭 → 24×24 다운샘플 → StandardScaler → KMeans(k=32)
       ▼
  geometry_cluster (샘플별 클러스터 ID)

[Step 3] cv-train  ─  StratifiedGroupKFold(n_splits=5)
  pooled = train(1,000) + dev(100)
  stratify: label × source_domain
  group: geometry_cluster   ← 유사 구조물의 train/valid 분리 누수 방지
       ▼
  fold_1/ … fold_5/
      ├── best_model.pt       (val logloss 최솟값 체크포인트)
      ├── temperature.json    (Temperature Scaling 결과)
      └── oof_valid.csv       (OOF 예측값)

[Step 4] make-submission
  fold별 best_model.pt 로드
       │  TTA(tta_passes 회) 평균 → Temperature Scaling → fold 앙상블 평균
       ▼
  submission.csv  (id · unstable_prob · stable_prob)
```

---

## Key Implementation Details

### 손실 함수 (Multi-Task Learning)

```
Total Loss = BCE(main_classifier, soft_target)
           + 0.20 × SmoothL1(motion_reg)
           + 0.15 × CrossEntropy(onset_head)
           + 0.15 × CrossEntropy(severity_head)
           + 0.08 × SmoothL1(support_reg)     [--enable-geometry-reasoning 시]
           + 0.08 × SmoothL1(margin_reg)      [--enable-geometry-reasoning 시]
           + 0.05 × CrossEntropy(domain_head) [--use-domain-head 시]
```

- **Soft target**: 영상 모션 강도에 따라 hard label 0/1 을 연속값으로 보정
- **GRL lambda**: 에포크 진행에 따라 0 → 1 선형 증가 (도메인 헤드 점진 활성화)

### 옵티마이저 전략

| 파라미터 그룹 | LR |
|---|---|
| Backbone (`front/top_encoder.backbone`) | `backbone_lr` (1e-5) |
| 나머지 (헤드·퓨전·projection) | `lr` (2e-4) |

- **옵티마이저**: AdamW (`weight_decay=1e-4`)
- **스케줄러**: Linear Warmup (3 epoch) → Cosine Annealing

### 학습 안정화

| 기법 | 설정 |
|---|---|
| AMP (자동 혼합 정밀도) | `torch.amp.autocast` + `GradScaler` |
| Gradient Clipping | `max_norm = 1.0` |
| Seed 고정 | `random / numpy / torch / CUDA` (seed=42) |
| Best Checkpoint | Validation log loss 최소값 기준 저장 |
| Calibration | Temperature Scaling (LBFGS 200 iter) |

### Data Augmentation (Train only)

```python
CenterPhysicsCrop(view)                            # 구조물 중심 크롭
Resize(image_size, image_size)                     # 336×336
RandomApply(ColorJitter(b=0.35, c=0.35, ...), p=0.8)
RandomApply(GaussianBlur(k=5, σ=(0.1, 1.6)),  p=0.35)
RandomAdjustSharpness(0.8,                     p=0.2)
RandomPerspective(distortion=0.10,             p=0.35)
RandomAffine(degrees=7, translate=0.05, scale=(0.92, 1.08))
ToTensor()
RandomErasing(p=0.10, scale=(0.02, 0.08))
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Training Configuration

| 파라미터 | 값 | 비고 |
|---|---|---|
| `--backbone` | `dinov2_vits14_reg` | DINOv2 ViT-S/14 with Registers |
| `--image-size` | `336` | DINOv2 14px 배수 (294 → 336) |
| `--batch-size` | `16` | Colab GPU 기준 |
| `--epochs` | `30` | 수렴 여유 확보 (25 → 30) |
| `--num-folds` | `5` | Geometry-clustered Grouped KFold |
| `--backbone-lr` | `1e-5` | Backbone 차별화 학습률 |
| `--weight-decay` | `1e-4` | AdamW 정규화 |
| `--warmup-epochs` | `3` | Linear warmup |
| `--tta-passes` | `4` | Test Time Augmentation 횟수 |
| `--use-domain-head` | ✅ | GRL 기반 도메인 적응 |
| `--enable-geometry-reasoning` | ✅ | 물리 기하 특징 14개 추가 |
| `--refresh-motion` | ✅ | motion_targets.csv 재추출 |

---

## How to Run

### 환경 설치

```bash
pip install torch torchvision tqdm numpy pandas scikit-learn \
            opencv-python-headless Pillow xformers
```

### A. Google Colab (권장)

**노트북 파일**: `0329_0.05_DINO_re (1).ipynb`

Drive에 아래 두 파일을 업로드한 뒤 셀을 순서대로 실행합니다.

```
/MyDrive/daCon/
├── physics_solution_0328.zip   # 본 프로젝트 코드
└── open.zip                    # 데이콘 제공 데이터셋
```

| 셀 번호 | 역할 |
|---|---|
| 0 | `xformers` 설치 |
| 1 | Google Drive 마운트 |
| 2 | 경로 변수 설정 |
| 3 | zip 존재 확인 |
| 4 | 프로젝트 코드 압축 해제 및 로컬 배치 |
| 5 | 파일 목록 확인 |
| 6 | `--use-domain-head` 패치 (run_colab_oneclick.py 자동 수정) |
| 7 | 사전 점검 (키워드 grep) |
| **8** | **파이프라인 실행** ← 핵심 |
| 9 | `submission.csv` Drive 백업 |

### B. 로컬 / 서버 직접 실행

```bash
# Step 1: motion targets 추출 (최초 1회)
python full_physics_solution.py extract-motion \
    --data-root /path/to/open

# Step 2: 전체 파이프라인 실행 (학습 + 제출 파일 생성)
python full_physics_solution.py full-run \
    --data-root /path/to/open \
    --out-dir runs/final \
    --backbone dinov2_vits14_reg \
    --image-size 336 \
    --batch-size 16 \
    --epochs 30 \
    --num-folds 5 \
    --backbone-lr 1e-5 \
    --weight-decay 1e-4 \
    --tta-passes 4 \
    --use-domain-head \
    --enable-geometry-reasoning \
    --refresh-motion
```

### C. 원클릭 실행기 (Colab 자동화)

```bash
bash run_colab_oneclick.sh \
    --drive-zip-path "/content/drive/MyDrive/daCon/open.zip" \
    --local-root "/content/runtime" \
    --drive-output-root "/content/drive/MyDrive/daCon/outputs" \
    --image-size 336 \
    --batch-size 16 \
    --epochs 30 \
    --use-domain-head \
    --enable-geometry-reasoning \
    --refresh-motion
```

### 출력 파일 구조

```
runs/final/
├── motion_targets.csv        # 영상 기반 보조 지도 신호
├── fold_1/
│   ├── best_model.pt         # 검증 logloss 최솟값 체크포인트
│   ├── temperature.json      # Temperature Scaling 온도 T
│   └── oof_valid.csv         # OOF 예측값 (pred_raw, pred_cal)
├── fold_2/ … fold_5/
├── cv_summary.csv            # 폴드별 성능 요약
├── oof_all.csv               # 전체 OOF 취합본
└── submission.csv            # 최종 제출 파일 (id · unstable_prob · stable_prob)
```

---

## Results

| 지표 | 값 |
|---|---|
| **Dev OOF Log Loss** | **0.041** |
| 대회 목표 범위 | 0.015 ~ 0.030 |
| 백본 | DINOv2 ViT-S/14 with Registers |
| 앙상블 | 5-fold × TTA-4 |

---

## Dependencies

| 패키지 | 버전(권장) | 용도 |
|---|---|---|
| `torch` | ≥ 2.0 | 모델 학습 및 추론 |
| `torchvision` | ≥ 0.15 | 이미지 변환 및 사전학습 모델 |
| `xformers` | 최신 | DINOv2 attention 가속 |
| `numpy` | ≥ 1.24 | 수치 연산 |
| `pandas` | ≥ 2.0 | 데이터 처리 |
| `scikit-learn` | ≥ 1.3 | KMeans · KFold · LogLoss · Calibration |
| `opencv-python-headless` | ≥ 4.8 | 영상 처리 · 기하학 특징 추출 |
| `Pillow` | ≥ 10.0 | 이미지 로딩 및 변환 |
| `tqdm` | ≥ 4.66 | 학습 진행 표시 |

---

## License

본 코드는 Dacon 월간 데이콘 대회 참가 목적으로 작성되었습니다.
외부 데이터를 사용하지 않았으며, 대회 제공 데이터만 활용하였습니다.
