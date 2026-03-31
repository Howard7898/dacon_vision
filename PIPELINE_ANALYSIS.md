# Physics-aware Dual-View Solution — 전체 파이프라인 상세 분석

## 1. 프로젝트 개요

### 1.1 목적
물리 시뮬레이션 환경에서 쌓아올린 구조물의 **안정성(stable/unstable)을 이진 분류**하는 대회형 머신러닝 솔루션이다. 정면(front) 이미지와 상면(top) 이미지 두 시점을 동시에 활용하고, 시뮬레이션 동영상에서 추출한 물리적 움직임 정보를 보조 학습 신호로 사용한다.

### 1.2 평가 지표
- **주 지표**: Log Loss (Binary Cross-Entropy)
- 목표 범위: DEV OOF LOGLOSS `0.015 ~ 0.03`
- `0.05` 이상이면 설계 문제, `0.0001` 이하면 과적합 의심

### 1.3 제출 형식
```
id, unstable_prob, stable_prob
TEST_0001, 0.05, 0.95
TEST_0002, 0.88, 0.12
...
```
- `unstable_prob = prob_unstable`, `stable_prob = 1 - prob_unstable`

---

## 2. 데이터 구조

### 2.1 디렉토리 레이아웃
```
dataset_root/
├── train.csv              # id, label (stable/unstable)
├── dev.csv                # id, label (stable/unstable)
├── sample_submission.csv  # id (test set)
├── train/
│   └── TRAIN_XXXX/
│       ├── front.png      # 정면 촬영 이미지
│       ├── top.png        # 상면 촬영 이미지
│       └── simulation.mp4 # 물리 시뮬레이션 동영상
├── dev/
│   └── DEV_XXXX/
│       ├── front.png
│       └── top.png
└── test/
    └── TEST_XXXX/
        ├── front.png
        └── top.png
```

### 2.2 데이터 분할 체계
| 분할 | 용도 | 라벨 유무 | 동영상 유무 | source_domain 값 |
|------|------|-----------|-------------|-----------------|
| train | 학습 | O | O (simulation.mp4) | 0 |
| dev | 검증/학습(CV 시) | O | X | 1 |
| test | 최종 추론 | X | X | -1 |

### 2.3 데이터 자동 감지
- 환경변수 `PHYSICS_DATA_ROOT` 우선 확인
- 미설정 시 CWD, 스크립트 디렉토리, 상위 디렉토리, `~/Downloads` 순서로 탐색
- `open`이 이름에 포함된 하위 디렉토리를 우선 검색
- 필수 파일 6개 (`train.csv`, `dev.csv`, `sample_submission.csv`, `train/`, `dev/`, `test/`) 가 모두 존재해야 유효

---

## 3. 전체 파이프라인 흐름

### 3.1 파이프라인 단계별 흐름도
```
┌─────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE (full-run)                    │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ 1. extract   │───>│ 2. cv-train      │───>│ 3. make          │  │
│  │    -motion   │    │    (pooled        │    │    -submission    │  │
│  │              │    │     grouped CV)   │    │                  │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│       │                     │                       │              │
│       v                     v                       v              │
│  motion_targets.csv   fold_1~5/best_model.pt   submission.csv     │
│                       fold_1~5/temperature.json                    │
│                       oof_all.csv                                  │
│                       cv_summary.csv                               │
└─────────────────────────────────────────────────────────────────────┘

별도 설계 단계 (full-run 에 포함되지 않음):
┌──────────────┐
│ train-design │  train만으로 학습, dev만으로 검증 (아키텍처/증강 설계 고정용)
└──────────────┘
```

### 3.2 CLI 서브커맨드 정리
| 커맨드 | 역할 | 입력 | 출력 |
|--------|------|------|------|
| `extract-motion` | 시뮬레이션 영상에서 motion 피처 추출 | train/*.mp4 | motion_targets.csv |
| `train-design` | 설계 검증 (train→dev 홀드아웃) | train.csv, dev.csv, 이미지 | runs/design/best_model.pt, design_metrics.json |
| `cv-train` | 최종 학습 (train+dev pooled grouped 5-fold CV) | train.csv, dev.csv, 이미지, motion_targets.csv | runs/final/fold_*/best_model.pt |
| `make-submission` | 학습된 fold 모델로 test 추론 | test 이미지, fold 모델들 | submission.csv |
| `full-run` | extract-motion → cv-train → make-submission 순차 실행 | 전체 | submission.csv |

---

## 4. Stage 1: Motion Target 추출 (extract-motion)

### 4.1 목적
train set에만 있는 `simulation.mp4` 동영상을 분석하여, 구조물의 물리적 붕괴 패턴을 수치화한 **보조 학습 신호(soft target + auxiliary targets)** 를 생성한다.

### 4.2 처리 과정
```
simulation.mp4
    │
    ├── 첫 프레임 추출 → first_gray (64×64 grayscale)
    │
    ├── 모든 프레임 순회
    │   ├── 현재 프레임 vs 첫 프레임 → MAD(Mean Absolute Difference) → mad_to_first[]
    │   └── 현재 프레임 vs 이전 프레임 → MAD → mad_prev[]
    │
    └── 통계량 계산
        ├── max_diff_first  : 첫 프레임 대비 최대 차이 (붕괴 강도)
        ├── mean_diff_first : 첫 프레임 대비 평균 차이
        ├── max_diff_prev   : 연속 프레임 최대 차이 (순간 최대 변화)
        ├── mean_diff_prev  : 연속 프레임 평균 차이
        ├── first_move_thr2  : MAD > 2.0 처음 도달 프레임 (-1 = 미도달)
        ├── first_move_thr5  : MAD > 5.0 처음 도달 프레임
        └── first_move_thr10 : MAD > 10.0 처음 도달 프레임
```

### 4.3 파생 피처
| 피처 | 값 | 설명 |
|------|----|------|
| `severity_bucket` | 0=tiny, 1=small, 2=mid, 3=large | max_diff_first 기반 (2.0/5.0/10.0 경계) |
| `onset_bucket` | 0=very_early, 1=early, 2=late, 3=no_strong_hit | 움직임 시작 시점 (프레임 10/20 경계) |
| `soft_target` | float [0.02, 0.98] | motion 기반 라벨 보정값 |

### 4.4 Soft Target 생성 로직
```python
motion_score = 0.65 * min(max_diff_first / 10.0, 1.5) + 0.35 * min(mean_diff_prev / 0.15, 1.5)
# stable (label=0):  clip(0.02 + 0.10 * motion_score, 0.02, 0.15)
# unstable (label=1): clip(0.65 + 0.30 * motion_score, 0.65, 0.98)
```
- 목적: hard label 대신 motion 강도를 반영한 확률로 BCE 학습 → logloss 캘리브레이션 개선
- stable인데 약간 흔들린 경우 → 0.02보다 약간 높은 soft target
- unstable인데 약하게 무너진 경우 → 0.98보다 낮은 soft target

### 4.5 출력
- `motion_targets.csv` : id, label, label_int, frames, fps, max_diff_first, mean_diff_first, max_diff_prev, mean_diff_prev, first_move_thr2, first_move_thr5, first_move_thr10, severity_bucket, onset_bucket, soft_target

---

## 5. Stage 2: 이미지 전처리 파이프라인

### 5.1 Checkerboard Top-View Rotation Normalization (기본 ON)
```
top.png 원본 RGB
    │
    ├── 전경(foreground) 마스크 추정
    │   ├── HSV 변환 → saturation, value 채널
    │   ├── Grayscale 변환
    │   ├── 적응적 임계값 (percentile 기반: sat>p60 | gray<p45 | val<p35)
    │   ├── Morphology Open(3×3) + Close(5×5)
    │   └── Connected Components → 최대 영역 선택
    │
    ├── 배경(background) 마스크 = 1 - 전경
    │   └── 배경이 너무 작으면 (< 3% 픽셀) → ring mask(테두리 10%) 사용
    │
    ├── 에지 검출 (배경 영역에서만)
    │   ├── Scharr gradient (gx, gy) → magnitude
    │   ├── Canny(40, 120)
    │   └── 배경 마스크와 AND
    │
    ├── Hough 직선 검출 (HoughLinesP)
    │   ├── threshold=30, minLineLength=24, maxLineGap=6
    │   └── 최대 400개 직선
    │
    ├── 체커보드 격자 각도 추정
    │   ├── 직선 각도 mod 90° → 원형 평균 (circular mean)
    │   ├── 180-bin 히스토그램 (길이 가중)
    │   └── rot_mod90 = arctan2(cy, cx) * (90/360) mod 90
    │
    ├── 신뢰도(confidence) 계산
    │   ├── peak_ratio_score: 1차 피크 / (1차 + 2차) = 0.35 가중
    │   ├── line_score: min(1, line_count / 60) = 0.25 가중
    │   ├── spread_score: 1 - spread/20 = 0.20 가중
    │   └── ortho_score: 직교 피크 / 1차 피크 = 0.20 가중
    │
    └── 회전 적용 조건
        ├── rot_ok = (line_count >= 10) AND (confidence >= 0.20)
        ├── 통과 시: cv2.warpAffine(angle_deg, pad_value=128)
        └── 미통과 시: 원본 그대로 사용, 결과 캐싱
```

**실험 결과 요약:**
| 전처리 조합 | Dev Logloss | 비고 |
|-------------|-------------|------|
| front_none + top_none | 0.3332 | 기준선 |
| front_none + top_rot | 0.3187 | 기본 채택 |
| front_none + top_rot_persp | 0.3389 | perspective는 오히려 악화 |

- top_rot 성공률: train 100%, dev 100%, test 99.4%
- top_rot_persp 성공률: train 1.8%, dev 23%, test 20.4% (너무 불안정)

### 5.2 CenterPhysicsCrop (물리 중심 크롭)
배경 누수(background leakage)를 줄이기 위해 구조물 중심 영역만 잘라낸다.

| 뷰 | 크롭 박스 (비율) | 실제 의미 |
|----|-----------------|-----------|
| front | (0.25w, 0.20h) ~ (0.75w, 0.88h) | 좌우 25% + 상단 20% + 하단 12% 제거 → 중앙 50%×68% |
| top | (0.29w, 0.29h) ~ (0.71w, 0.71h) | 상하좌우 29% 제거 → 중앙 42%×42% |

### 5.3 학습 시 Augmentation (Train Transform)
```
CenterPhysicsCrop
    → Resize(image_size × image_size)      # 기본 320×320 (macOS에서는 288×288)
    → ColorJitter(p=0.8)                   # brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04
    → GaussianBlur(p=0.35)                 # kernel=5, sigma=(0.1, 1.6)
    → RandomAdjustSharpness(p=0.2)         # factor=0.8
    → RandomPerspective(p=0.35)            # distortion=0.10
    → RandomAffine                         # degrees=±7, translate=(5%, 5%), scale=(0.92, 1.08)
    → ToTensor
    → RandomErasing(p=0.10)               # scale=(2%, 8%), value="random"
    → Normalize(ImageNet mean/std)
```

### 5.4 검증/추론 시 Transform (Valid Transform)
```
CenterPhysicsCrop
    → Resize(image_size × image_size)
    → ToTensor
    → Normalize(ImageNet mean/std)
```

---

## 6. Stage 3: Geometry Feature 추출 (선택적, 기본 OFF)

### 6.1 개요
front/top 이미지에서 구조물의 기하학적 특성 14개를 추출한다. `--enable-geometry-reasoning` 플래그로 활성화.

### 6.2 전경 마스크 추정
체커보드 정규화와 동일한 방식: HSV + Grayscale 적응 임계 → Morphology → Connected Components

### 6.3 추출 피처 목록 (14개)
| # | 피처명 | 설명 | 뷰 |
|---|--------|------|-----|
| 1 | `top_area_frac` | 전경 픽셀 비율 (전체 이미지 대비) | top |
| 2 | `top_support_width_frac` | 전경 바운딩박스 가로 비율 | top |
| 3 | `top_support_height_frac` | 전경 바운딩박스 세로 비율 | top |
| 4 | `top_fill_ratio` | 전경 면적 / 바운딩박스 면적 (밀도) | top |
| 5 | `top_centroid_dx` | 전경 무게중심 X 편차 [-1, 1] (0=정중앙) | top |
| 6 | `top_centroid_dy` | 전경 무게중심 Y 편차 [-1, 1] | top |
| 7 | `front_height_frac` | 전경 바운딩박스 높이 비율 | front |
| 8 | `front_width_frac` | 전경 바운딩박스 너비 비율 | front |
| 9 | `front_slenderness` | 높이/너비 (세장비, 높을수록 불안정) | front |
| 10 | `front_base_width_frac` | 하단 20% 밴드의 전경 너비 비율 | front |
| 11 | `front_top_width_frac` | 상단 25% 밴드의 전경 너비 비율 | front |
| 12 | `front_centroid_dx` | 전경 무게중심 X 편차 | front |
| 13 | `front_tilt` | 상단 중심 - 하단 중심 (기울기 프록시) | front |
| 14 | `front_top_heaviness` | 상반부 픽셀 / 전체 픽셀 (상부 비대 정도) | front |

### 6.4 Collapse Margin (붕괴 여유도)
기하 피처의 가중합으로 계산하는 붕괴 안정성 프록시:
```python
raw = (
    +1.20 × top_support_width_frac     # 넓은 지지면 → 안정
    +0.90 × front_base_width_frac      # 넓은 밑면 → 안정
    +0.50 × top_fill_ratio             # 빈 공간 적음 → 안정
    -0.75 × |top_centroid_dx|          # 중심 치우침 → 불안정
    -0.55 × |front_tilt|              # 기울어짐 → 불안정
    -0.20 × front_slenderness          # 가늘고 높음 → 불안정
    -0.25 × front_top_heaviness        # 위가 무거움 → 불안정
)
collapse_margin = clip(0.5 + 0.35 × raw, 0, 1)
```

### 6.5 Geometry Feature Cache
- 샘플 ID 기반 딕셔너리 캐시
- 동일 에폭 내 같은 샘플 재계산 방지

---

## 7. Stage 4: Geometry Clustering (Fold 구성)

### 7.1 목적
비슷한 구조물끼리 같은 fold에 묶어서 **정보 누수(leakage)를 방지**한다. 같은 구조 유형이 train fold와 valid fold에 동시에 들어가지 않도록 한다.

### 7.2 과정
```
train + dev 전체 샘플
    │
    ├── 각 샘플에서
    │   ├── front.png → 중앙 크롭 (80,96)~(320,288) → 24×24 grayscale → flatten (576차원)
    │   └── top.png   → 중앙 크롭 (112,112)~(272,272) → 24×24 grayscale → flatten (576차원)
    │
    ├── concat → 1152차원 벡터
    ├── StandardScaler (zero mean, unit variance)
    └── KMeans(n_clusters=32, n_init=20, random_state=42) → geometry_cluster 라벨
```

### 7.3 Fold 분할
```
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
├── stratify: label_int + source_domain 조합 (라벨과 출처 동시 균형)
└── group: geometry_cluster (같은 클러스터는 같은 fold에)
```

---

## 8. Stage 5: 모델 아키텍처 (DualViewPhysicsModel)

### 8.1 전체 구조
```
입력:
  front.png (3×320×320)  ─────────────────────────────────────────┐
  top.png   (3×320×320)  ─────────────────────────────────────┐   │
  geom_feat (14차원, OFF 시 미사용)  ────────────────────┐     │   │
                                                         │     │   │
                                                         v     v   v
                                                  [Geometry] [Top] [Front]
                                                  [Proj   ] [Enc] [Encoder]
                                                         │     │   │
                                                         │     v   v
                                                         │   t_emb f_emb
                                                         │  (512d) (512d)
                                                         │     │   │
                                                         │     │   ├── + view_embed[0] (학습 가능)
                                                         │     │   │
                                                         │     ├── + view_embed[1] (학습 가능)
                                                         │     │   │
                                                         │     v   v
                                                         │  ┌──────────┐
                                                         │  │ Transformer │
                                                         │  │ Encoder    │
                                                         │  │ (2 layers) │
                                                         │  └──────────┘
                                                         │       │
                                                         │       v
                                                         │   fused_mean (512d)
                                                         │       │
                                                         │   cat(f_emb, t_emb, fused_mean) = image_feat (1536d)
                                                         │       │
                                                         │       ├── [Geometry ON] cat(image_feat, geom_emb) = feat (1792d)
                                                         v       ├── [Geometry OFF] feat = image_feat (1536d)
                                                                 │
                                              ┌──────────────────┼──────────────────────────────┐
                                              │                  │                              │
                                              v                  v                              v
                                        ┌──────────┐    ┌──────────────┐              ┌──────────────┐
                                        │Classifier│    │ Motion Reg   │              │ Onset Head   │
                                        │→ logit(1)│    │→ (2) values  │              │→ (4) classes │
                                        └──────────┘    └──────────────┘              └──────────────┘
                                                                                              │
                                                                                      ┌──────────────┐
                                                                                      │Severity Head │
                                                                                      │→ (4) classes │
                                                                                      └──────────────┘
                                        [Geometry ON 추가]:
                                        ┌──────────────┐  ┌──────────────┐
                                        │Support Reg   │  │Margin Reg    │
                                        │→ (2) values  │  │→ (1) value   │
                                        └──────────────┘  └──────────────┘

                                        [Domain Head ON 추가]:
                                        ┌──────────────────────────┐
                                        │Domain Head (GRL 적용)     │
                                        │→ (2) classes (train/dev) │
                                        └──────────────────────────┘
```

### 8.2 ViewEncoder (Front/Top 각각 독립)
```
Backbone (DINOv2 ViT-S/14, pretrained)
    │ 출력: cls token (B × 384)
    v
Linear Projection (384 → 512)
    → GELU
    → Dropout(0.15)
    │ 출력: (B × 512)
```

### 8.3 지원 Backbone 목록
| Backbone | Feature Dim | ImageNet Weights |
|----------|-------------|------------------|
| `dinov2_vits14` (기본) | 384 | Meta DINOv2 ViT-S/14 pretrained |
| `efficientnet_v2_s` | 1280 | EfficientNet_V2_S_Weights.DEFAULT |
| `resnet50` | 2048 | ResNet50_Weights.DEFAULT |
| `convnext_tiny` | 768 | ConvNeXt_Tiny_Weights.DEFAULT |
| `convnext_small` | 768 | ConvNeXt_Small_Weights.DEFAULT |

### 8.4 View Fusion (Transformer Encoder)
```
tokens = stack([f_emb + view_embed[0], t_emb + view_embed[1]])  → (B × 2 × 512)
    │
    v
TransformerEncoder
    ├── num_layers: 2
    ├── d_model: 512
    ├── nhead: 8
    ├── dim_feedforward: 2048 (512 × 4)
    ├── dropout: 0.10
    ├── activation: GELU
    ├── norm_first: True (Pre-LN)
    └── batch_first: True
    │
    v
LayerNorm → mean(dim=1) → fused_mean (B × 512)
```

### 8.5 최종 특징 벡터
```
image_feat = cat(f_emb, t_emb, fused_mean) → (B × 1536)

[Geometry OFF]: feat = image_feat (1536d)
[Geometry ON]:
    geom_emb = LayerNorm(14d) → Linear(14→256) → GELU → Dropout(0.10)
    feat = cat(image_feat, geom_emb) → (B × 1792)
```

### 8.6 출력 헤드 상세
| 헤드 | 구조 | 출력 차원 | 활성화 조건 |
|------|------|-----------|-------------|
| **Classifier** | Linear(feat→512) → GELU → Dropout(0.20) → Linear(512→1) | 1 (logit) | 항상 |
| **Motion Reg** | Linear(feat→256) → GELU → Linear(256→2) | 2 (max_diff, mean_diff) | 항상 |
| **Onset Head** | Linear(feat→256) → GELU → Linear(256→4) | 4 (onset bucket class) | 항상 |
| **Severity Head** | Linear(feat→256) → GELU → Linear(256→4) | 4 (severity bucket class) | 항상 |
| **Support Reg** | Linear(image_feat→256) → GELU → Linear(256→2) | 2 (support width, area) | Geometry ON |
| **Margin Reg** | Linear(image_feat→256) → GELU → Linear(256→1) | 1 (collapse margin) | Geometry ON |
| **Domain Head** | GRL → Linear(feat→256) → GELU → Linear(256→2) | 2 (train vs dev) | Domain ON |

### 8.7 Gradient Reversal Layer (GRL)
- Domain Head 앞에 위치
- forward: 항등 함수 (입력 그대로 통과)
- backward: gradient에 `-λ`를 곱함 (gradient 역전)
- λ = min(epoch / (epochs-1), 1.0) : 학습 초기에는 약하게, 후반으로 갈수록 강하게 역전
- 목적: train/dev 도메인 구분 불가능한 특징 학습 (domain-invariant feature)

---

## 9. Stage 6: 손실 함수 (Multi-Task Loss)

### 9.1 전체 손실 구성
```
Total Loss = loss_main
           + 0.20 × loss_motion
           + 0.15 × loss_onset
           + 0.15 × loss_severity
           + 0.08 × loss_support      (Geometry ON일 때만)
           + 0.08 × loss_margin       (Geometry ON일 때만)
           + 0.05 × loss_domain       (Domain Head ON일 때만)
```

### 9.2 각 손실 상세
| 손실 | 함수 | Target | 유효 조건 |
|------|------|--------|-----------|
| `loss_main` | `BCEWithLogitsLoss` | soft_target (있으면) 또는 hard label | label >= 0 |
| `loss_motion` | `SmoothL1Loss` | [max_diff_first/10, mean_diff_prev/0.15] clamp(0,2) | motion 값이 NaN이 아닐 때 |
| `loss_onset` | `CrossEntropyLoss` | onset_bucket (0~3) | onset >= 0 |
| `loss_severity` | `CrossEntropyLoss` | severity_bucket (0~3) | severity >= 0 |
| `loss_support` | `SmoothL1Loss` | [top_support_width_frac, top_area_frac] | Geometry ON |
| `loss_margin` | `SmoothL1Loss` | collapse_margin | Geometry ON |
| `loss_domain` | `CrossEntropyLoss` | source_domain (0=train, 1=dev) | Domain ON, domain >= 0 |

### 9.3 Soft Target 우선순위
```python
target = soft_target if not NaN else hard_label
target = 0 if label < 0 (unlabeled)
```

---

## 10. Stage 7: 학습 루프

### 10.1 Optimizer & Scheduler
| 항목 | 설정 |
|------|------|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| AMP | CUDA일 때 자동 활성화 (GradScaler) |
| Gradient Clipping | max_norm=1.0 |
| Seed | 42 (모든 RNG 고정) |

### 10.2 학습 흐름 (fit_one_fold)
```
for epoch in range(12):
    # 1. Train
    for batch in train_loader:
        forward → compute_losses → backward → clip_grad → optimizer.step
    scheduler.step()

    # 2. Validate (no augmentation, tta=1)
    valid_pred = predict_loader(model, valid_loader, tta=1)
    logloss, auc = evaluate(valid_pred, valid_labels)
    if logloss < best → save best_state

# 3. Best model 복원
model.load_state_dict(best_state)

# 4. Temperature Scaling (validation logits 기반)
calibrator = TemperatureScaler()
temperature = calibrator.fit(valid_logits, valid_labels)  # LBFGS, max_iter=200

# 5. 저장
torch.save(best_state, fold_dir/best_model.pt)
save temperature.json: {temperature, valid_logloss, valid_auc}
save oof_valid.csv: {id, label_int, source_domain, logit, pred_raw, pred_cal}
```

### 10.3 Temperature Scaling
- 학습된 모델의 logit에 학습 가능한 스칼라 T로 나눔: `calibrated_logit = logit / T`
- LBFGS optimizer로 validation BCE를 최소화하여 T 추정
- T > 1 이면 과신(overconfident) 보정, T < 1 이면 과소신뢰 보정
- T는 0.1 이상으로 클램핑

---

## 11. Stage 8: 워크플로우 분기

### 11.1 Design Holdout (train-design)
```
데이터: train만 학습, dev만 검증 (1 fold)
목적: 아키텍처/증강/손실 설계 검증
출력: design_metrics.json, dev_logloss_report.json
판단 기준:
  - DEV LOGLOSS 0.015~0.03 → 목표 밴드 (OK)
  - > 0.05 → 설계 문제 (재설계 필요)
  - < 0.0001 → 과적합 의심
```

### 11.2 Pooled Grouped CV (cv-train)
```
데이터: train + dev 합친 후 5-fold CV
분할: StratifiedGroupKFold (label+domain 층화, geometry_cluster 그룹)
각 fold: fit_one_fold → best_model.pt + temperature.json + oof_valid.csv
종합: cv_summary.csv, oof_all.csv, dev_oof_logloss_report.json
```

### 11.3 Full Run (full-run)
```python
1. motion_targets.csv가 없거나 --refresh-motion → extract_motion_targets()
2. run_pooled_grouped_cv(cfg)
3. make_submission(cfg)
```

---

## 12. Stage 9: 추론 및 앙상블 (make-submission)

### 12.1 Fold 품질 필터링
각 fold의 `temperature.json`을 읽어서:
- `temperature <= 0` 또는 `valid_logloss > 2.0` → 해당 fold **제외**
- 모든 fold가 필터링되면 → 필터 없이 전체 사용 (fallback)

### 12.2 추론 과정
```
for each valid fold:
    model = DualViewPhysicsModel(pretrained=False)
    model.load_state_dict(fold_dir/best_model.pt)

    for batch in test_loader:
        probs_accum = 0
        for tta_pass in range(4):    # TTA 4회
            prob = sigmoid(model.forward(front, top, geom_feat))
            probs_accum += prob
        pred = probs_accum / 4

    # Temperature Scaling 적용
    logit = log(pred / (1 - pred))          # prob → logit 역변환
    calibrated_pred = sigmoid(logit / T)    # T로 나눠서 재캘리브레이션
```

### 12.3 Fold Ensemble
```python
prob_unstable = mean(pred_fold_1, pred_fold_2, ..., pred_fold_5)  # 단순 평균
unstable_prob = prob_unstable
stable_prob = 1.0 - prob_unstable
```

### 12.4 TTA (Test Time Augmentation)
- 4 passes (기본값)
- 학습 시 사용하는 train transform (랜덤 증강 포함)을 추론에도 적용
- 각 pass의 예측 확률을 평균

---

## 13. DualViewDataset 상세

### 13.1 __getitem__ 반환 딕셔너리
| 키 | 타입 | 설명 |
|----|------|------|
| `id` | str | 샘플 ID (e.g., "TRAIN_0001") |
| `front` | Tensor (3×H×W) | 전처리된 정면 이미지 |
| `top` | Tensor (3×H×W) | 전처리된 상면 이미지 (체커보드 정규화 포함) |
| `geom_feat` | Tensor (14,) | 기하 피처 벡터 |
| `support_target` | Tensor (2,) | [top_support_width_frac, top_area_frac] |
| `collapse_margin` | Tensor (1,) | 붕괴 여유도 스칼라 |
| `label` | Tensor (1,) | 0.0=stable, 1.0=unstable, -1.0=unlabeled |
| `max_diff_first` | Tensor (1,) | motion 최대 차이 (NaN 가능) |
| `mean_diff_prev` | Tensor (1,) | motion 연속 평균 차이 (NaN 가능) |
| `soft_target` | Tensor (1,) | motion 기반 soft label (NaN 가능) |
| `severity_bucket` | Tensor (1,) long | 0~3 또는 -1 |
| `onset_bucket` | Tensor (1,) long | 0~3 또는 -1 |
| `source_domain` | Tensor (1,) long | 0=train, 1=dev, -1=test |

### 13.2 DataLoader 설정
| 설정 | Train | Valid/Test |
|------|-------|------------|
| batch_size | cfg.batch_size (기본 8, macOS 4) | max(cfg.batch_size, 8) |
| shuffle | True | False |
| num_workers | 0 (macOS), 4 이하 (Linux) | 동일 |
| pin_memory | True (CUDA), False (기타) | 동일 |

---

## 14. 실행 환경 구성

### 14.1 디바이스 자동 감지
```python
우선순위: CUDA → MPS (Apple Silicon) → CPU
추가 최적화:
  - float32_matmul_precision = "high"
  - cudnn.benchmark = True (CUDA)
```

### 14.2 macOS 실행 (.command 파일)
```
full_pipeline.command (Finder 더블클릭 가능)
├── PHYSICS_DATA_ROOT 환경변수 확인
├── .venv 없으면 → bootstrap_mac_env.sh 실행
│   ├── python3 -m venv .venv
│   └── pip install -r requirements-mac.txt
├── 패키지 검증 (cv2, numpy, pandas, sklearn, torch, torchvision)
├── 실패 시 → bootstrap_mac_env.sh 재실행
└── python full_physics_solution.py full-run \
      --image-size 294 --batch-size 4 --num-workers 0 --tta-passes 4
```

| 스크립트 | 실행 내용 |
|----------|----------|
| `train.command` | extract-motion → cv-train |
| `infer.command` | make-submission |
| `full_pipeline.command` | full-run (전체) |

### 14.3 Colab 실행 (run_colab_oneclick.py)
```
1. Google Drive 마운트
2. Python 패키지 확인/설치 (pip install -q)
3. GPU 정보 출력 (nvidia-smi)
4. Drive zip → Colab 로컬 디스크 복사 (I/O 속도 개선)
5. zip 해제 → dataset_root 자동 탐색
6. full_physics_solution.py full-run 실행
7. 결과물 (submission.csv, 모델, 로그) → Drive에 복사
   └── run_{timestamp}/ 디렉토리에 아카이브
```

**Colab 기본 설정:**
| 항목 | 값 |
|------|----|
| batch_size | 12 |
| num_workers | 2 |
| image_size | 320 |
| Drive zip 기본 경로 | `/content/drive/MyDrive/open (7).zip` |
| 로컬 런타임 루트 | `/content/physics_solution_runtime` |
| Drive 출력 루트 | `/content/drive/MyDrive/physics_solution_outputs` |

### 14.4 의존성 패키지
```
numpy, pandas, Pillow, scikit-learn, opencv-python, torch, torchvision, tqdm
```

---

## 15. 파일 구조 및 역할 정리

| 파일 | 역할 | 줄 수 |
|------|------|-------|
| `full_physics_solution.py` | **핵심 코드 전체** — motion 추출, 데이터셋, 모델, 학습, 추론, CLI | ~1420 |
| `checkerboard_rectification.py` | top-view 체커보드 회전 정규화 모듈 | ~184 |
| `geometry_reasoning.py` | 기하 피처 14개 추출 + collapse margin 계산 | ~185 |
| `run_colab_oneclick.py` | Colab 원클릭 실행 래퍼 | ~291 |
| `bootstrap_mac_env.sh` | macOS venv 생성 + 패키지 설치 | ~21 |
| `full_pipeline.command` | macOS 원클릭: full-run 실행 | ~47 |
| `train.command` | macOS 원클릭: motion 추출 + CV 학습 | ~48 |
| `infer.command` | macOS 원클릭: submission 생성 | ~45 |
| `make_colab_bundle.sh` | 코드만 담긴 경량 zip 번들 생성 | - |
| `PhysicsSolution_Colab_OneClick.ipynb` | Colab 노트북 진입점 | - |
| `requirements-mac.txt` | macOS pip 의존성 목록 | 8 |
| `checkerboard_eval_summary.md` | 체커보드 정규화 실험 결과 요약 | - |

---

## 16. 하이퍼파라미터 전체 정리

### 16.1 모델 관련
| 파라미터 | 기본값 | CLI 플래그 |
|----------|--------|-----------|
| backbone | dinov2_vits14 | `--backbone` |
| pretrained | False (CLI에서 `--pretrained`로 활성화) | `--pretrained` |
| emb_dim | 512 | (코드 고정) |
| image_size | 294 | `--image-size` |
| geometry_reasoning | False | `--enable-geometry-reasoning` |
| domain_head | False | `--use-domain-head` |
| checkerboard_top_normalize | True | `--no-checkerboard-top-normalize` |

### 16.2 학습 관련
| 파라미터 | 기본값 | CLI 플래그 |
|----------|--------|-----------|
| batch_size | 8 (macOS: 4) | `--batch-size` |
| epochs | 12 | `--epochs` |
| lr | 2e-4 | `--lr` |
| weight_decay | 1e-4 | `--weight-decay` |
| num_folds | 5 | `--num-folds` |
| seed | 42 | `--seed` |
| grad_clip | 1.0 | (코드 고정) |
| use_amp | True (CUDA만) | `--no-amp` |
| num_workers | 0 (macOS), min(4, cpu//2) (Linux) | `--num-workers` |

### 16.3 보조 손실 가중치
| 파라미터 | 가중치 |
|----------|--------|
| aux_motion_weight | 0.20 |
| aux_onset_weight | 0.15 |
| aux_severity_weight | 0.15 |
| aux_support_weight | 0.08 (Geometry ON) |
| aux_margin_weight | 0.08 (Geometry ON) |
| domain_weight | 0.05 (Domain ON) |

### 16.4 추론 관련
| 파라미터 | 기본값 | CLI 플래그 |
|----------|--------|-----------|
| tta_passes | 4 | `--tta-passes` |

### 16.5 Motion 추출 관련 (코드 고정)
| 파라미터 | 값 |
|----------|---|
| resize | 64×64 |
| thr_low | 2.0 |
| thr_mid | 5.0 |
| thr_high | 10.0 |

### 16.6 Clustering 관련 (코드 고정)
| 파라미터 | 값 |
|----------|---|
| n_clusters | 32 |
| front_crop | (96, 80, 288, 320) |
| top_crop | (112, 112, 272, 272) |
| downsample | 24×24 |
| n_init | 20 |

### 16.7 Checkerboard 관련 (코드 고정)
| 파라미터 | 값 |
|----------|---|
| ring_ratio | 0.10 |
| rot_line_min | 10 |
| rot_conf_min | 0.20 |
| pad_value | 128 |
| Hough threshold | 30 |
| Hough minLineLength | 24 |
| Hough maxLineGap | 6 |

---

## 17. 출력 디렉토리 구조

```
runs/final/
├── fold_1/
│   ├── best_model.pt            # 최적 epoch 모델 가중치
│   ├── temperature.json         # {temperature, valid_logloss, valid_auc}
│   └── oof_valid.csv            # Out-of-fold 검증 예측
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
├── cv_summary.csv               # fold별 메트릭 요약
├── oof_all.csv                  # 전체 OOF 예측 통합
├── dev_oof_logloss_report.json  # DEV 샘플만의 OOF logloss 평가
└── submission.csv               # 최종 제출 파일
```

---

## 18. 핵심 설계 결정 요약

| 설계 결정 | 채택 | 이유 |
|-----------|------|------|
| Dual-view (front + top) | O | 두 시점이 보완적 정보 제공 |
| Soft target from motion | O | 경계 샘플의 과신 방지, logloss 개선 |
| Checkerboard top rotation | O | dev logloss 0.3332→0.3187 개선, 성공률 99%+ |
| Full perspective rectification | X | 성공률 1.8~23%로 불안정 |
| Geometry reasoning | 선택적(OFF) | 검증 미완료, 안전하게 별도 실험 분리 |
| Domain adversarial (GRL) | 선택적(OFF) | train/dev 도메인 차이 보정 실험용 |
| Temperature scaling | O | 모델 과신/과소신뢰 보정 |
| Geometry-clustered grouping | O | 비슷한 구조물의 fold 간 누수 방지 |
| Multi-task auxiliary losses | O | motion/onset/severity 보조 학습으로 표현력 강화 |
| Design → CV 2단계 워크플로우 | O | 설계 결정과 최종 학습 분리 (dev 오염 방지) |
