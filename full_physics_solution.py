# =============================================================================
# full_physics_solution.py
# -----------------------------------------------------------------------------
# 작성자  : (공개 생략)
# 인코딩  : UTF-8
# Python  : 3.10+
# 설명    : 물리 구조물 안정성 예측을 위한 이중 뷰(front/top) 학습 파이프라인
#           (동영상 모션 추출 → 기하 클러스터 폴드 → 체커보드 정규화 →
#            듀얼뷰 모델 학습 → 온도 스케일링 → 폴드 앙상블 → 제출파일 생성)
# =============================================================================
"""
Physics-aware dual-view training pipeline for structure stability prediction.

    What this script implements:
    1. Video-derived motion target extraction from train/simulation.mp4
    2. Geometry-clustered fold construction to reduce leakage
    3. Checkerboard-guided top-view rotation normalization
    4. Dual-view static student model (front/top)
    5. Auxiliary supervision from video motion targets
    6. Temperature scaling for logloss calibration
    7. Fold training + OOF + test ensembling

What this script intentionally does NOT pretend to do:
- fully solved checkerboard homography rectification
- fully validated end-to-end leaderboard performance inside this environment

Recommended workflow:
A. Freeze architecture using train -> dev holdout only
   python full_physics_solution.py extract-motion --data-root /path/to/open
   python full_physics_solution.py train-design --data-root /path/to/open --out-dir runs/design

B. After architecture freeze, use pooled train+dev grouped CV
   python full_physics_solution.py cv-train --data-root /path/to/open --out-dir runs/final
   python full_physics_solution.py make-submission --data-root /path/to/open --run-dir runs/final
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import argparse
import dataclasses
import gc
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
from geometry_reasoning import GEOMETRY_FEATURE_NAMES, GeometryFeatureCache


# ============================================================
# 전역 상수 및 기본값
# ============================================================

REQUIRED_DATASET_CHILDREN = (
    "train.csv",
    "dev.csv",
    "sample_submission.csv",
    "train",
    "dev",
    "test",
)  # 유효한 데이터셋 루트가 포함해야 하는 파일/디렉토리

DEV_LOGLOSS_TARGET_LOW = 0.015     # dev OOF logloss 목표 하한
DEV_LOGLOSS_TARGET_HIGH = 0.03     # dev OOF logloss 목표 상한
DEV_LOGLOSS_OVERFIT_LOW = 0.0001   # 이 값 미만이면 과적합 경고
DEV_LOGLOSS_DESIGN_BAD_HIGH = 0.05 # 이 값 초과이면 설계 문제 경고

DEFAULT_BACKBONE = "dinov2_vits14_reg"   # 기본 백본 이름
DEFAULT_IMAGE_SIZE = 294                  # 기본 입력 이미지 크기 (픽셀)

SUPPORTED_BACKBONES = [
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "efficientnet_v2_s",
    "resnet50",
    "convnext_tiny",
    "convnext_small",
]  # 지원하는 백본 목록


# ============================================================
# 유틸리티 함수
# ============================================================

def set_seed(seed: int = 42) -> None:
    """모든 난수 생성기의 시드를 고정한다.

    Args:
        seed: 고정할 시드값 (기본값 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """디렉토리가 존재하지 않으면 생성한다.

    Args:
        path: 생성할 디렉토리 경로.

    Returns:
        생성(또는 기존) 경로의 Path 객체.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_project_dir() -> Path:
    """현재 스크립트가 위치한 디렉토리를 반환한다.

    Returns:
        스크립트 디렉토리 Path 객체.
    """
    return Path(__file__).resolve().parent


def default_runs_dir(name: str) -> Path:
    """프로젝트 디렉토리 내 runs/<name> 경로를 반환한다.

    Args:
        name: 실행 구분 이름 (예: 'design', 'final').

    Returns:
        runs/<name> Path 객체.
    """
    return default_project_dir() / "runs" / name


def default_num_workers() -> int:
    """플랫폼에 맞는 DataLoader 워커 수를 반환한다.

    Returns:
        워커 수 (macOS는 0, 그 외는 CPU 수의 절반, 최대 4).
    """
    if sys.platform == "darwin":
        return 0  # macOS에서 멀티프로세싱 이슈 회피
    cpu_count = os.cpu_count() or 1
    return min(4, max(cpu_count // 2, 1))


def is_dataset_root(path: str | Path) -> bool:
    """주어진 경로가 유효한 데이터셋 루트인지 확인한다.

    Args:
        path: 검사할 경로.

    Returns:
        필수 자식 파일/디렉토리가 모두 존재하면 True.
    """
    path = Path(path).expanduser()
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_DATASET_CHILDREN)


def _unique_paths(paths: Iterable[str | Path | None]) -> List[Path]:
    """경로 목록에서 중복을 제거하고 유효한 Path 목록을 반환한다.

    Args:
        paths: 경로 이터러블 (None 포함 가능).

    Returns:
        중복 제거된 절대 경로 리스트.
    """
    seen: set[str] = set()
    resolved: List[Path] = []
    for path in paths:
        if path is None or str(path).strip() == "":
            continue
        try:
            candidate = Path(path).expanduser().resolve()
        except OSError:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate)
    return resolved


def _iter_dataset_search_roots() -> List[Path]:
    """데이터셋 자동 탐색에 사용할 기본 루트 경로 목록을 반환한다.

    Returns:
        탐색 루트 경로 리스트.
    """
    script_dir = default_project_dir()
    downloads_dir = Path.home() / "Downloads"
    return _unique_paths([Path.cwd(), script_dir, script_dir.parent, downloads_dir])


def resolve_data_root(explicit: Optional[str]) -> Path:
    """데이터셋 루트 경로를 결정한다.

    명시적 경로가 없으면 환경변수 → 현재 디렉토리 → 스크립트 디렉토리 순으로 탐색한다.

    Args:
        explicit: --data-root 인자값 (없으면 None).

    Returns:
        유효한 데이터셋 루트 Path.

    Raises:
        FileNotFoundError: 데이터셋 루트를 찾지 못했을 때.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not is_dataset_root(candidate):
            raise FileNotFoundError(
                f"Dataset root not found or incomplete: {candidate}\n"
                "Expected train.csv, dev.csv, sample_submission.csv and train/dev/test directories."
            )
        return candidate

    env_root = os.environ.get("PHYSICS_DATA_ROOT")
    script_dir = default_project_dir()
    downloads_dir = Path.home() / "Downloads"
    explicit_candidates = _unique_paths(
        [
            env_root,
            Path.cwd(),
            script_dir,
            Path.cwd() / "open",
            Path.cwd() / "open (7) 2",
            script_dir.parent / "open",
            script_dir.parent / "open (7) 2",
            downloads_dir / "open",
            downloads_dir / "open (7) 2",
        ]
    )
    for candidate in explicit_candidates:
        if is_dataset_root(candidate):
            return candidate

    for root in _iter_dataset_search_roots():
        if not root.exists() or not root.is_dir():
            continue
        preferred: List[Path] = []
        other_dirs: List[Path] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if "open" in child.name.lower():
                preferred.append(child)
            else:
                other_dirs.append(child)
        for candidate in preferred + other_dirs:
            if is_dataset_root(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not auto-detect the dataset root. Pass --data-root or set PHYSICS_DATA_ROOT."
    )


def default_motion_csv(data_root: str | Path) -> Path:
    """데이터 루트 내 모션 타겟 CSV 기본 경로를 반환한다.

    Args:
        data_root: 데이터셋 루트 경로.

    Returns:
        motion_targets.csv Path 객체.
    """
    return Path(data_root) / "motion_targets.csv"


def resolve_motion_csv(data_root: str | Path, motion_csv: Optional[str]) -> Path:
    """모션 CSV 경로를 결정한다.

    Args:
        data_root: 데이터셋 루트 경로.
        motion_csv: 명시적 경로 (없으면 None).

    Returns:
        확정된 모션 CSV Path 객체.
    """
    if motion_csv is None:
        return default_motion_csv(data_root)
    return Path(motion_csv).expanduser().resolve()


# ============================================================
# 디바이스 유틸리티
# ============================================================

def get_runtime_device() -> torch.device:
    """사용 가능한 최적의 연산 디바이스를 반환한다.

    Returns:
        cuda > mps > cpu 순서로 선택된 device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_pin_memory(device: torch.device) -> bool:
    """CUDA 환경에서 pin_memory 사용 여부를 반환한다.

    Args:
        device: 연산 디바이스.

    Returns:
        CUDA이면 True.
    """
    return device.type == "cuda"


def use_non_blocking(device: torch.device) -> bool:
    """CUDA 환경에서 non_blocking 전송 사용 여부를 반환한다.

    Args:
        device: 연산 디바이스.

    Returns:
        CUDA이면 True.
    """
    return device.type == "cuda"


def optimize_runtime_for_device(device: torch.device) -> None:
    """디바이스에 맞게 PyTorch 런타임을 최적화한다.

    Args:
        device: 연산 디바이스.
    """
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")  # Tensor Core 활용
        except RuntimeError:
            pass
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True  # 고정 입력 크기 시 연산 최적화


def describe_runtime_device(device: torch.device) -> str:
    """디바이스 정보를 사람이 읽기 쉬운 문자열로 반환한다.

    Args:
        device: 연산 디바이스.

    Returns:
        디바이스 설명 문자열.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"cuda ({name}, {total_gb:.1f} GB)"
    if device.type == "mps":
        return "mps"
    return "cpu"


# ============================================================
# 데이터 유틸리티
# ============================================================

def read_csv(path: str | Path) -> pd.DataFrame:
    """UTF-8-BOM을 처리하며 CSV 파일을 읽는다.

    Args:
        path: CSV 파일 경로.

    Returns:
        읽어들인 DataFrame.
    """
    return pd.read_csv(path, encoding="utf-8-sig")


def label_to_int(label: str) -> int:
    """레이블 문자열을 정수로 변환한다.

    Args:
        label: 'unstable' 또는 'stable'.

    Returns:
        unstable이면 1, stable이면 0.
    """
    return 1 if label == "unstable" else 0


def required_image_multiple(backbone_name: str) -> Optional[int]:
    """백본에서 요구하는 이미지 크기의 배수를 반환한다.

    Args:
        backbone_name: 백본 이름 문자열.

    Returns:
        DINOv2는 14, 그 외는 None.
    """
    name = backbone_name.lower()
    if name.startswith("dinov2_"):
        return 14  # DINOv2 패치 크기 14의 배수여야 함
    return None


def validate_image_size(backbone_name: str, image_size: int) -> None:
    """이미지 크기가 백본 요구사항을 만족하는지 검증한다.

    Args:
        backbone_name: 백본 이름.
        image_size: 검사할 이미지 크기.

    Raises:
        ValueError: 이미지 크기가 요구 배수의 배수가 아닐 때.
    """
    multiple = required_image_multiple(backbone_name)
    if multiple is not None and image_size % multiple != 0:
        raise ValueError(
            f"Backbone '{backbone_name}' requires --image-size to be a multiple of {multiple}, "
            f"but got {image_size}."
        )


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """NumPy 배열에 시그모이드 함수를 적용한다.

    Args:
        x: 입력 배열.

    Returns:
        시그모이드 적용 결과 배열.
    """
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_dev_logloss(logloss_value: float) -> Dict[str, str | float]:
    """dev OOF logloss를 평가하고 상태 보고서를 반환한다.

    Args:
        logloss_value: 계산된 logloss 값.

    Returns:
        metric, status, message, target_low, target_high 키를 가진 딕셔너리.
    """
    if logloss_value < DEV_LOGLOSS_OVERFIT_LOW:
        status = "overfit_risk"
        message = "DEV OOF LOGLOSS is below 0.0001. This usually means severe dev overfit."
    elif logloss_value > DEV_LOGLOSS_DESIGN_BAD_HIGH:
        status = "design_problem"
        message = "DEV OOF LOGLOSS is above 0.05. This usually means the design is off."
    elif DEV_LOGLOSS_TARGET_LOW <= logloss_value <= DEV_LOGLOSS_TARGET_HIGH:
        status = "target_band"
        message = "DEV OOF LOGLOSS is inside the target band (0.015 to 0.03)."
    else:
        status = "outside_target_band"
        message = "DEV OOF LOGLOSS is usable but outside the preferred 0.015 to 0.03 band."
    return {
        "metric": float(logloss_value),
        "status": status,
        "message": message,
        "target_low": DEV_LOGLOSS_TARGET_LOW,
        "target_high": DEV_LOGLOSS_TARGET_HIGH,
    }


def print_dev_logloss_report(name: str, logloss_value: float) -> Dict[str, str | float]:
    """dev OOF logloss 평가 결과를 콘솔에 출력하고 보고서를 반환한다.

    Args:
        name: 실행 단계 이름 (예: 'train-design').
        logloss_value: 계산된 logloss 값.

    Returns:
        evaluate_dev_logloss()와 동일한 보고서 딕셔너리.
    """
    report = evaluate_dev_logloss(logloss_value)
    print(
        f"[{name}] DEV OOF LOGLOSS={logloss_value:.6f} "
        f"(target {DEV_LOGLOSS_TARGET_LOW:.3f}~{DEV_LOGLOSS_TARGET_HIGH:.3f}) -> {report['status']}"
    )
    print(f"[{name}] {report['message']}")
    return report


# ============================================================
# Motion Target Extraction
# ============================================================

@dataclass
class MotionExtractionConfig:
    """동영상 모션 타겟 추출 설정.

    Attributes:
        resize: 프레임 리사이즈 크기 (너비, 높이).
        thr_low: 낮은 모션 임계값 (픽셀 절대차 평균).
        thr_mid: 중간 모션 임계값.
        thr_high: 높은 모션 임계값.
    """

    resize: Tuple[int, int] = (64, 64)  # 처리 속도를 위한 다운샘플 크기
    thr_low: float = 2.0                # 작은 움직임 감지 임계값
    thr_mid: float = 5.0                # 중간 움직임 감지 임계값
    thr_high: float = 10.0              # 큰 움직임 감지 임계값


def _first_hit(arr: np.ndarray, thr: float) -> int:
    """배열에서 임계값을 처음 초과하는 인덱스를 반환한다.

    Args:
        arr: 탐색할 1D 배열.
        thr: 임계값.

    Returns:
        처음 초과하는 인덱스+1, 없으면 -1.
    """
    idx = np.where(arr > thr)[0]
    return int(idx[0] + 1) if len(idx) else -1


def _severity_bucket(max_diff_first: float) -> int:
    """최대 모션 강도로부터 심각도 버킷을 계산한다.

    Args:
        max_diff_first: 첫 프레임 대비 최대 평균 절대차.

    Returns:
        0=tiny, 1=small, 2=mid, 3=large.
    """
    if max_diff_first < 2.0:   # 거의 움직임 없음
        return 0
    if max_diff_first < 5.0:   # 작은 움직임
        return 1
    if max_diff_first < 10.0:  # 중간 움직임
        return 2
    return 3                   # 큰 움직임 (붕괴)


def _onset_bucket(first_move_thr2: int, first_move_thr5: int) -> int:
    """최초 움직임 발생 시점으로부터 온셋 버킷을 계산한다.

    Args:
        first_move_thr2: thr_low 기준 최초 움직임 프레임.
        first_move_thr5: thr_mid 기준 최초 움직임 프레임.

    Returns:
        0=very_early, 1=early, 2=late, 3=no_strong_hit.
    """
    onset = first_move_thr5 if first_move_thr5 >= 0 else first_move_thr2
    if onset < 0:
        return 3   # 강한 움직임 없음
    if onset < 10:
        return 0   # 매우 이른 붕괴 (10프레임 이내)
    if onset < 20:
        return 1   # 이른 붕괴 (20프레임 이내)
    return 2       # 늦은 붕괴


def _soft_target_from_motion(label_int: int, max_diff_first: float, mean_diff_prev: float) -> float:
    """모션 정보로부터 소프트 레이블(Soft Target)을 생성한다.

    logloss 과신뢰(overconfidence)를 완화하기 위해 경계 근처의 레이블을 부드럽게 처리한다.
    하드 레이블은 암묵적으로 유지된다.

    Args:
        label_int: 하드 레이블 (0=stable, 1=unstable).
        max_diff_first: 첫 프레임 대비 최대 평균 절대차.
        mean_diff_prev: 이전 프레임 대비 평균 절대차.

    Returns:
        소프트 확률값 (float, 0.0~1.0).
    """
    # 모션 크기를 0~1.5로 거친 정규화
    motion_score = (
        0.65 * min(max_diff_first / 10.0, 1.5)
        + 0.35 * min(mean_diff_prev / 0.15, 1.5)
    )
    motion_score = min(max(motion_score, 0.0), 1.5)

    if label_int == 0:
        # stable: 약간의 흔들림은 허용하지만 낮은 확률 유지
        return float(np.clip(0.02 + 0.10 * min(motion_score, 1.0), 0.02, 0.15))
    # unstable: 가벼운 붕괴는 격렬한 붕괴보다 소프트하게
    return float(np.clip(0.65 + 0.30 * min(motion_score, 1.0), 0.65, 0.98))


def extract_motion_targets(
    data_root: str | Path,
    out_csv: str | Path,
    cfg: MotionExtractionConfig,
) -> pd.DataFrame:
    """학습 동영상에서 모션 타겟을 추출하여 CSV로 저장한다.

    Args:
        data_root: 데이터셋 루트 경로.
        out_csv: 출력 CSV 파일 경로.
        cfg: 모션 추출 설정.

    Returns:
        추출된 모션 타겟 DataFrame.
    """
    data_root = Path(data_root)
    train_df = read_csv(data_root / "train.csv")
    rows: List[Dict[str, float | int | str]] = []

    sample_iter = train_df[["id", "label"]].itertuples(index=False)
    sample_iter = tqdm(sample_iter, total=len(train_df), desc="extract-motion", dynamic_ncols=True)
    for sid, label in sample_iter:
        video_path = data_root / "train" / sid / "simulation.mp4"
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ok, first = cap.read()
        if not ok:
            cap.release()
            continue

        # 첫 프레임을 기준 그레이스케일로 저장
        first_gray = (
            cv2.cvtColor(cv2.resize(first, cfg.resize), cv2.COLOR_BGR2GRAY)
            .astype(np.float32)
        )
        prev_gray = first_gray
        mad_to_first: List[float] = []  # 첫 프레임 대비 MAD 시계열
        mad_prev: List[float] = []      # 이전 프레임 대비 MAD 시계열

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = (
                cv2.cvtColor(cv2.resize(frame, cfg.resize), cv2.COLOR_BGR2GRAY)
                .astype(np.float32)
            )
            mad_to_first.append(float(np.mean(np.abs(gray - first_gray))))
            mad_prev.append(float(np.mean(np.abs(gray - prev_gray))))
            prev_gray = gray
        cap.release()

        if len(mad_to_first) == 0:
            max_diff_first = 0.0
            mean_diff_first = 0.0
            max_diff_prev = 0.0
            mean_diff_prev = 0.0
            first_move_thr2 = -1
            first_move_thr5 = -1
            first_move_thr10 = -1
        else:
            arr_first = np.array(mad_to_first, dtype=np.float32)
            arr_prev = np.array(mad_prev, dtype=np.float32)
            max_diff_first = float(arr_first.max())
            mean_diff_first = float(arr_first.mean())
            max_diff_prev = float(arr_prev.max())
            mean_diff_prev = float(arr_prev.mean())
            first_move_thr2 = _first_hit(arr_first, cfg.thr_low)   # 처음 thr_low 초과 프레임
            first_move_thr5 = _first_hit(arr_first, cfg.thr_mid)   # 처음 thr_mid 초과 프레임
            first_move_thr10 = _first_hit(arr_first, cfg.thr_high) # 처음 thr_high 초과 프레임

        label_int = label_to_int(label)
        rows.append(
            {
                "id": sid,
                "label": label,
                "label_int": label_int,
                "frames": total,
                "fps": fps,
                "max_diff_first": max_diff_first,
                "mean_diff_first": mean_diff_first,
                "max_diff_prev": max_diff_prev,
                "mean_diff_prev": mean_diff_prev,
                "first_move_thr2": first_move_thr2,
                "first_move_thr5": first_move_thr5,
                "first_move_thr10": first_move_thr10,
                "severity_bucket": _severity_bucket(max_diff_first),
                "onset_bucket": _onset_bucket(first_move_thr2, first_move_thr5),
                "soft_target": _soft_target_from_motion(label_int, max_diff_first, mean_diff_prev),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)
    return out_df


# ============================================================
# Geometry Clustering for Grouped CV
# ============================================================

@dataclass
class ClusterConfig:
    """기하학적 클러스터링 설정.

    Attributes:
        n_clusters: K-Means 클러스터 수.
        front_crop: 전면 뷰 크롭 박스 (x1, y1, x2, y2).
        top_crop: 상단 뷰 크롭 박스 (x1, y1, x2, y2).
        downsample: 다운샘플 크기 (너비, 높이).
        random_state: K-Means 난수 시드.
    """

    n_clusters: int = 32                              # 폴드 분리를 위한 클러스터 수
    front_crop: Tuple[int, int, int, int] = (96, 80, 288, 320)   # 전면 뷰 크롭 좌표
    top_crop: Tuple[int, int, int, int] = (112, 112, 272, 272)   # 상단 뷰 크롭 좌표
    downsample: Tuple[int, int] = (24, 24)            # 특징 추출용 다운샘플 크기
    random_state: int = 42                            # 재현성을 위한 난수 시드


def _load_center_gray(
    data_root: Path,
    split: str,
    sid: str,
    view: str,
    crop: Tuple[int, int, int, int],
    size: Tuple[int, int],
) -> np.ndarray:
    """이미지를 크롭·다운샘플·그레이스케일로 변환하여 1D 벡터로 반환한다.

    Args:
        data_root: 데이터셋 루트 경로.
        split: 'train' 또는 'dev'.
        sid: 샘플 ID.
        view: 'front' 또는 'top'.
        crop: (x1, y1, x2, y2) 크롭 좌표.
        size: 다운샘플 목표 크기 (너비, 높이).

    Returns:
        정규화된 그레이스케일 1D 벡터 (float32, 0~1).
    """
    arr = cv2.cvtColor(
        cv2.imread(str(data_root / split / sid / f"{view}.png")),
        cv2.COLOR_BGR2RGB,
    )
    x1, y1, x2, y2 = crop
    arr = arr[y1:y2, x1:x2]
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = np.array(Image.fromarray(arr).resize(size), dtype=np.float32) / 255.0  # 0~1 정규화
    return arr.reshape(-1)


def build_geometry_clusters(
    data_root: str | Path,
    train_df: pd.DataFrame,
    dev_df: Optional[pd.DataFrame],
    cfg: ClusterConfig,
) -> pd.DataFrame:
    """전면·상단 뷰 외관 특징으로 기하학적 클러스터를 생성한다.

    훈련과 검증 데이터를 합쳐 K-Means로 클러스터링하여
    GroupKFold에 사용할 그룹 레이블을 생성한다.

    Args:
        data_root: 데이터셋 루트 경로.
        train_df: 훈련 DataFrame.
        dev_df: 검증 DataFrame (None이면 사용 안 함).
        cfg: 클러스터링 설정.

    Returns:
        'id'와 'geometry_cluster' 컬럼을 가진 DataFrame.
    """
    data_root = Path(data_root)
    parts: List[pd.DataFrame] = [train_df.copy()]
    if dev_df is not None:
        parts.append(dev_df.copy())
    all_df = pd.concat(parts, ignore_index=True)
    if all_df.empty:
        return pd.DataFrame(columns=["id", "geometry_cluster"])

    feats: List[np.ndarray] = []
    for sid in all_df["id"].tolist():
        split = "train" if sid.startswith("TRAIN") else "dev"
        front = _load_center_gray(data_root, split, sid, "front", cfg.front_crop, cfg.downsample)
        top = _load_center_gray(data_root, split, sid, "top", cfg.top_crop, cfg.downsample)
        feats.append(np.concatenate([front, top]))  # 전면+상단 특징 연결
    X = np.stack(feats)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    n_clusters = max(1, min(cfg.n_clusters, len(all_df)))
    km = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init=20)
    clusters = km.fit_predict(Xs)
    out = all_df[["id"]].copy()
    out["geometry_cluster"] = clusters
    return out


# ============================================================
# Image Transforms
# ============================================================

class CenterPhysicsCrop:
    """구조물 중심 영역을 크롭하여 배경 누수를 줄이는 변환.

    전면 뷰와 상단 뷰에 서로 다른 크롭 비율을 적용한다.

    Attributes:
        view: 'front' 또는 'top'.
    """

    def __init__(self, view: str):
        """초기화.

        Args:
            view: 뷰 종류 ('front' 또는 'top').
        """
        self.view = view

    def __call__(self, img: Image.Image) -> Image.Image:
        """이미지를 중심 영역으로 크롭한다.

        Args:
            img: 크롭할 PIL 이미지.

        Returns:
            크롭된 PIL 이미지.
        """
        w, h = img.size
        if self.view == "front":
            box = (int(0.25 * w), int(0.20 * h), int(0.75 * w), int(0.88 * h))  # 전면: 좌우 25%, 상하 12/88%
        else:
            box = (int(0.29 * w), int(0.29 * h), int(0.71 * w), int(0.71 * h))  # 상단: 좌우우 29~71%
        return img.crop(box)


def build_train_transform(view: str, image_size: int) -> transforms.Compose:
    """학습용 이미지 변환 파이프라인을 생성한다.

    Args:
        view: 뷰 종류 ('front' 또는 'top').
        image_size: 출력 이미지 크기 (픽셀).

    Returns:
        학습용 transforms.Compose 객체.
    """
    return transforms.Compose(
        [
            CenterPhysicsCrop(view=view),
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04
                )
            ], p=0.8),  # 색상 증강: 80% 확률
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.35
            ),  # 가우시안 블러: 35% 확률
            transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.2),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.35),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"
            ),  # 랜덤 지우기: 10% 확률
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_valid_transform(view: str, image_size: int) -> transforms.Compose:
    """검증/추론용 이미지 변환 파이프라인을 생성한다.

    Args:
        view: 뷰 종류 ('front' 또는 'top').
        image_size: 출력 이미지 크기 (픽셀).

    Returns:
        검증용 transforms.Compose 객체 (증강 없음).
    """
    return transforms.Compose(
        [
            CenterPhysicsCrop(view=view),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ============================================================
# Dataset
# ============================================================

class DualViewDataset(Dataset):
    """전면·상단 이중 뷰 이미지와 보조 타겟을 로드하는 Dataset.

    Attributes:
        data_root: 데이터셋 루트 경로.
        df: 샘플 정보 DataFrame.
        split_map: 샘플 ID → 분할명('train'/'dev'/'test') 매핑.
        front_transform: 전면 뷰 이미지 변환.
        top_transform: 상단 뷰 이미지 변환.
        training: 학습 모드 여부.
        top_normalizer: 체커보드 상단 뷰 정규화기 (비활성화 시 None).
        geometry_cache: 기하학적 특징 캐시.
    """

    def __init__(
        self,
        data_root: str | Path,
        frame_df: pd.DataFrame,
        split_map: Dict[str, str],
        front_transform: transforms.Compose,
        top_transform: transforms.Compose,
        training: bool,
        checkerboard_top_normalize: bool = True,
    ) -> None:
        """초기화.

        Args:
            data_root: 데이터셋 루트 경로.
            frame_df: 샘플 정보 DataFrame.
            split_map: 샘플 ID → 분할명 매핑.
            front_transform: 전면 뷰 이미지 변환.
            top_transform: 상단 뷰 이미지 변환.
            training: 학습 모드 여부.
            checkerboard_top_normalize: 체커보드 정규화 활성화 여부.
        """
        self.data_root = Path(data_root)
        self.df = frame_df.reset_index(drop=True).copy()
        self.split_map = split_map
        self.front_transform = front_transform
        self.top_transform = top_transform
        self.training = training
        self.top_normalizer = (
            CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True))
            if checkerboard_top_normalize
            else None
        )
        self.geometry_cache = GeometryFeatureCache()

    def __len__(self) -> int:
        """데이터셋 크기를 반환한다."""
        return len(self.df)

    def _load_img(self, sid: str, view: str) -> tuple[Path, Image.Image]:
        """샘플 ID와 뷰 이름으로 이미지를 로드한다.

        Args:
            sid: 샘플 ID.
            view: 'front' 또는 'top'.

        Returns:
            (이미지 경로, PIL 이미지) 튜플.
        """
        split = self.split_map[sid]
        path = self.data_root / split / sid / f"{view}.png"
        image = Image.open(path).convert("RGB")
        if view == "top" and self.top_normalizer is not None:
            image = self.top_normalizer.normalize(path, image)
        return path, image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """인덱스로 샘플을 반환한다.

        Args:
            idx: 샘플 인덱스.

        Returns:
            id, front, top, geom_feat, support_target, collapse_margin,
            label 및 보조 타겟을 포함하는 딕셔너리.
        """
        row = self.df.iloc[idx]
        sid = row["id"]
        _front_path, front_img = self._load_img(sid, "front")
        _top_path, top_img = self._load_img(sid, "top")
        geom_vec, support_target, collapse_margin = self.geometry_cache.get(
            sid,
            np.asarray(front_img, dtype=np.uint8),
            np.asarray(top_img, dtype=np.uint8),
        )
        front = self.front_transform(front_img)
        top = self.top_transform(top_img)

        sample: Dict[str, torch.Tensor | str] = {
            "id": sid,
            "front": front,
            "top": top,
            "geom_feat": torch.tensor(geom_vec, dtype=torch.float32),
            "support_target": torch.tensor(support_target, dtype=torch.float32),
            "collapse_margin": torch.tensor(float(collapse_margin), dtype=torch.float32),
        }

        if "label_int" in row:
            sample["label"] = torch.tensor(float(row["label_int"]), dtype=torch.float32)
        else:
            sample["label"] = torch.tensor(-1.0, dtype=torch.float32)

        # 보조 타겟 (NaN 안전 처리)
        aux_float_cols = ["max_diff_first", "mean_diff_prev", "soft_target"]
        aux_int_cols = ["severity_bucket", "onset_bucket", "source_domain"]
        for col in aux_float_cols:
            val = row[col] if col in row and pd.notna(row[col]) else np.nan
            sample[col] = torch.tensor(
                float(val) if pd.notna(val) else float("nan"),
                dtype=torch.float32,
            )
        for col in aux_int_cols:
            val = row[col] if col in row and pd.notna(row[col]) else -1
            sample[col] = torch.tensor(int(val), dtype=torch.long)
        return sample


# ============================================================
# Model
# ============================================================

class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) 풀링 레이어.

    학습 가능한 파라미터 p로 평균-최대 사이의 풀링을 수행한다.

    Attributes:
        p: 풀링 지수 (학습 가능).
        eps: 수치 안정성을 위한 엡실론.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        """초기화.

        Args:
            p: 초기 풀링 지수 (기본값 3.0).
            eps: 수치 안정성 엡실론.
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)  # 학습 가능한 풀링 지수
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GeM 풀링을 수행한다.

        Args:
            x: (B, C, H, W) 특징 맵.

        Returns:
            (B, C) 풀링된 특징 벡터.
        """
        return (
            F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1)))
            .pow(1.0 / self.p)
            .flatten(1)
        )


def create_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """백본 네트워크와 특징 차원을 생성한다.

    Args:
        name: 백본 이름 (SUPPORTED_BACKBONES 참고).
        pretrained: 사전학습 가중치 사용 여부.

    Returns:
        (backbone 모듈, 출력 특징 차원) 튜플.

    Raises:
        ValueError: 지원하지 않는 백본 이름일 때.
    """
    name = name.lower()
    if name in ("dinov2_vits14", "dinov2_vits14_reg"):
        backbone = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        feat = 768  # CLS 토큰(384) + 패치 토큰 평균(384) 연결
        return backbone, feat
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=weights)
        feat = 768
        backbone = nn.Sequential(
            m.features, nn.LayerNorm((feat, 1, 1), eps=1e-6, elementwise_affine=True)
        )
        return backbone, feat
    if name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        m = models.convnext_small(weights=weights)
        feat = 768
        backbone = nn.Sequential(
            m.features, nn.LayerNorm((feat, 1, 1), eps=1e-6, elementwise_affine=True)
        )
        return backbone, feat
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_v2_s(weights=weights)
        feat = 1280
        backbone = m.features
        return backbone, feat
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        feat = 2048
        backbone = nn.Sequential(*(list(m.children())[:-2]))  # avgpool·fc 제거
        return backbone, feat
    raise ValueError(f"Unsupported backbone: {name}")


class ViewEncoder(nn.Module):
    """단일 뷰 이미지를 인코딩하는 모듈.

    백본 + GeM 풀링 + 선형 프로젝션으로 구성된다.

    Attributes:
        backbone_name: 백본 이름 (소문자).
        backbone: 백본 네트워크.
        pool: GeM 풀링 레이어.
        proj: 선형 프로젝션 헤드.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True, out_dim: int = 512):
        """초기화.

        Args:
            backbone_name: 백본 이름.
            pretrained: 사전학습 가중치 사용 여부.
            out_dim: 출력 임베딩 차원.
        """
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.backbone, feat_dim = create_backbone(backbone_name, pretrained=pretrained)
        self.pool = GeM()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.3),  # 과적합 방지 드롭아웃
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """뷰 이미지를 임베딩 벡터로 인코딩한다.

        Args:
            x: (B, 3, H, W) 입력 이미지 텐서.

        Returns:
            (B, out_dim) 임베딩 텐서.
        """
        if self.backbone_name.startswith("dinov2_"):
            feat_dict = self.backbone.forward_features(x)
            cls = feat_dict["x_norm_clstoken"]         # (B, 384) CLS 토큰
            patches = feat_dict["x_norm_patchtokens"]  # (B, N, 384) 패치 토큰
            feat = torch.cat([cls, patches.mean(dim=1)], dim=1)  # (B, 768) 연결
        else:
            fmap = self.backbone(x)
            if isinstance(fmap, (list, tuple)):
                fmap = fmap[-1]
            if fmap.ndim == 2:
                feat = fmap
            else:
                feat = self.pool(fmap)  # GeM 풀링으로 공간 차원 제거
        return self.proj(feat)


class GradientReversal(torch.autograd.Function):
    """도메인 적응을 위한 그래디언트 역전 레이어.

    순전파는 항등 함수, 역전파는 그래디언트에 -λ를 곱한다.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        """순전파: 항등 함수.

        Args:
            ctx: 컨텍스트 (lambd 저장).
            x: 입력 텐서.
            lambd: 역전 강도 계수.

        Returns:
            입력과 동일한 텐서.
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """역전파: 그래디언트에 -λ 적용.

        Args:
            ctx: 순전파에서 저장된 컨텍스트.
            grad_output: 상위 그래디언트.

        Returns:
            역전된 그래디언트, None (lambd는 학습 대상 아님).
        """
        return -ctx.lambd * grad_output, None


class DualViewPhysicsModel(nn.Module):
    """전면·상단 이중 뷰 물리 안정성 예측 모델.

    전면/상단 인코더 → Transformer 퓨전 → 분류 헤드 +
    보조 헤드(모션 회귀, 온셋 분류, 심각도 분류, 기하학, 도메인) 구조.

    Attributes:
        use_geometry_reasoning: 기하학적 특징 브랜치 활성화 여부.
        front_encoder: 전면 뷰 인코더.
        top_encoder: 상단 뷰 인코더.
        view_embed: 뷰별 위치 임베딩 파라미터.
        fusion: Transformer 퓨전 인코더.
        norm: 퓨전 출력 LayerNorm.
        classifier: 주 분류 헤드.
        motion_reg: 모션 회귀 헤드.
        onset_head: 온셋 버킷 분류 헤드.
        severity_head: 심각도 버킷 분류 헤드.
        use_domain_head: 도메인 적응 헤드 활성화 여부.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        emb_dim: int = 512,
        use_domain_head: bool = False,
        geometry_dim: int = len(GEOMETRY_FEATURE_NAMES),
        use_geometry_reasoning: bool = False,
    ) -> None:
        """초기화.

        Args:
            backbone_name: 백본 이름.
            pretrained: 사전학습 가중치 사용 여부.
            emb_dim: 임베딩 차원.
            use_domain_head: 도메인 적응 헤드 사용 여부.
            geometry_dim: 기하학적 특징 입력 차원.
            use_geometry_reasoning: 기하학적 특징 브랜치 사용 여부.
        """
        super().__init__()
        self.use_geometry_reasoning = use_geometry_reasoning
        self.front_encoder = ViewEncoder(backbone_name, pretrained=pretrained, out_dim=emb_dim)
        self.top_encoder = ViewEncoder(backbone_name, pretrained=pretrained, out_dim=emb_dim)
        self.view_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)  # 뷰 위치 임베딩

        # Transformer 퓨전 인코더 (2레이어)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=emb_dim * 4,
            dropout=0.35,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(emb_dim)

        # 특징 차원 계산
        image_feat_dim = emb_dim * 3                                      # front + top + fused_mean
        geom_emb_dim = emb_dim // 2
        fused_feat_dim = (
            image_feat_dim + geom_emb_dim if use_geometry_reasoning else image_feat_dim
        )

        if use_geometry_reasoning:
            self.geometry_proj = nn.Sequential(
                nn.LayerNorm(geometry_dim),
                nn.Linear(geometry_dim, geom_emb_dim),
                nn.GELU(),
                nn.Dropout(0.25),
            )

        # 주 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(fused_feat_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.5),   # 과적합 방지 강한 드롭아웃
            nn.Linear(emb_dim, 1),
        )

        # 보조 헤드들
        self.motion_reg = nn.Sequential(
            nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2)
        )
        self.onset_head = nn.Sequential(
            nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4)
        )

        if use_geometry_reasoning:
            self.support_reg = nn.Sequential(
                nn.Linear(image_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2)
            )
            self.margin_reg = nn.Sequential(
                nn.Linear(image_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 1)
            )

        self.use_domain_head = use_domain_head
        if use_domain_head:
            self.domain_head = nn.Sequential(
                nn.Linear(fused_feat_dim, emb_dim // 2),
                nn.GELU(),
                nn.Linear(emb_dim // 2, 2),
            )

    def forward(
        self,
        front: torch.Tensor,
        top: torch.Tensor,
        geom_feat: torch.Tensor,
        grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """순전파.

        Args:
            front: (B, 3, H, W) 전면 뷰 이미지 텐서.
            top: (B, 3, H, W) 상단 뷰 이미지 텐서.
            geom_feat: (B, geometry_dim) 기하학적 특징 텐서.
            grl_lambda: 그래디언트 역전 강도 (도메인 헤드용).

        Returns:
            logit, motion_reg, onset_logit, severity_logit, feat 키를 포함하는
            출력 딕셔너리 (기하학/도메인 헤드는 활성화 시 추가).
        """
        f = self.front_encoder(front)  # (B, emb_dim) 전면 임베딩
        t = self.top_encoder(top)      # (B, emb_dim) 상단 임베딩

        # 뷰 위치 임베딩 추가 후 Transformer 퓨전
        tokens = torch.stack([f + self.view_embed[0], t + self.view_embed[1]], dim=1)
        fused = self.fusion(tokens)
        fused_mean = self.norm(fused.mean(dim=1))          # 토큰 평균 → LayerNorm

        image_feat = torch.cat([f, t, fused_mean], dim=1)  # (B, emb_dim*3) 이미지 특징

        if self.use_geometry_reasoning:
            geom_emb = self.geometry_proj(geom_feat)
            feat = torch.cat([image_feat, geom_emb], dim=1)  # 기하학 특징 연결
        else:
            feat = image_feat

        out = {
            "logit": self.classifier(feat).squeeze(1),        # 주 분류 로짓
            "motion_reg": self.motion_reg(feat),               # 모션 회귀 출력
            "onset_logit": self.onset_head(feat),              # 온셋 분류 로짓
            "severity_logit": self.severity_head(feat),        # 심각도 분류 로짓
            "feat": feat,
        }
        if self.use_geometry_reasoning:
            out["support_reg"] = self.support_reg(image_feat)       # 지지면 회귀
            out["margin_reg"] = self.margin_reg(image_feat).squeeze(1)  # 붕괴 여유도 회귀
        if self.use_domain_head:
            rev = GradientReversal.apply(feat, grl_lambda)
            out["domain_logit"] = self.domain_head(rev)              # 도메인 분류 로짓
        return out


# ============================================================
# Losses and Calibration
# ============================================================

@dataclass
class TrainConfig:
    """학습 파이프라인 전체 설정.

    Attributes:
        data_root: 데이터셋 루트 경로.
        out_dir: 학습 결과 저장 디렉토리.
        motion_csv: 모션 타겟 CSV 경로 (None이면 자동 결정).
        backbone: 백본 이름.
        pretrained: 사전학습 가중치 사용 여부.
        image_size: 입력 이미지 크기 (픽셀).
        batch_size: 배치 크기.
        num_workers: DataLoader 워커 수.
        epochs: 학습 에폭 수.
        lr: 헤드 학습률.
        backbone_lr: 백본 학습률 (DINOv2 전용).
        warmup_epochs: 웜업 에폭 수.
        weight_decay: AdamW 가중치 감쇠.
        num_folds: CV 폴드 수.
        seed: 난수 시드.
        use_domain_head: 도메인 적응 헤드 사용 여부.
        use_amp: 자동 혼합 정밀도 사용 여부.
        grad_clip: 그래디언트 클리핑 최댓값.
        aux_motion_weight: 모션 회귀 손실 가중치.
        aux_onset_weight: 온셋 손실 가중치.
        aux_severity_weight: 심각도 손실 가중치.
        aux_support_weight: 지지면 손실 가중치.
        aux_margin_weight: 붕괴 여유도 손실 가중치.
        domain_weight: 도메인 손실 가중치.
        tta_passes: TTA(Test-Time Augmentation) 횟수.
        checkerboard_top_normalize: 체커보드 정규화 사용 여부.
        use_geometry_reasoning: 기하학적 특징 브랜치 사용 여부.
    """

    data_root: str = ""
    out_dir: str = str(default_runs_dir("default"))
    motion_csv: Optional[str] = None
    backbone: str = DEFAULT_BACKBONE
    pretrained: bool = True
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 8
    num_workers: int = default_num_workers()
    epochs: int = 20
    lr: float = 2e-4                    # 헤드 학습률
    backbone_lr: float = 1e-5           # 백본 미세조정 학습률 (DINOv2)
    warmup_epochs: int = 3              # 선형 웜업 에폭 수
    weight_decay: float = 1e-4          # AdamW 가중치 감쇠
    num_folds: int = 5
    seed: int = 42
    use_domain_head: bool = False
    use_amp: bool = True                # CUDA AMP 활성화
    grad_clip: float = 1.0              # 그래디언트 클리핑 기준값
    aux_motion_weight: float = 0.20     # 모션 회귀 손실 가중치
    aux_onset_weight: float = 0.15      # 온셋 분류 손실 가중치
    aux_severity_weight: float = 0.15   # 심각도 분류 손실 가중치
    aux_support_weight: float = 0.08    # 지지면 회귀 손실 가중치
    aux_margin_weight: float = 0.08     # 붕괴 여유도 손실 가중치
    domain_weight: float = 0.05         # 도메인 분류 손실 가중치
    tta_passes: int = 4                 # TTA 반복 횟수
    checkerboard_top_normalize: bool = True
    use_geometry_reasoning: bool = False


class TemperatureScaler(nn.Module):
    """로짓을 온도 파라미터로 스케일링하는 캘리브레이터.

    LBFGS로 검증 세트 NLL을 최소화하여 온도를 학습한다.

    Attributes:
        temperature: 학습 가능한 온도 파라미터.
    """

    def __init__(self) -> None:
        """초기화."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # 초기 온도=1 (변환 없음)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """로짓을 온도로 나눠 반환한다.

        Args:
            logits: 스케일링할 로짓 텐서.

        Returns:
            온도 스케일링된 로짓.
        """
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
        """검증 세트 NLL을 최소화하는 온도를 학습한다.

        Args:
            logits: 검증 세트 로짓 배열.
            y_true: 검증 세트 레이블 배열 (0 또는 1).
            max_iter: LBFGS 최대 반복 횟수.

        Returns:
            학습된 온도 파라미터 값.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        x = torch.tensor(logits, dtype=torch.float32, device=device)
        y = torch.tensor(y_true, dtype=torch.float32, device=device)
        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self.forward(x), y)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=0.1)  # 온도 최솟값 보장
        return float(self.temperature.detach().cpu().item())


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """모델 출력과 배치 타겟으로부터 총 손실을 계산한다.

    소프트 레이블이 있으면 소프트 레이블을, 없으면 하드 레이블을 사용한다.

    Args:
        outputs: DualViewPhysicsModel.forward()의 출력 딕셔너리.
        batch: DataLoader 배치 딕셔너리.
        cfg: 학습 설정 (손실 가중치 포함).

    Returns:
        (총 손실 텐서, 각 손실 항목 딕셔너리) 튜플.
    """
    label = batch["label"]
    hard_target = label
    soft_target = batch["soft_target"]
    # 소프트 타겟이 있으면 소프트, 없으면 하드 사용; 레이블 없는 샘플은 0으로 처리
    target = torch.where(torch.isnan(soft_target), hard_target, soft_target)
    target = torch.where(label < 0, torch.zeros_like(target), target)

    # 주 분류 손실 (레이블이 있는 샘플만)
    valid_main = label >= 0
    if valid_main.any():
        loss_main = F.binary_cross_entropy_with_logits(
            outputs["logit"][valid_main], target[valid_main]
        )
    else:
        loss_main = outputs["logit"].sum() * 0.0

    # 모션 회귀 손실: [max_diff_first, mean_diff_prev] 정규화 후 Smooth L1
    motion_tgt = torch.stack([batch["max_diff_first"], batch["mean_diff_prev"]], dim=1)
    valid_motion = ~torch.isnan(motion_tgt).any(dim=1)
    if valid_motion.any():
        pred = outputs["motion_reg"][valid_motion]
        tgt = motion_tgt[valid_motion]
        # 손실 안정화를 위한 약한 정규화
        tgt_norm = torch.stack(
            [tgt[:, 0] / 10.0, tgt[:, 1] / 0.15], dim=1
        ).clamp(min=0.0, max=2.0)
        loss_motion = F.smooth_l1_loss(pred, tgt_norm)
    else:
        loss_motion = outputs["motion_reg"].sum() * 0.0

    # 온셋 분류 손실
    onset = batch["onset_bucket"]
    valid_onset = onset >= 0
    if valid_onset.any():
        loss_onset = F.cross_entropy(outputs["onset_logit"][valid_onset], onset[valid_onset])
    else:
        loss_onset = outputs["onset_logit"].sum() * 0.0

    # 심각도 분류 손실
    sev = batch["severity_bucket"]
    valid_sev = sev >= 0
    if valid_sev.any():
        loss_sev = F.cross_entropy(outputs["severity_logit"][valid_sev], sev[valid_sev])
    else:
        loss_sev = outputs["severity_logit"].sum() * 0.0

    # 지지면 회귀 손실 (기하학 브랜치)
    if cfg.use_geometry_reasoning and "support_reg" in outputs:
        support_tgt = batch["support_target"]
        loss_support = F.smooth_l1_loss(outputs["support_reg"], support_tgt)
    else:
        loss_support = outputs["logit"].sum() * 0.0

    # 붕괴 여유도 회귀 손실 (기하학 브랜치)
    if cfg.use_geometry_reasoning and "margin_reg" in outputs:
        margin_tgt = batch["collapse_margin"]
        loss_margin = F.smooth_l1_loss(outputs["margin_reg"], margin_tgt)
    else:
        loss_margin = outputs["logit"].sum() * 0.0

    # 도메인 분류 손실 (도메인 적응 브랜치)
    if cfg.use_domain_head and "domain_logit" in outputs:
        dom = batch["source_domain"]
        valid_dom = dom >= 0
        if valid_dom.any():
            loss_dom = F.cross_entropy(outputs["domain_logit"][valid_dom], dom[valid_dom])
        else:
            loss_dom = outputs["domain_logit"].sum() * 0.0
    else:
        loss_dom = outputs["logit"].sum() * 0.0

    # 가중합 총 손실
    loss = (
        loss_main
        + cfg.aux_motion_weight * loss_motion
        + cfg.aux_onset_weight * loss_onset
        + cfg.aux_severity_weight * loss_sev
        + cfg.aux_support_weight * loss_support
        + cfg.aux_margin_weight * loss_margin
        + cfg.domain_weight * loss_dom
    )

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "loss_main": float(loss_main.detach().cpu().item()),
        "loss_motion": float(loss_motion.detach().cpu().item()),
        "loss_onset": float(loss_onset.detach().cpu().item()),
        "loss_sev": float(loss_sev.detach().cpu().item()),
        "loss_support": float(loss_support.detach().cpu().item()),
        "loss_margin": float(loss_margin.detach().cpu().item()),
        "loss_dom": float(loss_dom.detach().cpu().item()),
    }
    return loss, metrics


# ============================================================
# Train / Eval Loops
# ============================================================

def _move_batch_to_device(
    batch: Dict[str, torch.Tensor | str],
    device: torch.device,
) -> Dict[str, torch.Tensor | str]:
    """배치 딕셔너리의 텐서를 지정 디바이스로 이동한다.

    Args:
        batch: 배치 딕셔너리.
        device: 목표 디바이스.

    Returns:
        텐서가 이동된 배치 딕셔너리.
    """
    out: Dict[str, torch.Tensor | str] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=use_non_blocking(device))
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_passes: int = 1,
    desc: str = "predict",
) -> pd.DataFrame:
    """DataLoader 전체에 대해 예측 확률을 계산한다.

    Args:
        model: 추론할 모델.
        loader: 데이터 로더.
        device: 연산 디바이스.
        tta_passes: TTA 반복 횟수 (1이면 단일 추론).
        desc: tqdm 진행바 설명.

    Returns:
        'id'와 'pred' 컬럼을 가진 예측 DataFrame.
    """
    model.eval()
    rows: List[Dict[str, float | str]] = []
    progress = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        ids = batch["id"]
        probs_accum: Optional[torch.Tensor] = None
        batch_dev = _move_batch_to_device(batch, device)
        for _ in range(tta_passes):
            out = model(
                batch_dev["front"], batch_dev["top"], batch_dev["geom_feat"], grl_lambda=0.0
            )
            probs = torch.sigmoid(out["logit"]).detach().cpu()
            probs_accum = probs if probs_accum is None else probs_accum + probs
        probs_accum = probs_accum / float(tta_passes)  # TTA 평균
        for sid, p in zip(ids, probs_accum.numpy().tolist()):
            rows.append({"id": sid, "pred": float(p)})
    return pd.DataFrame(rows)


def fit_one_fold(
    cfg: TrainConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    split_map: Dict[str, str],
    fold_dir: Path,
) -> Dict[str, float]:
    """단일 폴드 학습을 수행하고 온도 스케일링 후 결과를 저장한다.

    Args:
        cfg: 학습 설정.
        train_df: 학습 DataFrame.
        valid_df: 검증 DataFrame.
        split_map: 샘플 ID → 분할명 매핑.
        fold_dir: 폴드 결과 저장 디렉토리.

    Returns:
        valid_logloss, valid_auc, temperature 키를 가진 메트릭 딕셔너리.
    """
    device = get_runtime_device()
    optimize_runtime_for_device(device)

    # 변환 파이프라인 생성
    front_train_tf = build_train_transform("front", cfg.image_size)
    top_train_tf = build_train_transform("top", cfg.image_size)
    front_valid_tf = build_valid_transform("front", cfg.image_size)
    top_valid_tf = build_valid_transform("top", cfg.image_size)
    pin_memory = use_pin_memory(device)

    # 데이터셋 및 로더 생성
    train_ds = DualViewDataset(
        cfg.data_root,
        train_df,
        split_map,
        front_train_tf,
        top_train_tf,
        training=True,
        checkerboard_top_normalize=cfg.checkerboard_top_normalize,
    )
    valid_ds = DualViewDataset(
        cfg.data_root,
        valid_df,
        split_map,
        front_valid_tf,
        top_valid_tf,
        training=False,
        checkerboard_top_normalize=cfg.checkerboard_top_normalize,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=max(cfg.batch_size, 8),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    # 모델 생성
    model = DualViewPhysicsModel(
        cfg.backbone,
        pretrained=cfg.pretrained,
        use_domain_head=cfg.use_domain_head,
        use_geometry_reasoning=cfg.use_geometry_reasoning,
    ).to(device)

    # DINOv2: 백본과 헤드를 다른 학습률로 분리
    is_dinov2 = cfg.backbone.lower().startswith("dinov2_")
    if is_dinov2:
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if "front_encoder.backbone" in name or "top_encoder.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": cfg.backbone_lr},  # 백본 느린 학습률
            {"params": head_params, "lr": cfg.lr},               # 헤드 빠른 학습률
        ]
    else:
        param_groups = list(model.parameters())

    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 웜업 + 코사인 스케줄러 구성
    if cfg.warmup_epochs > 0 and cfg.epochs > cfg.warmup_epochs:
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[cfg.warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    best_score = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(cfg.epochs):
        model.train()
        train_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"train {epoch + 1}/{cfg.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in train_bar:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            # GRL 람다: 에폭에 비례하여 0→1로 증가
            grl_lambda = (
                min(epoch / max(cfg.epochs - 1, 1), 1.0) if cfg.use_domain_head else 0.0
            )
            with torch.amp.autocast(
                device_type="cuda", enabled=cfg.use_amp and device.type == "cuda"
            ):
                outputs = model(
                    batch["front"], batch["top"], batch["geom_feat"], grl_lambda=grl_lambda
                )
                loss, _ = compute_losses(outputs, batch, cfg)
            if hasattr(train_bar, "set_postfix"):
                train_bar.set_postfix(loss=f"{loss.detach().cpu().item():.4f}")
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # 결정적 변환으로 검증 수행
        valid_pred = predict_loader(
            model, valid_loader, device=device, tta_passes=1,
            desc=f"valid {epoch + 1}/{cfg.epochs}",
        )
        y_true = valid_df["label_int"].values
        y_pred = valid_pred.sort_values("id")["pred"].values
        # id 기준 정렬로 안전하게 병합
        merged = valid_df[["id", "label_int"]].merge(valid_pred, on="id", how="left")
        ll = log_loss(merged["label_int"].values, merged["pred"].values, labels=[0, 1])
        if len(set(merged["label_int"].values)) < 2:
            auc = 0.5  # 단일 클래스면 AUC 정의 불가
        else:
            auc = roc_auc_score(merged["label_int"].values, merged["pred"].values)
        if ll < best_score:
            best_score = ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch+1:02d} valid_logloss={ll:.6f} valid_auc={auc:.6f}")

    assert best_state is not None
    model.load_state_dict(best_state)

    # 검증 로짓으로 온도 스케일링 캘리브레이션
    model.eval()
    logits_rows: List[Tuple[str, float]] = []
    with torch.no_grad():
        for batch in valid_loader:
            ids = batch["id"]
            batch = _move_batch_to_device(batch, device)
            out = model(batch["front"], batch["top"], batch["geom_feat"], grl_lambda=0.0)
            logit = out["logit"].detach().cpu().numpy()
            logits_rows.extend(list(zip(ids, logit.tolist())))
    logits_df = pd.DataFrame(logits_rows, columns=["id", "logit"])
    merged = valid_df[["id", "label_int"]].merge(logits_df, on="id", how="left")
    calibrator = TemperatureScaler()
    temp = calibrator.fit(
        merged["logit"].values.astype(np.float32),
        merged["label_int"].values.astype(np.float32),
    )
    cal_prob = sigmoid_np(merged["logit"].values / temp)
    ll_cal = log_loss(merged["label_int"].values, cal_prob, labels=[0, 1])
    if len(set(merged["label_int"].values)) < 2:
        auc_cal = 0.5
    else:
        auc_cal = roc_auc_score(merged["label_int"].values, cal_prob)

    # 체크포인트 및 메타데이터 저장
    torch.save(best_state, fold_dir / "best_model.pt")
    with open(fold_dir / "temperature.json", "w", encoding="utf-8") as f:
        json.dump(
            {"temperature": temp, "valid_logloss": ll_cal, "valid_auc": auc_cal},
            f, ensure_ascii=False, indent=2,
        )

    # OOF 예측 저장
    oof = valid_df[["id", "label_int", "source_domain"]].merge(
        merged[["id", "logit"]], on="id", how="left"
    )
    oof["pred_raw"] = sigmoid_np(merged["logit"].values)  # 캘리브레이션 전 확률
    oof["pred_cal"] = cal_prob                             # 온도 스케일링 후 확률
    oof.to_csv(fold_dir / "oof_valid.csv", index=False)

    return {"valid_logloss": float(ll_cal), "valid_auc": float(auc_cal), "temperature": float(temp)}


# ============================================================
# Data Assembly and Fold Generation
# ============================================================

def prepare_tables(
    data_root: str | Path,
    motion_csv: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """학습·검증·테스트 DataFrame을 준비하고 보조 열을 추가한다.

    Args:
        data_root: 데이터셋 루트 경로.
        motion_csv: 모션 타겟 CSV 경로 (None이거나 파일 없으면 미병합).

    Returns:
        (train_df, dev_df, test_df) 튜플.
    """
    data_root = Path(data_root)
    train_df = read_csv(data_root / "train.csv")
    dev_df = read_csv(data_root / "dev.csv")
    test_df = read_csv(data_root / "sample_submission.csv")

    # 정수 레이블 및 도메인 레이블 추가
    train_df["label_int"] = train_df["label"].map(label_to_int).astype(int)
    dev_df["label_int"] = dev_df["label"].map(label_to_int).astype(int)
    train_df["source_domain"] = 0   # 학습 도메인
    dev_df["source_domain"] = 1     # 검증 도메인
    test_df["source_domain"] = -1   # 테스트 (도메인 레이블 없음)

    # 모션 타겟 병합 (존재하는 경우)
    if motion_csv is not None and Path(motion_csv).exists():
        motion_df = pd.read_csv(motion_csv)
        train_df = train_df.merge(
            motion_df.drop(columns=["label"], errors="ignore"),
            on=["id", "label_int"],
            how="left",
        )
    return train_df, dev_df, test_df


def make_split_map(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """샘플 ID → 분할명 매핑 딕셔너리를 생성한다.

    Args:
        train_df: 학습 DataFrame.
        dev_df: 검증 DataFrame.
        test_df: 테스트 DataFrame (None이면 포함 안 함).

    Returns:
        {샘플ID: 분할명} 딕셔너리.
    """
    split_map = {sid: "train" for sid in train_df["id"].tolist()}
    split_map.update({sid: "dev" for sid in dev_df["id"].tolist()})
    if test_df is not None:
        split_map.update({sid: "test" for sid in test_df["id"].tolist()})
    return split_map


# ============================================================
# Workflows
# ============================================================

def run_design_holdout(cfg: TrainConfig) -> None:
    """설계 단계: 훈련 데이터로만 학습하고 dev로 검증한다.

    아키텍처·증강·손실 설계를 확정하기 위해 사용한다.
    완료 후 design_metrics.json과 dev_logloss_report.json을 저장한다.

    Args:
        cfg: 학습 설정.
    """
    out_dir = ensure_dir(cfg.out_dir)
    train_df, dev_df, _ = prepare_tables(cfg.data_root, cfg.motion_csv)
    split_map = make_split_map(train_df, dev_df)
    metrics = fit_one_fold(cfg, train_df, dev_df, split_map, out_dir)
    dev_report = print_dev_logloss_report("train-design", float(metrics["valid_logloss"]))
    with open(out_dir / "design_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "dev_logloss_report.json", "w", encoding="utf-8") as f:
        json.dump(dev_report, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_pooled_grouped_cv(cfg: TrainConfig) -> None:
    """최종 단계: 훈련+검증 풀링, 기하 클러스터 그룹 CV, 폴드 앙상블.

    설계 확정 후 실행한다. 완료 후 cv_summary.csv와 oof_all.csv를 저장한다.

    Args:
        cfg: 학습 설정.
    """
    out_dir = ensure_dir(cfg.out_dir)
    train_df, dev_df, test_df = prepare_tables(cfg.data_root, cfg.motion_csv)
    pooled = pd.concat([train_df, dev_df], ignore_index=True)  # 풀링된 전체 데이터

    # 기하 클러스터링으로 그룹 레이블 생성
    cluster_df = build_geometry_clusters(cfg.data_root, train_df, dev_df, ClusterConfig())
    pooled = pooled.merge(cluster_df, on="id", how="left")

    # 레이블+도메인 조합으로 층화, 기하 클러스터로 그룹 분리
    y = pooled["label_int"].astype(str) + "_" + pooled["source_domain"].astype(str)
    groups = pooled["geometry_cluster"].astype(int).values

    sgkf = StratifiedGroupKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
    split_map = make_split_map(train_df, dev_df, test_df)

    summary_rows: List[Dict[str, float | int]] = []
    oof_all: List[pd.DataFrame] = []
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(pooled, y, groups), start=1):
        fold_dir = ensure_dir(out_dir / f"fold_{fold}")
        tr_df = pooled.iloc[tr_idx].reset_index(drop=True)
        va_df = pooled.iloc[va_idx].reset_index(drop=True)
        print(f"\n===== Fold {fold}/{cfg.num_folds} =====")
        metrics = fit_one_fold(cfg, tr_df, va_df, split_map, fold_dir)
        metrics["fold"] = fold
        summary_rows.append(metrics)
        oof_df = pd.read_csv(fold_dir / "oof_valid.csv")
        oof_df["fold"] = fold
        oof_all.append(oof_df)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # VRAM 해제

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    oof_all_df = pd.concat(oof_all, ignore_index=True)
    oof_all_df.to_csv(out_dir / "oof_all.csv", index=False)

    # dev 도메인 OOF logloss 보고
    dev_oof = oof_all_df[oof_all_df["source_domain"] == 1].copy()
    if not dev_oof.empty:
        dev_oof_logloss = log_loss(
            dev_oof["label_int"].values, dev_oof["pred_cal"].values, labels=[0, 1]
        )
        dev_report = print_dev_logloss_report("cv-train", float(dev_oof_logloss))
        with open(out_dir / "dev_oof_logloss_report.json", "w", encoding="utf-8") as f:
            json.dump(dev_report, f, ensure_ascii=False, indent=2)
    print(summary)


def run_full_pipeline(cfg: TrainConfig, refresh_motion: bool = False) -> None:
    """모션 추출 → 풀링 CV → 제출파일 생성의 전체 파이프라인을 실행한다.

    Args:
        cfg: 학습 설정.
        refresh_motion: True이면 기존 모션 CSV를 무시하고 재추출.
    """
    motion_csv = (
        Path(cfg.motion_csv) if cfg.motion_csv is not None
        else default_motion_csv(cfg.data_root)
    )
    if refresh_motion or not motion_csv.exists():
        print(f"Extracting motion targets -> {motion_csv}")
        extract_motion_targets(cfg.data_root, motion_csv, MotionExtractionConfig())
    else:
        print(f"Using existing motion targets: {motion_csv}")

    full_cfg = dataclasses.replace(cfg, motion_csv=str(motion_csv))
    run_pooled_grouped_cv(full_cfg)
    make_submission(full_cfg, full_cfg.out_dir)


def make_submission(cfg: TrainConfig, run_dir: str | Path) -> None:
    """폴드 모델들로 테스트 예측을 앙상블하여 제출 파일을 생성한다.

    품질 필터링을 통해 이상 폴드를 제외하고, 모두 제외되면 전체 폴드를 사용한다.
    DACON 리더보드 형식(id, unstable_prob, stable_prob)으로 저장한다.

    Args:
        cfg: 학습 설정.
        run_dir: 폴드 디렉토리가 있는 실행 결과 경로.
    """
    run_dir = Path(run_dir)
    train_df, dev_df, test_df = prepare_tables(cfg.data_root, cfg.motion_csv)
    split_map = make_split_map(train_df, dev_df, test_df)
    device = get_runtime_device()
    pin_memory = use_pin_memory(device)

    front_tf = build_valid_transform("front", cfg.image_size)
    top_tf = build_valid_transform("top", cfg.image_size)
    test_ds = DualViewDataset(
        cfg.data_root,
        test_df,
        split_map,
        front_tf,
        top_tf,
        training=False,
        checkerboard_top_normalize=cfg.checkerboard_top_normalize,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(cfg.batch_size, 8),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {run_dir}")

    pred_frames: List[pd.DataFrame] = []
    skipped_folds: List[str] = []
    for fold_dir in fold_dirs:
        # 품질 필터링: 온도 이상이나 logloss 과다 폴드 제외
        temp_path = fold_dir / "temperature.json"
        if temp_path.exists():
            with open(temp_path, "r", encoding="utf-8") as f:
                temp_info = json.load(f)
            fold_temp = temp_info.get("temperature", 1.0)
            fold_ll = temp_info.get("valid_logloss", 0.0)
            if fold_temp <= 0 or fold_ll > 2.0:  # 온도 이상 또는 logloss 2.0 초과
                print(
                    f"[WARNING] Skipping {fold_dir.name}: "
                    f"temperature={fold_temp:.4f}, valid_logloss={fold_ll:.4f}"
                )
                skipped_folds.append(fold_dir.name)
                continue

        model = DualViewPhysicsModel(
            cfg.backbone,
            pretrained=False,
            use_domain_head=cfg.use_domain_head,
            use_geometry_reasoning=cfg.use_geometry_reasoning,
        ).to(device)
        state = torch.load(fold_dir / "best_model.pt", map_location=device)
        model.load_state_dict(state)
        preds = predict_loader(
            model, test_loader, device, tta_passes=cfg.tta_passes,
            desc=f"{fold_dir.name} test",
        )
        if temp_path.exists():
            with open(temp_path, "r", encoding="utf-8") as f:
                temp = json.load(f).get("temperature", 1.0)
            # 확률 → 로짓 → 온도 스케일링 → 재확률
            p = preds["pred"].clip(1e-6, 1 - 1e-6).values
            logit = np.log(p / (1 - p))
            preds["pred"] = sigmoid_np(logit / temp)
        preds = preds.rename(columns={"pred": f"pred_{fold_dir.name}"})
        pred_frames.append(preds)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 폴백: 모든 폴드가 필터링되면 필터 없이 전체 재실행
    if not pred_frames:
        print("[WARNING] All folds were filtered out. Using all folds without quality filtering.")
        skipped_folds.clear()
        for fold_dir in fold_dirs:
            model = DualViewPhysicsModel(
                cfg.backbone,
                pretrained=False,
                use_domain_head=cfg.use_domain_head,
                use_geometry_reasoning=cfg.use_geometry_reasoning,
            ).to(device)
            state = torch.load(fold_dir / "best_model.pt", map_location=device)
            model.load_state_dict(state)
            preds = predict_loader(
                model, test_loader, device, tta_passes=cfg.tta_passes,
                desc=f"{fold_dir.name} test",
            )
            temp_path = fold_dir / "temperature.json"
            if temp_path.exists():
                with open(temp_path, "r", encoding="utf-8") as f:
                    temp = json.load(f).get("temperature", 1.0)
                p = preds["pred"].clip(1e-6, 1 - 1e-6).values
                logit = np.log(p / (1 - p))
                preds["pred"] = sigmoid_np(logit / max(temp, 0.1))  # 최소 온도 0.1 보장
            preds = preds.rename(columns={"pred": f"pred_{fold_dir.name}"})
            pred_frames.append(preds)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if skipped_folds:
        print(f"[INFO] Used {len(pred_frames)} folds, skipped {len(skipped_folds)}: {skipped_folds}")

    # 폴드 예측 평균 앙상블
    sub = test_df[["id"]].copy()
    for pf in pred_frames:
        sub = sub.merge(pf, on="id", how="left")
    pred_cols = [c for c in sub.columns if c.startswith("pred_")]
    sub["prob_unstable"] = sub[pred_cols].mean(axis=1)

    # DACON 리더보드 형식: id, unstable_prob, stable_prob
    out = test_df[["id"]].copy()
    out["unstable_prob"] = sub["prob_unstable"]
    out["stable_prob"] = 1.0 - sub["prob_unstable"]

    out.to_csv(run_dir / "submission.csv", index=False)
    print(f"Saved: {run_dir / 'submission.csv'}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱하여 Namespace로 반환한다.

    서브커맨드: extract-motion, train-design, cv-train, make-submission, full-run.

    Returns:
        파싱된 인자 Namespace.
    """
    parser = argparse.ArgumentParser(description="Physics-aware dual-view solution")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-motion")
    p_extract.add_argument("--data-root", type=str, default=None)
    p_extract.add_argument("--out-csv", type=str, default=None)

    def add_train_args(p: argparse.ArgumentParser, default_out_dir: Path) -> None:
        """학습 서브커맨드에 공통 인자를 추가한다.

        Args:
            p: 대상 ArgumentParser.
            default_out_dir: 기본 출력 디렉토리 경로.
        """
        p.add_argument("--data-root", type=str, default=None)
        p.add_argument("--out-dir", type=str, default=str(default_out_dir))
        p.add_argument("--motion-csv", type=str, default=None)
        p.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=SUPPORTED_BACKBONES)
        p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
        p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
        p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--num-workers", type=int, default=default_num_workers())
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--backbone-lr", type=float, default=1e-5)
        p.add_argument("--warmup-epochs", type=int, default=3)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--num-folds", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--use-domain-head", action="store_true")
        p.add_argument("--enable-geometry-reasoning", action="store_true")
        p.add_argument("--no-amp", action="store_true")
        p.add_argument("--tta-passes", type=int, default=4)
        p.add_argument("--no-checkerboard-top-normalize", action="store_true")

    p_design = sub.add_parser("train-design")
    add_train_args(p_design, default_runs_dir("design"))

    p_cv = sub.add_parser("cv-train")
    add_train_args(p_cv, default_runs_dir("final"))

    p_sub = sub.add_parser("make-submission")
    add_train_args(p_sub, default_runs_dir("final"))
    p_sub.add_argument("--run-dir", type=str, default=None)

    p_full = sub.add_parser("full-run")
    add_train_args(p_full, default_runs_dir("final"))
    p_full.add_argument("--refresh-motion", action="store_true")

    return parser.parse_args()


def namespace_to_cfg(ns: argparse.Namespace, data_root: Path) -> TrainConfig:
    """파싱된 Namespace를 TrainConfig 데이터클래스로 변환한다.

    Args:
        ns: argparse.Namespace 객체.
        data_root: 확정된 데이터셋 루트 경로.

    Returns:
        TrainConfig 데이터클래스 인스턴스.
    """
    return TrainConfig(
        data_root=str(data_root),
        out_dir=str(Path(ns.out_dir).expanduser().resolve()),
        motion_csv=str(resolve_motion_csv(data_root, ns.motion_csv)),
        backbone=ns.backbone,
        pretrained=bool(ns.pretrained),
        image_size=ns.image_size,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        epochs=ns.epochs,
        lr=ns.lr,
        backbone_lr=ns.backbone_lr,
        warmup_epochs=ns.warmup_epochs,
        weight_decay=ns.weight_decay,
        num_folds=ns.num_folds,
        seed=ns.seed,
        use_domain_head=bool(ns.use_domain_head),
        use_geometry_reasoning=bool(ns.enable_geometry_reasoning),
        use_amp=not bool(ns.no_amp),
        tta_passes=ns.tta_passes,
        checkerboard_top_normalize=not bool(ns.no_checkerboard_top_normalize),
    )


def validate_cfg(cfg: TrainConfig) -> None:
    """TrainConfig의 유효성을 검증한다.

    Args:
        cfg: 검증할 학습 설정.

    Raises:
        ValueError: 이미지 크기가 백본 요구사항을 만족하지 않을 때.
    """
    validate_image_size(cfg.backbone, cfg.image_size)


def main() -> None:
    """CLI 진입점.

    인자를 파싱하여 extract-motion / train-design / cv-train /
    make-submission / full-run 중 해당 워크플로를 실행한다.
    """
    args = parse_args()
    data_root = resolve_data_root(getattr(args, "data_root", None))

    if args.command == "extract-motion":
        set_seed(42)
        out_csv = resolve_motion_csv(data_root, args.out_csv)
        print(f"Resolved data_root: {data_root}")
        print(f"Saving motion targets to: {out_csv}")
        out_df = extract_motion_targets(data_root, out_csv, MotionExtractionConfig())
        print(out_df.head())
        print(out_df.groupby("label")[["max_diff_first", "mean_diff_first", "mean_diff_prev"]].mean())
        return

    cfg = namespace_to_cfg(args, data_root)
    validate_cfg(cfg)
    set_seed(cfg.seed)
    runtime_device = get_runtime_device()
    optimize_runtime_for_device(runtime_device)
    ensure_dir(cfg.out_dir)
    print(f"Resolved data_root: {cfg.data_root}")
    print(f"Resolved out_dir: {cfg.out_dir}")
    if cfg.motion_csv is not None:
        print(f"Resolved motion_csv: {cfg.motion_csv}")
    print(f"Runtime device: {describe_runtime_device(runtime_device)}")
    print(f"Checkerboard top normalization: {cfg.checkerboard_top_normalize}")
    print(f"Geometry reasoning branch: {cfg.use_geometry_reasoning}")

    if args.command == "train-design":
        run_design_holdout(cfg)
    elif args.command == "cv-train":
        run_pooled_grouped_cv(cfg)
    elif args.command == "make-submission":
        run_dir = (
            Path(args.run_dir).expanduser().resolve()
            if args.run_dir is not None
            else Path(cfg.out_dir)
        )
        print(f"Resolved run_dir: {run_dir}")
        make_submission(cfg, run_dir)
    elif args.command == "full-run":
        run_full_pipeline(cfg, refresh_motion=bool(args.refresh_motion))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
