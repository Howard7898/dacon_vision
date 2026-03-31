# =============================================================================
# geometry_reasoning.py
# -----------------------------------------------------------------------------
# 작성자  : (공개 생략)
# 인코딩  : UTF-8
# Python  : 3.10+
# 설명    : 전면·상단 뷰 기하학적 특징 추출 및 붕괴 여유도(Collapse Margin) 계산
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import cv2
import numpy as np


# ============================================================
# 상수 정의
# ============================================================

GEOMETRY_FEATURE_NAMES = [
    "top_area_frac",           # 상단 뷰 전경 면적 비율
    "top_support_width_frac",  # 상단 뷰 지지 너비 비율
    "top_support_height_frac", # 상단 뷰 지지 높이 비율
    "top_fill_ratio",          # 상단 뷰 바운딩박스 대비 채움 비율
    "top_centroid_dx",         # 상단 뷰 무게중심 x 편차 (정규화)
    "top_centroid_dy",         # 상단 뷰 무게중심 y 편차 (정규화)
    "front_height_frac",       # 전면 뷰 높이 비율
    "front_width_frac",        # 전면 뷰 너비 비율
    "front_slenderness",       # 전면 뷰 세장비 (높이/너비)
    "front_base_width_frac",   # 전면 뷰 하단 20% 영역 너비
    "front_top_width_frac",    # 전면 뷰 상단 25% 영역 너비
    "front_centroid_dx",       # 전면 뷰 무게중심 x 편차 (정규화)
    "front_tilt",              # 전면 뷰 기울기 (상단 중심 - 하단 중심)
    "front_top_heaviness",     # 전면 뷰 상반부 픽셀 비율 (무거울수록 불안정)
]


# ============================================================
# 설정 데이터클래스
# ============================================================

@dataclass(frozen=True)
class GeometryReasoningConfig:
    """기하학적 특징 추출 설정.

    Attributes:
        min_component_area_ratio: 유효 연결 성분 최소 면적 비율 (전체 면적 대비).
    """

    min_component_area_ratio: float = 0.002  # 전체 이미지 면적의 0.2% 미만 성분은 무시


# ============================================================
# 전경 마스크 추정
# ============================================================

def estimate_foreground_mask(rgb: np.ndarray, cfg: GeometryReasoningConfig) -> np.ndarray:
    """RGB 이미지에서 전경 마스크를 추정한다.

    채도·밝기·명도 퍼센타일 기반 이진화 후 가장 큰 유효 연결 성분을 선택한다.

    Args:
        rgb: (H, W, 3) RGB 이미지 배열.
        cfg: 추출 설정.

    Returns:
        (H, W) 이진 마스크 (uint8, 0 또는 1).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]  # 채도 채널
    val = hsv[:, :, 2]  # 명도 채널

    s_thr = float(np.percentile(sat, 60.0))   # 채도 상위 40% 기준
    g_thr = float(np.percentile(gray, 45.0))  # 밝기 하위 45% 기준
    v_thr = float(np.percentile(val, 35.0))   # 명도 하위 35% 기준

    # 채도가 높거나, 밝기가 낮거나, 명도가 낮은 픽셀 = 전경
    mask = ((sat > s_thr) | (gray < g_thr) | (val < v_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)   # 작은 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)  # 내부 구멍 메우기

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return (mask > 0).astype(np.uint8)

    h, w = gray.shape
    min_area = int(cfg.min_component_area_ratio * h * w)  # 유효 성분 최소 면적 (픽셀)
    best_idx = 1
    best_area = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < min_area or ww < 8 or hh < 8 or area >= 0.995 * h * w:
            continue  # 너무 작거나 너무 큰 성분 스킵
        if area > best_area:
            best_idx = i
            best_area = area
    return (labels == best_idx).astype(np.uint8)


# ============================================================
# 내부 유틸리티 함수
# ============================================================

def _safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """마스크의 바운딩 박스를 안전하게 반환한다.

    전경 픽셀이 없으면 이미지 전체를 반환한다.

    Args:
        mask: (H, W) 이진 마스크.

    Returns:
        (x_min, y_min, x_max, y_max) 바운딩 박스 좌표.
    """
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1  # 전경 없으면 전체 이미지 반환
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _band_stats(mask: np.ndarray, y1: int, y2: int) -> tuple[float, float]:
    """마스크의 수평 띠(band) 영역에서 너비와 중심 x 좌표를 계산한다.

    Args:
        mask: (H, W) 이진 마스크.
        y1: 띠 상단 y 좌표 (포함).
        y2: 띠 하단 y 좌표 (미포함).

    Returns:
        width: 전경 픽셀의 x 범위를 이미지 너비로 정규화한 값.
        center: 전경 픽셀의 평균 x 좌표를 이미지 너비로 정규화한 값.
    """
    band = mask[max(0, y1):max(y1 + 1, y2), :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    width = float(xs.max() - xs.min() + 1) / float(mask.shape[1])
    center = float(xs.mean() / max(mask.shape[1] - 1, 1))
    return width, center


# ============================================================
# 기하학적 특징 추출
# ============================================================

def extract_geometry_features(
    front_rgb: np.ndarray,
    top_rgb: np.ndarray,
    cfg: GeometryReasoningConfig,
) -> Dict[str, float]:
    """전면·상단 뷰 이미지에서 기하학적 특징을 추출한다.

    Args:
        front_rgb: (H, W, 3) 전면 뷰 RGB 이미지.
        top_rgb: (H, W, 3) 상단 뷰 RGB 이미지.
        cfg: 추출 설정.

    Returns:
        GEOMETRY_FEATURE_NAMES에 정의된 키로 구성된 특징 딕셔너리.
    """
    top_mask = estimate_foreground_mask(top_rgb, cfg)
    front_mask = estimate_foreground_mask(front_rgb, cfg)

    th, tw = top_mask.shape    # 상단 뷰 이미지 크기
    fh, fw = front_mask.shape  # 전면 뷰 이미지 크기

    # 바운딩 박스 추출
    tx1, ty1, tx2, ty2 = _safe_bbox(top_mask)
    fx1, fy1, fx2, fy2 = _safe_bbox(front_mask)

    # 상단 뷰 특징 계산
    top_area = float(top_mask.mean())                                        # 전경 면적 비율
    top_bbox_area = max(1.0, float((tx2 - tx1 + 1) * (ty2 - ty1 + 1)))
    top_fill_ratio = float(top_mask.sum() / top_bbox_area)                   # 바운딩박스 채움 비율
    top_width_frac = float((tx2 - tx1 + 1) / max(tw, 1))                    # 지지 너비 비율
    top_height_frac = float((ty2 - ty1 + 1) / max(th, 1))                   # 지지 높이 비율

    top_ys, top_xs = np.where(top_mask > 0)
    if len(top_xs) == 0:
        top_centroid_dx = 0.0
        top_centroid_dy = 0.0
    else:
        # 무게중심을 [-1, 1] 범위로 정규화
        top_centroid_dx = float((top_xs.mean() / max(tw - 1, 1) - 0.5) * 2.0)
        top_centroid_dy = float((top_ys.mean() / max(th - 1, 1) - 0.5) * 2.0)

    # 전면 뷰 특징 계산
    front_height_frac = float((fy2 - fy1 + 1) / max(fh, 1))                  # 높이 비율
    front_width_frac = float((fx2 - fx1 + 1) / max(fw, 1))                   # 너비 비율
    front_slenderness = float((fy2 - fy1 + 1) / max(fx2 - fx1 + 1, 1))       # 세장비 (높이/너비)

    front_ys, front_xs = np.where(front_mask > 0)
    if len(front_xs) == 0:
        front_centroid_dx = 0.0
    else:
        front_centroid_dx = float((front_xs.mean() / max(fw - 1, 1) - 0.5) * 2.0)

    # 전면 뷰 상단·하단 띠 통계 (기울기·하중 분포 추정)
    bbox_h = max(fy2 - fy1 + 1, 1)
    top_band_width, top_band_center = _band_stats(
        front_mask, fy1, fy1 + max(1, int(round(0.25 * bbox_h)))  # 상단 25% 띠
    )
    base_band_width, base_band_center = _band_stats(
        front_mask, fy2 - max(1, int(round(0.20 * bbox_h))) + 1, fy2 + 1  # 하단 20% 띠
    )
    mid_y = fy1 + bbox_h // 2
    top_pixels = float(front_mask[fy1:mid_y, :].sum())             # 상반부 전경 픽셀 수
    base_pixels = float(front_mask[mid_y:fy2 + 1, :].sum())        # 하반부 전경 픽셀 수
    top_heaviness = top_pixels / max(top_pixels + base_pixels, 1.0)  # 상반부 픽셀 비율
    front_tilt = float(top_band_center - base_band_center)           # 기울기 (양수=오른쪽)

    features = {
        "top_area_frac": top_area,
        "top_support_width_frac": top_width_frac,
        "top_support_height_frac": top_height_frac,
        "top_fill_ratio": top_fill_ratio,
        "top_centroid_dx": top_centroid_dx,
        "top_centroid_dy": top_centroid_dy,
        "front_height_frac": front_height_frac,
        "front_width_frac": front_width_frac,
        "front_slenderness": front_slenderness,
        "front_base_width_frac": base_band_width,
        "front_top_width_frac": top_band_width,
        "front_centroid_dx": front_centroid_dx,
        "front_tilt": front_tilt,
        "front_top_heaviness": top_heaviness,
    }
    return features


# ============================================================
# 붕괴 여유도 계산
# ============================================================

def collapse_margin_from_features(features: Dict[str, float]) -> float:
    """기하학적 특징으로부터 붕괴 여유도(Collapse Margin)를 계산한다.

    선형 가중합으로 구조 안정성 점수를 계산하고 [0, 1] 범위로 클리핑한다.
    값이 높을수록 안정적이다.

    Args:
        features: extract_geometry_features()가 반환한 특징 딕셔너리.

    Returns:
        붕괴 여유도 (float, 0.0~1.0).
    """
    raw = (
        1.20 * features["top_support_width_frac"]   # 넓은 지지면 → 안정
        + 0.90 * features["front_base_width_frac"]  # 넓은 하단 → 안정
        + 0.50 * features["top_fill_ratio"]         # 높은 채움률 → 안정
        - 0.75 * abs(features["top_centroid_dx"])   # 무게중심 편심 → 불안정
        - 0.55 * abs(features["front_tilt"])        # 기울기 → 불안정
        - 0.20 * features["front_slenderness"]      # 높은 세장비 → 불안정
        - 0.25 * features["front_top_heaviness"]    # 위쪽 무거움 → 불안정
    )
    return float(np.clip(0.5 + 0.35 * raw, 0.0, 1.0))  # 0~1 범위로 정규화


# ============================================================
# 특징 벡터 생성 함수
# ============================================================

def geometry_feature_vector(
    front_rgb: np.ndarray,
    top_rgb: np.ndarray,
    cfg: GeometryReasoningConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """전면·상단 뷰로부터 기하학적 특징 벡터와 붕괴 여유도를 반환한다.

    Args:
        front_rgb: (H, W, 3) 전면 뷰 RGB 이미지.
        top_rgb: (H, W, 3) 상단 뷰 RGB 이미지.
        cfg: 추출 설정.

    Returns:
        vec: 전체 특징 벡터 (float32, shape=(14,)).
        support: 지지면 관련 특징 벡터 (float32, shape=(2,)).
        margin: 붕괴 여유도 (float, 0.0~1.0).
    """
    features = extract_geometry_features(front_rgb, top_rgb, cfg)
    vec = np.asarray([features[name] for name in GEOMETRY_FEATURE_NAMES], dtype=np.float32)
    support = np.asarray(
        [
            features["top_support_width_frac"],  # 지지면 너비
            features["top_area_frac"],            # 지지면 면적
        ],
        dtype=np.float32,
    )
    margin = collapse_margin_from_features(features)
    return vec, support, float(margin)


# ============================================================
# 특징 캐시 클래스
# ============================================================

class GeometryFeatureCache:
    """샘플 ID를 키로 기하학적 특징을 캐싱하는 클래스.

    동일 샘플에 대한 중복 특징 추출을 방지한다.

    Attributes:
        cfg: 특징 추출 설정.
    """

    def __init__(self, cfg: GeometryReasoningConfig | None = None) -> None:
        """초기화.

        Args:
            cfg: 추출 설정 (None이면 기본값 사용).
        """
        self.cfg = cfg or GeometryReasoningConfig()
        self._cache: Dict[str, tuple[np.ndarray, np.ndarray, float]] = {}  # sid → (vec, support, margin)

    def get(
        self,
        sid: str,
        front_rgb: np.ndarray,
        top_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """캐시에서 특징을 조회하거나, 없으면 계산 후 캐싱한다.

        Args:
            sid: 샘플 고유 ID (캐시 키).
            front_rgb: (H, W, 3) 전면 뷰 RGB 이미지.
            top_rgb: (H, W, 3) 상단 뷰 RGB 이미지.

        Returns:
            (vec, support, margin) 튜플.
        """
        cached = self._cache.get(sid)
        if cached is not None:
            return cached
        value = geometry_feature_vector(front_rgb, top_rgb, self.cfg)
        self._cache[sid] = value
        return value
