from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


GEOMETRY_FEATURE_NAMES = [
    "top_area_frac",
    "top_support_width_frac",
    "top_support_height_frac",
    "top_fill_ratio",
    "top_centroid_dx",
    "top_centroid_dy",
    "front_height_frac",
    "front_width_frac",
    "front_slenderness",
    "front_base_width_frac",
    "front_top_width_frac",
    "front_centroid_dx",
    "front_tilt",
    "front_top_heaviness",
]


@dataclass(frozen=True)
class GeometryReasoningConfig:
    min_component_area_ratio: float = 0.002


def estimate_foreground_mask(rgb: np.ndarray, cfg: GeometryReasoningConfig) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    s_thr = float(np.percentile(sat, 60.0))
    g_thr = float(np.percentile(gray, 45.0))
    v_thr = float(np.percentile(val, 35.0))

    mask = ((sat > s_thr) | (gray < g_thr) | (val < v_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return (mask > 0).astype(np.uint8)

    h, w = gray.shape
    min_area = int(cfg.min_component_area_ratio * h * w)
    best_idx = 1
    best_area = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < min_area or ww < 8 or hh < 8 or area >= 0.995 * h * w:
            continue
        if area > best_area:
            best_idx = i
            best_area = area
    return (labels == best_idx).astype(np.uint8)


def _safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _band_stats(mask: np.ndarray, y1: int, y2: int) -> tuple[float, float]:
    band = mask[max(0, y1):max(y1 + 1, y2), :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    width = float(xs.max() - xs.min() + 1) / float(mask.shape[1])
    center = float(xs.mean() / max(mask.shape[1] - 1, 1))
    return width, center


def extract_geometry_features(front_rgb: np.ndarray, top_rgb: np.ndarray, cfg: GeometryReasoningConfig) -> Dict[str, float]:
    top_mask = estimate_foreground_mask(top_rgb, cfg)
    front_mask = estimate_foreground_mask(front_rgb, cfg)

    th, tw = top_mask.shape
    fh, fw = front_mask.shape

    tx1, ty1, tx2, ty2 = _safe_bbox(top_mask)
    fx1, fy1, fx2, fy2 = _safe_bbox(front_mask)

    top_area = float(top_mask.mean())
    top_bbox_area = max(1.0, float((tx2 - tx1 + 1) * (ty2 - ty1 + 1)))
    top_fill_ratio = float(top_mask.sum() / top_bbox_area)
    top_width_frac = float((tx2 - tx1 + 1) / max(tw, 1))
    top_height_frac = float((ty2 - ty1 + 1) / max(th, 1))

    top_ys, top_xs = np.where(top_mask > 0)
    if len(top_xs) == 0:
        top_centroid_dx = 0.0
        top_centroid_dy = 0.0
    else:
        top_centroid_dx = float((top_xs.mean() / max(tw - 1, 1) - 0.5) * 2.0)
        top_centroid_dy = float((top_ys.mean() / max(th - 1, 1) - 0.5) * 2.0)

    front_height_frac = float((fy2 - fy1 + 1) / max(fh, 1))
    front_width_frac = float((fx2 - fx1 + 1) / max(fw, 1))
    front_slenderness = float((fy2 - fy1 + 1) / max(fx2 - fx1 + 1, 1))

    front_ys, front_xs = np.where(front_mask > 0)
    if len(front_xs) == 0:
        front_centroid_dx = 0.0
    else:
        front_centroid_dx = float((front_xs.mean() / max(fw - 1, 1) - 0.5) * 2.0)

    bbox_h = max(fy2 - fy1 + 1, 1)
    top_band_width, top_band_center = _band_stats(front_mask, fy1, fy1 + max(1, int(round(0.25 * bbox_h))))
    base_band_width, base_band_center = _band_stats(front_mask, fy2 - max(1, int(round(0.20 * bbox_h))) + 1, fy2 + 1)
    mid_y = fy1 + bbox_h // 2
    top_pixels = float(front_mask[fy1:mid_y, :].sum())
    base_pixels = float(front_mask[mid_y:fy2 + 1, :].sum())
    top_heaviness = top_pixels / max(top_pixels + base_pixels, 1.0)
    front_tilt = float(top_band_center - base_band_center)

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


def collapse_margin_from_features(features: Dict[str, float]) -> float:
    raw = (
        1.20 * features["top_support_width_frac"]
        + 0.90 * features["front_base_width_frac"]
        + 0.50 * features["top_fill_ratio"]
        - 0.75 * abs(features["top_centroid_dx"])
        - 0.55 * abs(features["front_tilt"])
        - 0.20 * features["front_slenderness"]
        - 0.25 * features["front_top_heaviness"]
    )
    return float(np.clip(0.5 + 0.35 * raw, 0.0, 1.0))


def geometry_feature_vector(front_rgb: np.ndarray, top_rgb: np.ndarray, cfg: GeometryReasoningConfig) -> tuple[np.ndarray, np.ndarray, float]:
    features = extract_geometry_features(front_rgb, top_rgb, cfg)
    vec = np.asarray([features[name] for name in GEOMETRY_FEATURE_NAMES], dtype=np.float32)
    support = np.asarray(
        [
            features["top_support_width_frac"],
            features["top_area_frac"],
        ],
        dtype=np.float32,
    )
    margin = collapse_margin_from_features(features)
    return vec, support, float(margin)


class GeometryFeatureCache:
    def __init__(self, cfg: GeometryReasoningConfig | None = None) -> None:
        self.cfg = cfg or GeometryReasoningConfig()
        self._cache: Dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    def get(self, sid: str, front_rgb: np.ndarray, top_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        cached = self._cache.get(sid)
        if cached is not None:
            return cached
        value = geometry_feature_vector(front_rgb, top_rgb, self.cfg)
        self._cache[sid] = value
        return value
