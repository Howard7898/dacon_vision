from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CheckerboardTopNormConfig:
    enabled: bool = True
    ring_ratio: float = 0.10
    rot_line_min: int = 10
    rot_conf_min: float = 0.20
    pad_value: int = 128


def _estimate_mask(rgb: np.ndarray) -> np.ndarray:
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

    best = 1
    best_area = 0
    h, w = gray.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area > best_area and ww > 8 and hh > 8 and area < 0.995 * h * w:
            best = i
            best_area = area
    return (labels == best).astype(np.uint8)


def _ring_mask(h: int, w: int, ratio: float) -> np.ndarray:
    r = max(1, int(round(min(h, w) * ratio)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:r, :] = 1
    mask[-r:, :] = 1
    mask[:, :r] = 1
    mask[:, -r:] = 1
    return mask


def _line_angles(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180.0, threshold=30, minLineLength=24, maxLineGap=6)
    if lines is None or len(lines) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    angs = []
    lens = []
    for line in lines[:400]:
        x1, y1, x2, y2 = line[0].tolist()
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        ln = float(np.hypot(dx, dy))
        if ln < 8:
            continue
        ang = (float(np.degrees(np.arctan2(dy, dx))) + 180.0) % 180.0
        angs.append(ang)
        lens.append(ln)
    if len(angs) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(angs, dtype=np.float32), np.asarray(lens, dtype=np.float32)


def estimate_top_rotation(rgb: np.ndarray, cfg: CheckerboardTopNormConfig) -> dict[str, float | bool | int | str]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = _estimate_mask(rgb)
    bg = (1 - fg).astype(np.uint8)
    if int(bg.sum()) < int(0.03 * h * w):
        bg = _ring_mask(h, w, cfg.ring_ratio)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)
    edges = cv2.Canny(mag, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=(bg * 255))

    angles, lengths = _line_angles(edges)
    line_n = int(len(angles))
    if line_n == 0:
        return {
            "angle_deg": 0.0,
            "rot_conf": 0.0,
            "rot_ok": False,
            "rot_fail_reason": "no_lines",
            "rot_line_count": 0,
        }

    hist, _ = np.histogram(angles, bins=180, range=(0.0, 180.0), weights=lengths)
    peak_primary = int(np.argmax(hist))
    peak_primary_value = float(hist[peak_primary])

    hist_secondary = hist.copy()
    for d in range(-8, 9):
        hist_secondary[(peak_primary + d) % 180] = 0.0
    peak_secondary = int(np.argmax(hist_secondary))
    peak_secondary_value = float(hist_secondary[peak_secondary])
    peak_orthogonal = float(hist[(peak_primary + 90) % 180])

    mods = np.mod(angles, 90.0)
    theta = mods * (2.0 * np.pi / 90.0)
    cx = float(np.sum(lengths * np.cos(theta)))
    cy = float(np.sum(lengths * np.sin(theta)))
    rot_mod90 = float((np.degrees(np.arctan2(cy, cx)) * (90.0 / 360.0)) % 90.0)

    peak_ratio_score = peak_primary_value / (peak_primary_value + peak_secondary_value + 1e-6)
    line_score = min(1.0, line_n / 60.0)
    spread = float(np.sqrt(np.average((mods - np.average(mods, weights=lengths)) ** 2, weights=lengths)))
    spread_score = float(np.clip(1.0 - spread / 20.0, 0.0, 1.0))
    ortho_score = float(np.clip(peak_orthogonal / (peak_primary_value + 1e-6), 0.0, 1.0))
    conf = float(np.clip(0.35 * peak_ratio_score + 0.25 * line_score + 0.20 * spread_score + 0.20 * ortho_score, 0.0, 1.0))

    reasons = []
    if line_n < cfg.rot_line_min:
        reasons.append("line_count_low")
    if conf < cfg.rot_conf_min:
        reasons.append("confidence_low")

    return {
        "angle_deg": -rot_mod90,
        "rot_conf": conf,
        "rot_ok": len(reasons) == 0,
        "rot_fail_reason": "|".join(reasons),
        "rot_line_count": line_n,
    }


def rotate_rgb(rgb: np.ndarray, angle_deg: float, pad_value: int) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value),
    )


class CheckerboardTopNormalizer:
    def __init__(self, cfg: Optional[CheckerboardTopNormConfig] = None) -> None:
        self.cfg = cfg or CheckerboardTopNormConfig()
        self._angle_cache: Dict[str, Optional[float]] = {}

    def normalize(self, path: str | Path, image: Image.Image) -> Image.Image:
        if not self.cfg.enabled:
            return image

        key = str(Path(path).expanduser().resolve())
        if key not in self._angle_cache:
            rgb = np.asarray(image.convert("RGB"))
            info = estimate_top_rotation(rgb, self.cfg)
            self._angle_cache[key] = float(info["angle_deg"]) if bool(info["rot_ok"]) else None

        angle = self._angle_cache[key]
        if angle is None:
            return image

        rgb = np.asarray(image.convert("RGB"))
        rotated = rotate_rgb(rgb, angle_deg=angle, pad_value=self.cfg.pad_value)
        return Image.fromarray(rotated)
