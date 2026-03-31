# =============================================================================
# checkerboard_rectification.py
# -----------------------------------------------------------------------------
# 작성자  : (공개 생략)
# 인코딩  : UTF-8
# Python  : 3.10+
# 설명    : 체커보드 상단 뷰 회전 정규화 모듈 (엣지 기반 각도 추정 + 회전 보정)
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import cv2
import numpy as np
from PIL import Image


# ============================================================
# 설정 데이터클래스
# ============================================================

@dataclass(frozen=True)
class CheckerboardTopNormConfig:
    """체커보드 상단 뷰 정규화 설정값 묶음.

    Attributes:
        enabled: 정규화 활성화 여부.
        ring_ratio: 배경 마스크 추출 시 테두리 영역 비율.
        rot_line_min: 회전 추정에 필요한 최소 선분 개수.
        rot_conf_min: 회전 보정을 수행할 최소 신뢰도 임계값.
        pad_value: 회전 후 패딩 픽셀값 (그레이 레벨).
    """

    enabled: bool = True
    ring_ratio: float = 0.10          # 테두리 영역 비율: 배경 마스크 부족 시 사용
    rot_line_min: int = 10            # 신뢰도 계산에 필요한 최소 선분 수
    rot_conf_min: float = 0.20        # 신뢰도 임계값: 이 값 미만이면 회전 보정 스킵
    pad_value: int = 128              # 회전 보정 후 빈 영역을 채울 그레이값


# ============================================================
# 내부 유틸리티 함수
# ============================================================

def _estimate_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB 이미지에서 전경 객체 마스크를 추정한다.

    채도·밝기·명도 퍼센타일 기반으로 이진 마스크를 생성하고,
    형태학적 연산으로 노이즈를 제거한 뒤 가장 큰 연결 성분을 반환한다.

    Args:
        rgb: (H, W, 3) RGB 이미지 배열.

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

    # 가장 넓은 유효 성분 선택 (이미지 전체 면적의 99.5% 미만인 것만 허용)
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
    """이미지 테두리 링 영역을 배경 마스크로 반환한다.

    Args:
        h: 이미지 높이.
        w: 이미지 너비.
        ratio: min(h, w) 대비 테두리 두께 비율.

    Returns:
        (H, W) 이진 마스크 (uint8, 0 또는 1).
    """
    r = max(1, int(round(min(h, w) * ratio)))  # 테두리 두께 (픽셀)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:r, :] = 1   # 상단 테두리
    mask[-r:, :] = 1  # 하단 테두리
    mask[:, :r] = 1   # 좌측 테두리
    mask[:, -r:] = 1  # 우측 테두리
    return mask


def _line_angles(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """에지 이미지에서 선분을 검출하고 각도와 길이를 반환한다.

    Hough 변환으로 선분을 검출하여 각 선분의 방향각(0~180°)과 길이를 추출한다.

    Args:
        edge: (H, W) 에지 이진 이미지.

    Returns:
        angles: 선분 방향각 배열 (float32, 도 단위, 0~180°).
        lengths: 선분 길이 배열 (float32, 픽셀 단위).
    """
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 180.0,
        threshold=30,      # Hough 투표 임계값
        minLineLength=24,  # 최소 선분 길이 (픽셀)
        maxLineGap=6,      # 선분 내 최대 허용 간격 (픽셀)
    )
    if lines is None or len(lines) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    angs = []
    lens = []
    for line in lines[:400]:  # 최대 400개 선분만 처리
        x1, y1, x2, y2 = line[0].tolist()
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        ln = float(np.hypot(dx, dy))
        if ln < 8:  # 너무 짧은 선분 스킵 (픽셀)
            continue
        ang = (float(np.degrees(np.arctan2(dy, dx))) + 180.0) % 180.0  # 0~180° 정규화
        angs.append(ang)
        lens.append(ln)
    if len(angs) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(angs, dtype=np.float32), np.asarray(lens, dtype=np.float32)


# ============================================================
# 회전 추정 및 보정
# ============================================================

def estimate_top_rotation(
    rgb: np.ndarray,
    cfg: CheckerboardTopNormConfig,
) -> dict[str, float | bool | int | str]:
    """상단 뷰 이미지에서 회전 각도를 추정한다.

    배경 영역의 에지를 분석하여 구조물의 기울기를 추정하고,
    신뢰도 점수와 함께 반환한다.

    Args:
        rgb: (H, W, 3) 상단 뷰 RGB 이미지.
        cfg: 회전 추정 설정.

    Returns:
        다음 키를 포함하는 딕셔너리:
            angle_deg: 추정된 회전 각도 (도).
            rot_conf: 신뢰도 점수 (0.0~1.0).
            rot_ok: 회전 보정 적용 여부.
            rot_fail_reason: 실패 사유 문자열 (성공 시 빈 문자열).
            rot_line_count: 검출된 선분 수.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = _estimate_mask(rgb)
    bg = (1 - fg).astype(np.uint8)
    if int(bg.sum()) < int(0.03 * h * w):  # 배경 비율이 3% 미만이면 링 마스크 대체
        bg = _ring_mask(h, w, cfg.ring_ratio)

    # Scharr 필터로 그래디언트 계산 후 Canny 에지 추출
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)  # 0~255 정규화
    edges = cv2.Canny(mag, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=(bg * 255))  # 배경 영역만 에지 유지

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

    # 각도 히스토그램에서 주 피크와 보조 피크 추출
    hist, _ = np.histogram(angles, bins=180, range=(0.0, 180.0), weights=lengths)
    peak_primary = int(np.argmax(hist))                       # 주 피크 각도 인덱스
    peak_primary_value = float(hist[peak_primary])

    hist_secondary = hist.copy()
    for d in range(-8, 9):
        hist_secondary[(peak_primary + d) % 180] = 0.0       # 주 피크 ±8° 제거
    peak_secondary = int(np.argmax(hist_secondary))
    peak_secondary_value = float(hist_secondary[peak_secondary])
    peak_orthogonal = float(hist[(peak_primary + 90) % 180])  # 직교 방향 피크값

    # 원형 통계로 mod-90 회전량 계산
    mods = np.mod(angles, 90.0)                              # 90° 주기로 접기
    theta = mods * (2.0 * np.pi / 90.0)                     # 원형 변환
    cx = float(np.sum(lengths * np.cos(theta)))
    cy = float(np.sum(lengths * np.sin(theta)))
    rot_mod90 = float((np.degrees(np.arctan2(cy, cx)) * (90.0 / 360.0)) % 90.0)

    # 신뢰도 점수 계산 (4가지 요소 가중합)
    peak_ratio_score = peak_primary_value / (peak_primary_value + peak_secondary_value + 1e-6)
    line_score = min(1.0, line_n / 60.0)                     # 선분 수 기반 점수 (60개 이상이면 1.0)
    spread = float(np.sqrt(np.average(
        (mods - np.average(mods, weights=lengths)) ** 2, weights=lengths
    )))
    spread_score = float(np.clip(1.0 - spread / 20.0, 0.0, 1.0))
    ortho_score = float(np.clip(peak_orthogonal / (peak_primary_value + 1e-6), 0.0, 1.0))
    conf = float(np.clip(
        0.35 * peak_ratio_score + 0.25 * line_score + 0.20 * spread_score + 0.20 * ortho_score,
        0.0,
        1.0,
    ))

    # 신뢰도 검증 및 실패 사유 수집
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
    """RGB 이미지를 중심 기준으로 회전한다.

    Args:
        rgb: (H, W, 3) RGB 이미지 배열.
        angle_deg: 회전 각도 (도, 양수=반시계).
        pad_value: 빈 영역을 채울 단일 채널 픽셀값 (0~255).

    Returns:
        회전된 (H, W, 3) RGB 이미지 배열.
    """
    if abs(angle_deg) < 1e-6:  # 회전량이 극히 작으면 원본 반환
        return rgb
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)  # 이미지 중심 기준 회전 행렬
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value),
    )


# ============================================================
# 정규화 클래스
# ============================================================

class CheckerboardTopNormalizer:
    """상단 뷰 이미지를 회전 보정하는 정규화기.

    이미지 경로를 키로 각도를 캐싱하여 동일 이미지를 여러 번 처리할 때
    중복 계산을 방지한다.

    Attributes:
        cfg: 정규화 설정.
    """

    def __init__(self, cfg: Optional[CheckerboardTopNormConfig] = None) -> None:
        """초기화.

        Args:
            cfg: 정규화 설정 (None이면 기본값 사용).
        """
        self.cfg = cfg or CheckerboardTopNormConfig()
        self._angle_cache: Dict[str, Optional[float]] = {}  # 경로 → 각도 캐시

    def normalize(self, path: str | Path, image: Image.Image) -> Image.Image:
        """이미지를 회전 보정하여 반환한다.

        설정이 비활성화되어 있거나 신뢰도가 낮으면 원본을 그대로 반환한다.

        Args:
            path: 이미지 파일 경로 (캐시 키로 사용).
            image: 정규화할 PIL 이미지.

        Returns:
            회전 보정된 PIL 이미지 (보정 불필요 시 원본 반환).
        """
        if not self.cfg.enabled:
            return image

        key = str(Path(path).expanduser().resolve())
        if key not in self._angle_cache:
            rgb = np.asarray(image.convert("RGB"))
            info = estimate_top_rotation(rgb, self.cfg)
            # 신뢰도 조건을 통과한 경우에만 각도 저장, 아니면 None
            self._angle_cache[key] = float(info["angle_deg"]) if bool(info["rot_ok"]) else None

        angle = self._angle_cache[key]
        if angle is None:
            return image

        rgb = np.asarray(image.convert("RGB"))
        rotated = rotate_rgb(rgb, angle_deg=angle, pad_value=self.cfg.pad_value)
        return Image.fromarray(rotated)
