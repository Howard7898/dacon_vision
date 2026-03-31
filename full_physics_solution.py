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

from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
from geometry_reasoning import GEOMETRY_FEATURE_NAMES, GeometryFeatureCache


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



REQUIRED_DATASET_CHILDREN = ("train.csv", "dev.csv", "sample_submission.csv", "train", "dev", "test")
DEV_LOGLOSS_TARGET_LOW = 0.015
DEV_LOGLOSS_TARGET_HIGH = 0.03
DEV_LOGLOSS_OVERFIT_LOW = 0.0001
DEV_LOGLOSS_DESIGN_BAD_HIGH = 0.05
DEFAULT_BACKBONE = "dinov2_vits14_reg"
DEFAULT_IMAGE_SIZE = 294
SUPPORTED_BACKBONES = [
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "efficientnet_v2_s",
    "resnet50",
    "convnext_tiny",
    "convnext_small",
]


def default_project_dir() -> Path:
    return Path(__file__).resolve().parent


def default_runs_dir(name: str) -> Path:
    return default_project_dir() / "runs" / name


def default_num_workers() -> int:
    if sys.platform == "darwin":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(4, max(cpu_count // 2, 1))


def is_dataset_root(path: str | Path) -> bool:
    path = Path(path).expanduser()
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_DATASET_CHILDREN)


def _unique_paths(paths: Iterable[str | Path | None]) -> List[Path]:
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
    script_dir = default_project_dir()
    downloads_dir = Path.home() / "Downloads"
    return _unique_paths([Path.cwd(), script_dir, script_dir.parent, downloads_dir])


def resolve_data_root(explicit: Optional[str]) -> Path:
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
    return Path(data_root) / "motion_targets.csv"


def resolve_motion_csv(data_root: str | Path, motion_csv: Optional[str]) -> Path:
    if motion_csv is None:
        return default_motion_csv(data_root)
    return Path(motion_csv).expanduser().resolve()


def get_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def use_non_blocking(device: torch.device) -> bool:
    return device.type == "cuda"


def optimize_runtime_for_device(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def describe_runtime_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"cuda ({name}, {total_gb:.1f} GB)"
    if device.type == "mps":
        return "mps"
    return "cpu"


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")



def label_to_int(label: str) -> int:
    return 1 if label == "unstable" else 0



def required_image_multiple(backbone_name: str) -> Optional[int]:
    name = backbone_name.lower()
    if name.startswith("dinov2_"):
        return 14
    return None



def validate_image_size(backbone_name: str, image_size: int) -> None:
    multiple = required_image_multiple(backbone_name)
    if multiple is not None and image_size % multiple != 0:
        raise ValueError(
            f"Backbone '{backbone_name}' requires --image-size to be a multiple of {multiple}, "
            f"but got {image_size}."
        )



def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_dev_logloss(logloss_value: float) -> Dict[str, str | float]:
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
    report = evaluate_dev_logloss(logloss_value)
    print(
        f"[{name}] DEV OOF LOGLOSS={logloss_value:.6f} "
        f"(target {DEV_LOGLOSS_TARGET_LOW:.3f}~{DEV_LOGLOSS_TARGET_HIGH:.3f}) -> {report['status']}"
    )
    print(f"[{name}] {report['message']}")
    return report


# -----------------------------------------------------------------------------
# Motion target extraction from train videos
# -----------------------------------------------------------------------------


@dataclass
class MotionExtractionConfig:
    resize: Tuple[int, int] = (64, 64)
    thr_low: float = 2.0
    thr_mid: float = 5.0
    thr_high: float = 10.0



def _first_hit(arr: np.ndarray, thr: float) -> int:
    idx = np.where(arr > thr)[0]
    return int(idx[0] + 1) if len(idx) else -1



def _severity_bucket(max_diff_first: float) -> int:
    # 0=tiny, 1=small, 2=mid, 3=large
    if max_diff_first < 2.0:
        return 0
    if max_diff_first < 5.0:
        return 1
    if max_diff_first < 10.0:
        return 2
    return 3



def _onset_bucket(first_move_thr2: int, first_move_thr5: int) -> int:
    # 0=very_early, 1=early, 2=late, 3=no_strong_hit
    onset = first_move_thr5 if first_move_thr5 >= 0 else first_move_thr2
    if onset < 0:
        return 3
    if onset < 10:
        return 0
    if onset < 20:
        return 1
    return 2



def _soft_target_from_motion(label_int: int, max_diff_first: float, mean_diff_prev: float) -> float:
    """
    Conservative soft target for logloss calibration.
    Hard label is still used implicitly; this only reduces overconfidence near the boundary.
    """
    # Normalize rough motion scale.
    motion_score = 0.65 * min(max_diff_first / 10.0, 1.5) + 0.35 * min(mean_diff_prev / 0.15, 1.5)
    motion_score = min(max(motion_score, 0.0), 1.5)

    if label_int == 0:
        # stable: allow slight wobble, but keep low probability.
        return float(np.clip(0.02 + 0.10 * min(motion_score, 1.0), 0.02, 0.15))
    # unstable: mild failures stay softer than violent collapses.
    return float(np.clip(0.65 + 0.30 * min(motion_score, 1.0), 0.65, 0.98))



def extract_motion_targets(data_root: str | Path, out_csv: str | Path, cfg: MotionExtractionConfig) -> pd.DataFrame:
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

        first_gray = cv2.cvtColor(cv2.resize(first, cfg.resize), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev_gray = first_gray
        mad_to_first: List[float] = []
        mad_prev: List[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, cfg.resize), cv2.COLOR_BGR2GRAY).astype(np.float32)
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
            first_move_thr2 = _first_hit(arr_first, cfg.thr_low)
            first_move_thr5 = _first_hit(arr_first, cfg.thr_mid)
            first_move_thr10 = _first_hit(arr_first, cfg.thr_high)

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


# -----------------------------------------------------------------------------
# Geometry clustering for grouped CV
# -----------------------------------------------------------------------------


@dataclass
class ClusterConfig:
    n_clusters: int = 32
    front_crop: Tuple[int, int, int, int] = (96, 80, 288, 320)
    top_crop: Tuple[int, int, int, int] = (112, 112, 272, 272)
    downsample: Tuple[int, int] = (24, 24)
    random_state: int = 42



def _load_center_gray(data_root: Path, split: str, sid: str, view: str, crop: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    arr = cv2.cvtColor(cv2.imread(str(data_root / split / sid / f"{view}.png")), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = crop
    arr = arr[y1:y2, x1:x2]
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = np.array(Image.fromarray(arr).resize(size), dtype=np.float32) / 255.0
    return arr.reshape(-1)



def build_geometry_clusters(data_root: str | Path, train_df: pd.DataFrame, dev_df: Optional[pd.DataFrame], cfg: ClusterConfig) -> pd.DataFrame:
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
        feats.append(np.concatenate([front, top]))
    X = np.stack(feats)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    n_clusters = max(1, min(cfg.n_clusters, len(all_df)))
    km = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init=20)
    clusters = km.fit_predict(Xs)
    out = all_df[["id"]].copy()
    out["geometry_cluster"] = clusters
    return out


# -----------------------------------------------------------------------------
# Image transforms
# -----------------------------------------------------------------------------


class CenterPhysicsCrop:
    """Keep the structure near the center while reducing background leakage."""

    def __init__(self, view: str):
        self.view = view

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if self.view == "front":
            box = (int(0.25 * w), int(0.20 * h), int(0.75 * w), int(0.88 * h))
        else:
            box = (int(0.29 * w), int(0.29 * h), int(0.71 * w), int(0.71 * h))
        return img.crop(box)



def build_train_transform(view: str, image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CenterPhysicsCrop(view=view),
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04)
            ], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.35),
            transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.2),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.35),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )



def build_valid_transform(view: str, image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CenterPhysicsCrop(view=view),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class DualViewDataset(Dataset):
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
        self.data_root = Path(data_root)
        self.df = frame_df.reset_index(drop=True).copy()
        self.split_map = split_map
        self.front_transform = front_transform
        self.top_transform = top_transform
        self.training = training
        self.top_normalizer = CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True)) if checkerboard_top_normalize else None
        self.geometry_cache = GeometryFeatureCache()

    def __len__(self) -> int:
        return len(self.df)

    def _load_img(self, sid: str, view: str) -> tuple[Path, Image.Image]:
        split = self.split_map[sid]
        path = self.data_root / split / sid / f"{view}.png"
        image = Image.open(path).convert("RGB")
        if view == "top" and self.top_normalizer is not None:
            image = self.top_normalizer.normalize(path, image)
        return path, image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
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

        # Optional auxiliary targets.
        aux_float_cols = ["max_diff_first", "mean_diff_prev", "soft_target"]
        aux_int_cols = ["severity_bucket", "onset_bucket", "source_domain"]
        for col in aux_float_cols:
            val = row[col] if col in row and pd.notna(row[col]) else np.nan
            sample[col] = torch.tensor(float(val) if pd.notna(val) else float("nan"), dtype=torch.float32)
        for col in aux_int_cols:
            val = row[col] if col in row and pd.notna(row[col]) else -1
            sample[col] = torch.tensor(int(val), dtype=torch.long)
        return sample


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)



def create_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name in ("dinov2_vits14", "dinov2_vits14_reg"):
        backbone = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        feat = 768  # CLS token (384) + mean of patch tokens (384) concatenated
        return backbone, feat
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=weights)
        feat = 768
        backbone = nn.Sequential(m.features, nn.LayerNorm((feat, 1, 1), eps=1e-6, elementwise_affine=True))
        return backbone, feat
    if name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        m = models.convnext_small(weights=weights)
        feat = 768
        backbone = nn.Sequential(m.features, nn.LayerNorm((feat, 1, 1), eps=1e-6, elementwise_affine=True))
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
        backbone = nn.Sequential(*(list(m.children())[:-2]))
        return backbone, feat
    raise ValueError(f"Unsupported backbone: {name}")


class ViewEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.backbone, feat_dim = create_backbone(backbone_name, pretrained=pretrained)
        self.pool = GeM()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name.startswith("dinov2_"):
            feat_dict = self.backbone.forward_features(x)
            cls = feat_dict["x_norm_clstoken"]             # (B, 384)
            patches = feat_dict["x_norm_patchtokens"]      # (B, N, 384)
            feat = torch.cat([cls, patches.mean(dim=1)], dim=1)  # (B, 768)
        else:
            fmap = self.backbone(x)
            if isinstance(fmap, (list, tuple)):
                fmap = fmap[-1]
            if fmap.ndim == 2:
                feat = fmap
            else:
                feat = self.pool(fmap)
        return self.proj(feat)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class DualViewPhysicsModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        emb_dim: int = 512,
        use_domain_head: bool = False,
        geometry_dim: int = len(GEOMETRY_FEATURE_NAMES),
        use_geometry_reasoning: bool = False,
    ) -> None:
        super().__init__()
        self.use_geometry_reasoning = use_geometry_reasoning
        self.front_encoder = ViewEncoder(backbone_name, pretrained=pretrained, out_dim=emb_dim)
        self.top_encoder = ViewEncoder(backbone_name, pretrained=pretrained, out_dim=emb_dim)
        self.view_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
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
        image_feat_dim = emb_dim * 3
        geom_emb_dim = emb_dim // 2
        fused_feat_dim = image_feat_dim + geom_emb_dim if use_geometry_reasoning else image_feat_dim
        if use_geometry_reasoning:
            self.geometry_proj = nn.Sequential(
                nn.LayerNorm(geometry_dim),
                nn.Linear(geometry_dim, geom_emb_dim),
                nn.GELU(),
                nn.Dropout(0.25),
            )
        self.classifier = nn.Sequential(
            nn.Linear(fused_feat_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1),
        )
        self.motion_reg = nn.Sequential(nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2))
        self.onset_head = nn.Sequential(nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))
        self.severity_head = nn.Sequential(nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))
        if use_geometry_reasoning:
            self.support_reg = nn.Sequential(nn.Linear(image_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2))
            self.margin_reg = nn.Sequential(nn.Linear(image_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 1))
        self.use_domain_head = use_domain_head
        if use_domain_head:
            self.domain_head = nn.Sequential(nn.Linear(fused_feat_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2))

    def forward(self, front: torch.Tensor, top: torch.Tensor, geom_feat: torch.Tensor, grl_lambda: float = 0.0) -> Dict[str, torch.Tensor]:
        f = self.front_encoder(front)
        t = self.top_encoder(top)
        tokens = torch.stack([f + self.view_embed[0], t + self.view_embed[1]], dim=1)
        fused = self.fusion(tokens)
        fused_mean = self.norm(fused.mean(dim=1))
        image_feat = torch.cat([f, t, fused_mean], dim=1)
        if self.use_geometry_reasoning:
            geom_emb = self.geometry_proj(geom_feat)
            feat = torch.cat([image_feat, geom_emb], dim=1)
        else:
            feat = image_feat
        out = {
            "logit": self.classifier(feat).squeeze(1),
            "motion_reg": self.motion_reg(feat),
            "onset_logit": self.onset_head(feat),
            "severity_logit": self.severity_head(feat),
            "feat": feat,
        }
        if self.use_geometry_reasoning:
            out["support_reg"] = self.support_reg(image_feat)
            out["margin_reg"] = self.margin_reg(image_feat).squeeze(1)
        if self.use_domain_head:
            rev = GradientReversal.apply(feat, grl_lambda)
            out["domain_logit"] = self.domain_head(rev)
        return out


# -----------------------------------------------------------------------------
# Losses and calibration
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_root: str = ""
    out_dir: str = str(default_runs_dir("default"))
    motion_csv: Optional[str] = None
    backbone: str = DEFAULT_BACKBONE
    pretrained: bool = True
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 8
    num_workers: int = default_num_workers()
    epochs: int = 20
    lr: float = 2e-4
    backbone_lr: float = 1e-5
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    num_folds: int = 5
    seed: int = 42
    use_domain_head: bool = False
    use_amp: bool = True
    grad_clip: float = 1.0
    aux_motion_weight: float = 0.20
    aux_onset_weight: float = 0.15
    aux_severity_weight: float = 0.15
    aux_support_weight: float = 0.08
    aux_margin_weight: float = 0.08
    domain_weight: float = 0.05
    tta_passes: int = 4
    checkerboard_top_normalize: bool = True
    use_geometry_reasoning: bool = False


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
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
            self.temperature.clamp_(min=0.1)
        return float(self.temperature.detach().cpu().item())



def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: TrainConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    label = batch["label"]
    hard_target = label
    soft_target = batch["soft_target"]
    target = torch.where(torch.isnan(soft_target), hard_target, soft_target)
    target = torch.where(label < 0, torch.zeros_like(target), target)

    valid_main = label >= 0
    if valid_main.any():
        loss_main = F.binary_cross_entropy_with_logits(outputs["logit"][valid_main], target[valid_main])
    else:
        loss_main = outputs["logit"].sum() * 0.0

    # Motion regression targets: [max_diff_first, mean_diff_prev]
    motion_tgt = torch.stack([batch["max_diff_first"], batch["mean_diff_prev"]], dim=1)
    valid_motion = ~torch.isnan(motion_tgt).any(dim=1)
    if valid_motion.any():
        pred = outputs["motion_reg"][valid_motion]
        tgt = motion_tgt[valid_motion]
        # Mild normalization to keep losses stable.
        tgt_norm = torch.stack([tgt[:, 0] / 10.0, tgt[:, 1] / 0.15], dim=1).clamp(min=0.0, max=2.0)
        loss_motion = F.smooth_l1_loss(pred, tgt_norm)
    else:
        loss_motion = outputs["motion_reg"].sum() * 0.0

    onset = batch["onset_bucket"]
    valid_onset = onset >= 0
    if valid_onset.any():
        loss_onset = F.cross_entropy(outputs["onset_logit"][valid_onset], onset[valid_onset])
    else:
        loss_onset = outputs["onset_logit"].sum() * 0.0

    sev = batch["severity_bucket"]
    valid_sev = sev >= 0
    if valid_sev.any():
        loss_sev = F.cross_entropy(outputs["severity_logit"][valid_sev], sev[valid_sev])
    else:
        loss_sev = outputs["severity_logit"].sum() * 0.0

    if cfg.use_geometry_reasoning and "support_reg" in outputs:
        support_tgt = batch["support_target"]
        loss_support = F.smooth_l1_loss(outputs["support_reg"], support_tgt)
    else:
        loss_support = outputs["logit"].sum() * 0.0

    if cfg.use_geometry_reasoning and "margin_reg" in outputs:
        margin_tgt = batch["collapse_margin"]
        loss_margin = F.smooth_l1_loss(outputs["margin_reg"], margin_tgt)
    else:
        loss_margin = outputs["logit"].sum() * 0.0

    if cfg.use_domain_head and "domain_logit" in outputs:
        dom = batch["source_domain"]
        valid_dom = dom >= 0
        if valid_dom.any():
            loss_dom = F.cross_entropy(outputs["domain_logit"][valid_dom], dom[valid_dom])
        else:
            loss_dom = outputs["domain_logit"].sum() * 0.0
    else:
        loss_dom = outputs["logit"].sum() * 0.0

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


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------



def _move_batch_to_device(batch: Dict[str, torch.Tensor | str], device: torch.device) -> Dict[str, torch.Tensor | str]:
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
    model.eval()
    rows: List[Dict[str, float | str]] = []
    progress = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        ids = batch["id"]
        probs_accum: Optional[torch.Tensor] = None
        batch_dev = _move_batch_to_device(batch, device)
        for _ in range(tta_passes):
            out = model(batch_dev["front"], batch_dev["top"], batch_dev["geom_feat"], grl_lambda=0.0)
            probs = torch.sigmoid(out["logit"]).detach().cpu()
            probs_accum = probs if probs_accum is None else probs_accum + probs
        probs_accum = probs_accum / float(tta_passes)
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
    device = get_runtime_device()
    optimize_runtime_for_device(device)
    front_train_tf = build_train_transform("front", cfg.image_size)
    top_train_tf = build_train_transform("top", cfg.image_size)
    front_valid_tf = build_valid_transform("front", cfg.image_size)
    top_valid_tf = build_valid_transform("top", cfg.image_size)
    pin_memory = use_pin_memory(device)

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
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=max(cfg.batch_size, 8), shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory)

    model = DualViewPhysicsModel(
        cfg.backbone,
        pretrained=cfg.pretrained,
        use_domain_head=cfg.use_domain_head,
        use_geometry_reasoning=cfg.use_geometry_reasoning,
    ).to(device)
    is_dinov2 = cfg.backbone.lower().startswith("dinov2_")
    if is_dinov2:
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if "front_encoder.backbone" in name or "top_encoder.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.lr},
        ]
    else:
        param_groups = list(model.parameters())
    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.warmup_epochs > 0 and cfg.epochs > cfg.warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[cfg.warmup_epochs])
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
            grl_lambda = min(epoch / max(cfg.epochs - 1, 1), 1.0) if cfg.use_domain_head else 0.0
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp and device.type == "cuda"):
                outputs = model(batch["front"], batch["top"], batch["geom_feat"], grl_lambda=grl_lambda)
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

        # Validation on deterministic transforms.
        valid_pred = predict_loader(model, valid_loader, device=device, tta_passes=1, desc=f"valid {epoch + 1}/{cfg.epochs}")
        y_true = valid_df["label_int"].values
        y_pred = valid_pred.sort_values("id")["pred"].values
        # align by id to be safe
        merged = valid_df[["id", "label_int"]].merge(valid_pred, on="id", how="left")
        ll = log_loss(merged["label_int"].values, merged["pred"].values, labels=[0, 1])
        if len(set(merged["label_int"].values)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(merged["label_int"].values, merged["pred"].values)
        if ll < best_score:
            best_score = ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch+1:02d} valid_logloss={ll:.6f} valid_auc={auc:.6f}")

    assert best_state is not None
    model.load_state_dict(best_state)

    # temperature scaling on validation logits
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
    temp = calibrator.fit(merged["logit"].values.astype(np.float32), merged["label_int"].values.astype(np.float32))
    cal_prob = sigmoid_np(merged["logit"].values / temp)
    ll_cal = log_loss(merged["label_int"].values, cal_prob, labels=[0, 1])
    if len(set(merged["label_int"].values)) < 2:
        auc_cal = 0.5
    else:
        auc_cal = roc_auc_score(merged["label_int"].values, cal_prob)

    torch.save(best_state, fold_dir / "best_model.pt")
    with open(fold_dir / "temperature.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": temp, "valid_logloss": ll_cal, "valid_auc": auc_cal}, f, ensure_ascii=False, indent=2)

    oof = valid_df[["id", "label_int", "source_domain"]].merge(merged[["id", "logit"]], on="id", how="left")
    oof["pred_raw"] = sigmoid_np(merged["logit"].values)
    oof["pred_cal"] = cal_prob
    oof.to_csv(fold_dir / "oof_valid.csv", index=False)

    return {"valid_logloss": float(ll_cal), "valid_auc": float(auc_cal), "temperature": float(temp)}


# -----------------------------------------------------------------------------
# Data assembly and fold generation
# -----------------------------------------------------------------------------



def prepare_tables(data_root: str | Path, motion_csv: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    train_df = read_csv(data_root / "train.csv")
    dev_df = read_csv(data_root / "dev.csv")
    test_df = read_csv(data_root / "sample_submission.csv")

    train_df["label_int"] = train_df["label"].map(label_to_int).astype(int)
    dev_df["label_int"] = dev_df["label"].map(label_to_int).astype(int)
    train_df["source_domain"] = 0
    dev_df["source_domain"] = 1
    test_df["source_domain"] = -1

    if motion_csv is not None and Path(motion_csv).exists():
        motion_df = pd.read_csv(motion_csv)
        train_df = train_df.merge(motion_df.drop(columns=["label"], errors="ignore"), on=["id", "label_int"], how="left")
    return train_df, dev_df, test_df



def make_split_map(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    split_map = {sid: "train" for sid in train_df["id"].tolist()}
    split_map.update({sid: "dev" for sid in dev_df["id"].tolist()})
    if test_df is not None:
        split_map.update({sid: "test" for sid in test_df["id"].tolist()})
    return split_map


# -----------------------------------------------------------------------------
# Workflows
# -----------------------------------------------------------------------------



def run_design_holdout(cfg: TrainConfig) -> None:
    """
    Design phase: train only on TRAIN, validate only on DEV.
    Use this to freeze architecture / augmentation / loss choices.
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
    """
    Final phase after design freeze:
    pooled train+dev, geometry-clustered grouped CV, fold ensemble.
    """
    out_dir = ensure_dir(cfg.out_dir)
    train_df, dev_df, test_df = prepare_tables(cfg.data_root, cfg.motion_csv)
    pooled = pd.concat([train_df, dev_df], ignore_index=True)

    cluster_df = build_geometry_clusters(cfg.data_root, train_df, dev_df, ClusterConfig())
    pooled = pooled.merge(cluster_df, on="id", how="left")

    # stratify jointly by label and source domain, group by geometry cluster.
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
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    oof_all_df = pd.concat(oof_all, ignore_index=True)
    oof_all_df.to_csv(out_dir / "oof_all.csv", index=False)
    dev_oof = oof_all_df[oof_all_df["source_domain"] == 1].copy()
    if not dev_oof.empty:
        dev_oof_logloss = log_loss(dev_oof["label_int"].values, dev_oof["pred_cal"].values, labels=[0, 1])
        dev_report = print_dev_logloss_report("cv-train", float(dev_oof_logloss))
        with open(out_dir / "dev_oof_logloss_report.json", "w", encoding="utf-8") as f:
            json.dump(dev_report, f, ensure_ascii=False, indent=2)
    print(summary)



def run_full_pipeline(cfg: TrainConfig, refresh_motion: bool = False) -> None:
    motion_csv = Path(cfg.motion_csv) if cfg.motion_csv is not None else default_motion_csv(cfg.data_root)
    if refresh_motion or not motion_csv.exists():
        print(f"Extracting motion targets -> {motion_csv}")
        extract_motion_targets(cfg.data_root, motion_csv, MotionExtractionConfig())
    else:
        print(f"Using existing motion targets: {motion_csv}")

    full_cfg = dataclasses.replace(cfg, motion_csv=str(motion_csv))
    run_pooled_grouped_cv(full_cfg)
    make_submission(full_cfg, full_cfg.out_dir)


def make_submission(cfg: TrainConfig, run_dir: str | Path) -> None:
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
    test_loader = DataLoader(test_ds, batch_size=max(cfg.batch_size, 8), shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory)

    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {run_dir}")

    pred_frames: List[pd.DataFrame] = []
    skipped_folds: List[str] = []
    for fold_dir in fold_dirs:
        # Check fold quality before loading model
        temp_path = fold_dir / "temperature.json"
        if temp_path.exists():
            with open(temp_path, "r", encoding="utf-8") as f:
                temp_info = json.load(f)
            fold_temp = temp_info.get("temperature", 1.0)
            fold_ll = temp_info.get("valid_logloss", 0.0)
            if fold_temp <= 0 or fold_ll > 2.0:
                print(f"[WARNING] Skipping {fold_dir.name}: temperature={fold_temp:.4f}, valid_logloss={fold_ll:.4f}")
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
        preds = predict_loader(model, test_loader, device, tta_passes=cfg.tta_passes, desc=f"{fold_dir.name} test")
        if temp_path.exists():
            with open(temp_path, "r", encoding="utf-8") as f:
                temp = json.load(f).get("temperature", 1.0)
            # convert back to logit -> calibrated probability
            p = preds["pred"].clip(1e-6, 1 - 1e-6).values
            logit = np.log(p / (1 - p))
            preds["pred"] = sigmoid_np(logit / temp)
        preds = preds.rename(columns={"pred": f"pred_{fold_dir.name}"})
        pred_frames.append(preds)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Fallback: if all folds were filtered, use all without filtering
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
            preds = predict_loader(model, test_loader, device, tta_passes=cfg.tta_passes, desc=f"{fold_dir.name} test")
            temp_path = fold_dir / "temperature.json"
            if temp_path.exists():
                with open(temp_path, "r", encoding="utf-8") as f:
                    temp = json.load(f).get("temperature", 1.0)
                p = preds["pred"].clip(1e-6, 1 - 1e-6).values
                logit = np.log(p / (1 - p))
                preds["pred"] = sigmoid_np(logit / max(temp, 0.1))
            preds = preds.rename(columns={"pred": f"pred_{fold_dir.name}"})
            pred_frames.append(preds)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if skipped_folds:
        print(f"[INFO] Used {len(pred_frames)} folds, skipped {len(skipped_folds)}: {skipped_folds}")

    sub = test_df[["id"]].copy()
    for pf in pred_frames:
        sub = sub.merge(pf, on="id", how="left")
    pred_cols = [c for c in sub.columns if c.startswith("pred_")]
    sub["prob_unstable"] = sub[pred_cols].mean(axis=1)

    # DACON leaderboard expects id, unstable_prob, stable_prob in this order.
    out = test_df[["id"]].copy()
    out["unstable_prob"] = sub["prob_unstable"]
    out["stable_prob"] = 1.0 - sub["prob_unstable"]

    out.to_csv(run_dir / "submission.csv", index=False)
    print(f"Saved: {run_dir / 'submission.csv'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physics-aware dual-view solution")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-motion")
    p_extract.add_argument("--data-root", type=str, default=None)
    p_extract.add_argument("--out-csv", type=str, default=None)

    def add_train_args(p: argparse.ArgumentParser, default_out_dir: Path) -> None:
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
    validate_image_size(cfg.backbone, cfg.image_size)



def main() -> None:
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
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir is not None else Path(cfg.out_dir)
        print(f"Resolved run_dir: {run_dir}")
        make_submission(cfg, run_dir)
    elif args.command == "full-run":
        run_full_pipeline(cfg, refresh_motion=bool(args.refresh_motion))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
