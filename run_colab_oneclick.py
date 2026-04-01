# =============================================================================
# run_colab_oneclick.py
# -----------------------------------------------------------------------------
# 작성자  : (공개 생략)
# 인코딩  : UTF-8
# Python  : 3.10+
# 설명    : Google Colab에서 원클릭으로 데이터셋 준비·학습·결과 저장을 수행하는
#           자동화 실행 스크립트
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile


# ============================================================
# 전역 상수 정의
# ============================================================

REQUIRED_DATASET_CHILDREN = (
    "train.csv",
    "dev.csv",
    "sample_submission.csv",
    "train",
    "dev",
    "test",
)  # 유효한 데이터셋 루트 디렉토리가 반드시 포함해야 하는 파일/디렉토리

DEFAULT_DRIVE_ZIP_PATH = "/content/drive/MyDrive/open (7).zip"       # Google Drive 내 기본 데이터셋 압축 경로
DEFAULT_LOCAL_ROOT = "/content/physics_solution_runtime"             # Colab 로컬 작업 루트 경로
DEFAULT_DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/physics_solution_outputs"  # 결과 저장 Drive 경로


# ============================================================
# 유틸리티 함수
# ============================================================

def print_stage(message: str) -> None:
    """타임스탬프와 함께 단계 메시지를 출력한다.

    Args:
        message: 출력할 단계 설명 문자열.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def running_in_colab() -> bool:
    """현재 실행 환경이 Google Colab인지 확인한다.

    Returns:
        Colab 환경이면 True, 아니면 False.
    """
    return "google.colab" in sys.modules or Path("/content").exists()


def ensure_drive_mounted() -> None:
    """Google Drive가 마운트되어 있지 않으면 마운트를 시도한다.

    Raises:
        RuntimeError: google.colab 모듈을 불러올 수 없을 때.
    """
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Google Drive is not mounted and google.colab is unavailable.") from exc
    drive.mount("/content/drive", force_remount=False)


def ensure_python_deps() -> None:
    """필수 Python 패키지가 설치되어 있는지 확인하고, 없으면 pip으로 설치한다."""
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python-headless"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("tqdm", "tqdm"),
    ]
    missing = []
    for module_name, package_name in required:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    if not missing:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + sorted(set(missing)),
        check=True,
    )


def copy_zip_to_local(source_zip: Path, local_zip: Path) -> Path:
    """Drive의 ZIP 파일을 Colab 로컬 디스크에 복사한다.

    크기와 수정 시각이 동일하면 복사를 건너뛴다.

    Args:
        source_zip: 원본 ZIP 파일 경로 (Drive).
        local_zip: 복사 대상 경로 (로컬).

    Returns:
        복사된(또는 기존) 로컬 ZIP 파일 경로.
    """
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    if local_zip.exists():
        src_stat = source_zip.stat()
        dst_stat = local_zip.stat()
        # 크기와 수정 시각이 동일하면 이미 최신 상태
        if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
            return local_zip
    shutil.copy2(source_zip, local_zip)
    return local_zip


def print_runtime_summary() -> None:
    """현재 실행 환경(GPU/CPU)의 하드웨어 정보를 출력한다."""
    try:
        import torch
    except ImportError:
        print("Runtime device: torch unavailable", flush=True)
        return

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Runtime GPU: {name} ({total_gb:.1f} GB VRAM)", flush=True)
        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
                check=False,
            )
        except FileNotFoundError:
            pass
        return
    print("Runtime device: CPU-only", flush=True)


# ============================================================
# 데이터셋 검색 및 추출
# ============================================================

def is_dataset_root(path: Path) -> bool:
    """주어진 경로가 유효한 데이터셋 루트 디렉토리인지 확인한다.

    Args:
        path: 검사할 디렉토리 경로.

    Returns:
        필수 파일·디렉토리가 모두 존재하면 True.
    """
    return path.is_dir() and all((path / child).exists() for child in REQUIRED_DATASET_CHILDREN)


def find_dataset_root(search_root: Path) -> Path:
    """검색 루트 아래에서 유효한 데이터셋 루트를 탐색한다.

    Args:
        search_root: 탐색을 시작할 루트 경로.

    Returns:
        발견된 데이터셋 루트 경로 (가장 얕은 경로 우선).

    Raises:
        FileNotFoundError: 유효한 데이터셋 루트를 찾지 못했을 때.
    """
    if is_dataset_root(search_root):
        return search_root

    candidates: list[Path] = []
    for current_root, dirnames, _filenames in os.walk(search_root):
        path = Path(current_root)
        if is_dataset_root(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset root under {search_root}. "
            "Expected train.csv, dev.csv, sample_submission.csv and train/dev/test directories."
        )
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def extract_dataset_zip(local_zip: Path, extract_root: Path, force_reextract: bool) -> Path:
    """로컬 ZIP 파일을 압축 해제하고 데이터셋 루트 경로를 반환한다.

    Args:
        local_zip: 로컬 ZIP 파일 경로.
        extract_root: 압축 해제 대상 디렉토리.
        force_reextract: True이면 기존 압축 해제 결과를 삭제하고 재압축 해제.

    Returns:
        압축 해제된 데이터셋 루트 경로.
    """
    if force_reextract and extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    dataset_root: Path | None = None
    if not force_reextract:
        try:
            dataset_root = find_dataset_root(extract_root)
        except FileNotFoundError:
            dataset_root = None
    if dataset_root is not None:
        return dataset_root

    with ZipFile(local_zip) as zip_file:
        zip_file.extractall(extract_root)
    return find_dataset_root(extract_root)


# ============================================================
# 파이프라인 실행
# ============================================================

def run_pipeline(
    project_root: Path,
    dataset_root: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    """full_physics_solution.py의 full-run 명령을 서브프로세스로 실행한다.

    Args:
        project_root: 프로젝트 루트 디렉토리 (full_physics_solution.py 위치).
        dataset_root: 데이터셋 루트 경로.
        run_dir: 실행 결과를 저장할 디렉토리.
        args: parse_args()로 파싱된 CLI 인자 Namespace.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    print_stage("Starting full pipeline")
    command = [
        sys.executable,
        str(project_root / "full_physics_solution.py"),
        "full-run",
        "--data-root",    str(dataset_root),
        "--out-dir",      str(run_dir),
        "--backbone",     args.backbone,
        "--image-size",   str(args.image_size),
        "--batch-size",   str(args.batch_size),
        "--epochs",       str(args.epochs),
        "--num-folds",    str(args.num_folds),
        "--num-workers",  str(args.num_workers),
        "--tta-passes",   str(args.tta_passes),
        "--backbone-lr",  str(args.backbone_lr),
        "--warmup-epochs", str(args.warmup_epochs),
        "--weight-decay", str(args.weight_decay),
    ]
    if not args.no_pretrained:
        command.append("--pretrained")
    if args.refresh_motion:
        command.append("--refresh-motion")
    if args.no_checkerboard_top_normalize:
        command.append("--no-checkerboard-top-normalize")
    if args.use_domain_head:
        command.append("--use-domain-head")
    if args.enable_geometry_reasoning:
        command.append("--enable-geometry-reasoning")

    env = os.environ.copy()
    env["PHYSICS_DATA_ROOT"] = str(dataset_root)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=project_root, check=True, env=env)


# ============================================================
# 결과 아티팩트 Drive 복사
# ============================================================

def copy_artifacts_to_drive(
    run_dir: Path,
    drive_output_root: Path,
    dataset_root: Path,
    project_root: Path,
    args: argparse.Namespace,
) -> Path:
    """실행 결과물을 Google Drive로 복사하고 요약 JSON을 작성한다.

    Args:
        run_dir: 학습 결과가 저장된 로컬 디렉토리.
        drive_output_root: Drive 내 결과 저장 루트 경로.
        dataset_root: 데이터셋 루트 경로.
        project_root: 프로젝트 루트 경로.
        args: CLI 인자 Namespace.

    Returns:
        생성된 아티팩트 루트 경로.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")         # 실행 타임스탬프
    artifact_root = drive_output_root / f"run_{stamp}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    # 학습 결과 전체 복사
    copied_run_dir = artifact_root / "run_dir"
    shutil.copytree(run_dir, copied_run_dir, dirs_exist_ok=True)

    # 제출 파일 복사
    submission = run_dir / "submission.csv"
    if submission.exists():
        shutil.copy2(submission, artifact_root / "submission.csv")

    # 모션 타겟 CSV 복사 (존재하는 경우)
    motion_csv = dataset_root / "motion_targets.csv"
    if motion_csv.exists():
        shutil.copy2(motion_csv, artifact_root / "motion_targets.csv")

    # 실행 요약 JSON 작성
    summary = {
        "timestamp": stamp,
        "run_dir": str(run_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "submission_csv": (
            str((artifact_root / "submission.csv").resolve()) if submission.exists() else ""
        ),
        "args": vars(args),
    }
    (artifact_root / "colab_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 프로젝트 소스 파일 복사
    for filename in [
        "full_physics_solution.py",
        "checkerboard_rectification.py",
        "geometry_reasoning.py",
        "run_colab_oneclick.py",
        "README_COLAB.md",
    ]:
        src = project_root / filename
        if src.exists():
            shutil.copy2(src, artifact_root / filename)

    # 전체를 ZIP으로 압축
    archive_path = shutil.make_archive(str(artifact_root), "zip", root_dir=artifact_root)
    print(f"Artifacts copied to: {artifact_root}")
    print(f"Artifacts zip: {archive_path}")
    return artifact_root


# ============================================================
# CLI 인자 파싱
# ============================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱하여 Namespace로 반환한다.

    Returns:
        파싱된 인자 Namespace.
    """
    parser = argparse.ArgumentParser(description="One-click Colab runner for physics_solution")
    parser.add_argument("--drive-zip-path", default=DEFAULT_DRIVE_ZIP_PATH)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--drive-output-root", default=DEFAULT_DRIVE_OUTPUT_ROOT)
    parser.add_argument(
        "--backbone",
        default="dinov2_vits14_reg",
        choices=[
            "dinov2_vits14",
            "dinov2_vits14_reg",
            "efficientnet_v2_s",
            "resnet50",
            "convnext_tiny",
            "convnext_small",
        ],
    )
    # ── 추천 파라미터 기본값 (v0.05) ──────────────────────────────────────────
    parser.add_argument("--image-size",    type=int,   default=336)     # 294 → 336
    parser.add_argument("--batch-size",    type=int,   default=32)      # 8   → 32
    parser.add_argument("--backbone-lr",   type=float, default=2e-5)    # 1e-5 → 2e-5
    parser.add_argument("--tta-passes",    type=int,   default=8)       # 4   → 8
    parser.add_argument("--num-workers",   type=int,   default=4)       # 2   → 4
    parser.add_argument("--weight-decay",  type=float, default=5e-5)    # 신규 추가
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=30)      # 20 → 30
    parser.add_argument("--warmup-epochs", type=int,   default=3)
    parser.add_argument("--num-folds",     type=int,   default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--refresh-motion", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--no-checkerboard-top-normalize", action="store_true")
    parser.add_argument("--use-domain-head",               action="store_true")
    parser.add_argument("--enable-geometry-reasoning",     action="store_true")
    return parser.parse_args()


# ============================================================
# 메인 진입점
# ============================================================

def main() -> None:
    """원클릭 Colab 실행 파이프라인의 진입점.

    순서대로 Drive 마운트 → 의존성 확인 → 데이터셋 준비 →
    학습 파이프라인 실행 → 결과 Drive 저장을 수행한다.
    """
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if running_in_colab():
        print_stage("Mounting Google Drive")
        ensure_drive_mounted()

    print_stage("Checking Python dependencies")
    ensure_python_deps()
    print_runtime_summary()

    # 데이터셋 ZIP 경로 확인
    drive_zip_path = Path(args.drive_zip_path).expanduser()
    if not drive_zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {drive_zip_path}")

    # 로컬 경로 구성
    local_root = Path(args.local_root).expanduser()
    local_zip_path = local_root / "input_zip" / drive_zip_path.name
    extract_root = local_root / "data"
    run_dir = local_root / "runs" / "final"
    drive_output_root = Path(args.drive_output_root).expanduser()

    print(f"Project root: {project_root}", flush=True)
    print(f"Drive zip path: {drive_zip_path}", flush=True)
    print(f"Local runtime root: {local_root}", flush=True)

    print_stage("Copying dataset zip to Colab local disk")
    copied_zip = copy_zip_to_local(drive_zip_path, local_zip_path)

    print_stage("Extracting dataset zip")
    dataset_root = extract_dataset_zip(
        copied_zip, extract_root, force_reextract=args.force_reextract
    )

    print(f"Copied zip: {copied_zip}", flush=True)
    print(f"Resolved dataset root: {dataset_root}", flush=True)
    print(f"Local run dir: {run_dir}", flush=True)

    run_pipeline(project_root, dataset_root, run_dir, args)

    print_stage("Copying artifacts back to Drive")
    copy_artifacts_to_drive(run_dir, drive_output_root, dataset_root, project_root, args)


if __name__ == "__main__":
    main()
