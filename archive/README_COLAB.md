# Colab One-Click

## What it does
- uses project zip path `/content/drive/MyDrive/physics_solution.zip`
- uses dataset zip path `/content/drive/MyDrive/open (7).zip`
- copies both zips to Colab local storage first
- extracts the project locally to `/content/physics_solution`
- extracts the dataset locally
- runs the full pipeline
- copies the final outputs back to Drive
- project zip should contain code only; do not include `.venv`, local caches, or dataset files

## Colab steps

### 1) Make a slim project zip on your Mac
```bash
cd /Users/mgo/Downloads/physics_solution
./make_colab_bundle.sh
```

기본 출력:
- `~/Desktop/physics_solution.zip`
- zip 안에는 `PhysicsSolution_Colab_OneClick.ipynb` 도 같이 들어갑니다.

### 2) Put two zip files in Drive
- project zip example path: `/content/drive/MyDrive/physics_solution.zip`
- dataset zip required default path: `/content/drive/MyDrive/open (7).zip`

### 3) Run one Colab cell
```bash
from google.colab import drive
drive.mount("/content/drive")

import shutil
from pathlib import Path

PROJECT_ZIP = Path("/content/drive/MyDrive/physics_solution.zip")
DATASET_ZIP = Path("/content/drive/MyDrive/open (7).zip")
LOCAL_PROJECT_ROOT = Path("/content/physics_solution")
LOCAL_PROJECT_ZIP = Path("/content/physics_solution_project.zip")

if LOCAL_PROJECT_ROOT.exists():
    shutil.rmtree(LOCAL_PROJECT_ROOT)
if LOCAL_PROJECT_ZIP.exists():
    LOCAL_PROJECT_ZIP.unlink()

shutil.copy2(PROJECT_ZIP, LOCAL_PROJECT_ZIP)
shutil.unpack_archive(str(LOCAL_PROJECT_ZIP), "/content")

%cd /content/physics_solution
!bash run_colab_oneclick.sh --drive-zip-path "{DATASET_ZIP}" --batch-size 12 --num-workers 2
```

이 셀 하나로:
- Drive 마운트
- project zip 로컬 압축해제
- dataset zip 로컬 복사/압축해제
- GPU 학습
- 결과 Drive 백업

진행 상태는 Colab 출력창에서 실시간으로 보입니다.
- `extract-motion` 진행바
- `train 1/12`, `valid 1/12` 진행바
- CUDA GPU 이름과 VRAM 출력

## Outputs
- local run dir: `/content/physics_solution_runtime/runs/final`
- Drive backup dir: `/content/drive/MyDrive/physics_solution_outputs/run_YYYYMMDD_HHMMSS`
- final submission: `/content/drive/MyDrive/physics_solution_outputs/run_YYYYMMDD_HHMMSS/submission.csv`

## Optional overrides
```bash
%cd /content/physics_solution
!bash run_colab_oneclick.sh \
  --drive-zip-path "/content/drive/MyDrive/open (7).zip" \
  --batch-size 12 \
  --epochs 12 \
  --num-folds 5 \
  --num-workers 2
```

기하 브랜치를 실험할 때만:
```bash
%cd /content/physics_solution
!bash run_colab_oneclick.sh --enable-geometry-reasoning
```
