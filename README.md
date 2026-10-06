# Adaptive Multi-Domain Diabetic Retinopathy Detection

This project implements a diabetic retinopathy (DR) screening and grading pipeline that combines:

- spatial retinal fundus encoding with EfficientNet-B0
- vessel structural features
- frequency-domain spectral features
- ARPS-adaptive image processing path selection
- binary screening and five-class ordinal grading
- consistency checking between binary and severity predictions
- model-specific binary and five-class Grad-CAM explainability for the spatial branch

## Project status

This repository contains trained binary and five-class checkpoints, locked-test reports, a CLI inference command, and a Streamlit demo. See [reports/project_status.md](reports/project_status.md) for the evidence-based audit, measured gaps, and limitations. This is a research/demo prototype, not a clinical diagnostic device.

## Prerequisites

Use the existing `.venv` in VS Code when available. From PowerShell at the project root:

```powershell
. .\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements-lock.txt
```

## Validate

```powershell
python scripts/test_imports.py
python -m pytest -q
```

The current full suite result is recorded in `logs/validation/pytest_latest.txt`.

## Run inference

Inference uses the existing checkpoints and does not train:

```powershell
python scripts/predict.py --image "C:\path\to\fundus_image.jpg"
```

JSON results and an original/processed/binary-Grad-CAM/five-class-Grad-CAM panel are written under `outputs/inference/`.

## Run the Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

The app accepts one or more JPG/JPEG/PNG files and uses the same `DRInferenceEngine` as the CLI.

## Training (optional; not required for inference)

```bash
python scripts/train_binary.py
python scripts/train_five_class.py
```

These scripts train the binary and severity models on the official split manifest and save checkpoints under `models/`.

## Data layout

- `data/splits/official_split_manifest.csv` — dataset manifest and labels
- `data/metadata/` — cached metadata and selection outputs
- `models/binary/` — binary checkpoint artifacts
- `models/five_class/` — five-class checkpoint artifacts
- `reports/` — evaluation summaries and predictions
- `outputs/inference/` — per-image CLI result JSON and visualization
- `figures/` — generated evaluation charts and saved Grad-CAM overlays

## Notes

- The verified inference path is CPU-based on Windows.
- The five-class locked-test macro-F1 is lower than the binary screening result; see `reports/five_class_results.csv`.
- Only APTOS-derived evaluation is available; cross-dataset validation and research/IP readiness work remain outstanding.
- Do not tune ARPS weights on the locked test set. Any calibration must use validation data and be documented.
