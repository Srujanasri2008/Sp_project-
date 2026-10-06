# VS Code Run Guide

This guide uses the existing trained checkpoints. It does not retrain models by default.

## 1. Open the project and choose Python

1. In VS Code, choose **File > Open Folder** and open the `DR_Project` root folder.
2. Open **Terminal > New Terminal** and select PowerShell.
3. Select the existing interpreter with **Ctrl+Shift+P > Python: Select Interpreter**, then choose:

```text
.venv\Scripts\python.exe
```

If the environment is missing, create it from the project root:

```powershell
py -3.12 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

For an existing environment, activate it:

```powershell
. .\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for this terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

Check that Python points to the project environment:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

## 2. Validate before use

Run dependency checks and the complete automated suite:

```powershell
python scripts/test_imports.py
python -m pytest -q
```

Current audit result: 9 tests passed. Pytest output is saved by the validation runner:

```powershell
python scripts/run_validation.py
```

The log and JSON summary are written under `logs/validation/`.

## 3. Run the command-line inference test

Use a retinal fundus image in JPG, JPEG, or PNG format:

```powershell
python scripts/predict.py --image "C:\path\to\your\fundus_image.jpg"

python scripts/predict.py --image "C:\Users\SRUJANA BANDARU\Pictures\fundus.jpg" --device cpu

python scripts/predict.py --image "PASTE_THE_REAL_IMAGE_PATH_HERE" --device cpu
```

CPU is the default. To state it explicitly:

```powershell
python scripts/predict.py --image "C:\path\to\your\fundus_image.jpg" --device cpu
```

The command uses both existing checkpoints and the same `DRInferenceEngine` as the web app. It prints:

- input validity
- selected ARPS path and score
- each candidate path's score and constraint result
- binary label and Normal/DR probabilities
- five-class label and class probabilities
- consistency status
- binary-model Grad-CAM for the binary predicted class
- five-class-model Grad-CAM for the five-class predicted severity

It saves `result.json` and `inference_visualization.png` under a unique directory in:

```text
outputs/inference/
```

The visualization shows the original image, selected-path processed image, binary-model Grad-CAM, and five-class-model Grad-CAM. The JSON records whether each map was generated and contains candidate metrics and complete prediction details. A non-image/unsupported file is rejected; a missing model checkpoint exits with an error and will not fall back to random weights.

A known local sample is available for a quick check:

```powershell
python scripts/predict.py --image "data/raw/images/aptos_2389.png"
```

## 4. Run the user-facing Streamlit app

From the project root and with the environment active:

```powershell
streamlit run app/streamlit_app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`. Upload one image or select multiple supported images. The app calls the same inference engine as the CLI. Each valid image gets a **Prediction** tab with screening/severity/confidence/path summaries and both probability distributions, an **Image review** tab with original/processed images and separate binary/five-class Grad-CAMs, and a **Processing details** tab with candidate scores, constraints, and quality measurements. The prediction tab also offers a result JSON download.

If port 8501 is already occupied, choose another port:

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```

The app records prediction metadata in `logs/inference/predictions.csv` and separately named binary/five-class Grad-CAM overlays in `figures/gradcam/`. It stores a truncated image digest rather than the uploaded filename or original upload in that log. CLI outputs include the source path in the local JSON result; review that file before sharing it.

The interface is a research/demo tool, not a medical device or diagnosis. Use only images you are authorized to process, and do not use its output alone for clinical decisions.

## 5. Check model performance without retraining

Previously completed locked-test results are in:

- Binary model: `reports/binary_results.csv` and `reports/binary_test_predictions.csv`
- Five-class model: `reports/five_class_results.csv` and `reports/five_class_test_predictions.csv`
- Baseline/AGCWD/proposed comparison: `reports/three_system_comparison.csv`
- Ablation: `reports/ablation_results.csv`
- Robustness: `reports/robustness_results.csv`
- Statistical tests: `reports/mcnemar_pairwise_results.csv` and `reports/quality_association_results.csv`
- Error analysis: `reports/error_analysis/`

A one-image prediction tests execution, not performance. Use the saved locked-test reports for model-level performance. Current five-class results are weaker than binary screening and should be interpreted cautiously.

To regenerate report figures from existing CSVs, without training:

```powershell
python scripts/generate_figures.py
```

## 6. Verify that outputs were created

After CLI inference, check the most recent output directory and read its result JSON:

```powershell
Get-ChildItem outputs/inference -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem outputs/inference -Recurse -File | Select-Object FullName, Length
Get-Content (Get-ChildItem outputs/inference -Filter result.json -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

Expected inference output files are `result.json` and `inference_visualization.png`. In the JSON, verify `is_valid`, `selected_processing_path`, `arps_score`, all three `candidate_scores`, both predictions and probability objects, `consistency`, `binary_gradcam_generated`, and `five_class_gradcam_generated`.

Check the saved locked-test model results without retraining:

```powershell
Import-Csv reports/binary_results.csv | Format-List
Import-Csv reports/five_class_results.csv | Format-List
Import-Csv reports/three_system_comparison.csv | Format-Table
Import-Csv reports/ablation_results.csv | Format-Table
Import-Csv reports/robustness_results.csv | Select-Object -First 12 | Format-Table
Import-Csv reports/mcnemar_pairwise_results.csv | Format-Table
Import-Csv reports/quality_association_results.csv | Format-Table
Import-Csv reports/error_analysis/error_summary.csv | Format-Table
```

Verify the checkpoint and report files exist:

```powershell
Test-Path models/binary/best_model.pt
Test-Path models/five_class/best_model.pt
Test-Path reports/binary_results.csv
Test-Path reports/five_class_results.csv
Test-Path reports/three_system_comparison.csv
Test-Path reports/ablation_results.csv
Test-Path reports/robustness_results.csv
Test-Path reports/mcnemar_pairwise_results.csv
Test-Path reports/error_analysis/error_summary.csv
```

Each should print `True`. For the automated suite, `python -m pytest -q` should finish with `9 passed` for the audited code state; use the actual current terminal result if tests have changed.

## 7. Full pipeline reproduction commands (optional)

The data, checkpoints, predictions, and analysis reports already exist. Do not run this sequence just to use or test the app. It can download data, recompute a full-dataset cache, retrain models, take hours on CPU, and overwrite experiment artifacts.

Only for an intentional clean reproduction, from project root:

```powershell
python scripts/setup_environment.py
python -m pip install -r requirements-lock.txt
python scripts/prepare_dataset.py
python scripts/verify_processing.py
python scripts/cache_arps_selection.py
python scripts/verify_models.py
python scripts/train_binary.py
python scripts/train_five_class.py
python scripts/run_comparisons.py
python scripts/run_ablation.py
python scripts/run_robustness.py
python scripts/run_statistics.py
python scripts/run_error_analysis.py
python scripts/generate_figures.py
python -m pytest -q
```

`prepare_dataset.py` downloads the configured Hugging Face dataset. Skip it when the existing local dataset is to be used. The cache, training, comparison, and ablation commands are CPU-expensive. The report files above are the artifacts to inspect after each analysis stage.

## 8. Training and full experiment commands (optional)

Training is **not required** to run inference because both checkpoints are present. CPU training takes a long time and should only be started if you intentionally want to reproduce or change a model. These commands can resume/write checkpoints and may update model artifacts:

```powershell
python scripts/train_binary.py
python scripts/train_five_class.py
```

Comparison and ablation scripts also train models and may take many hours on CPU:

```powershell
python scripts/run_comparisons.py
python scripts/run_ablation.py
```

Analysis scripts use saved predictions/reports and do not train models:

```powershell
python scripts/run_robustness.py
python scripts/run_statistics.py
python scripts/run_error_analysis.py
```

Do not alter ARPS weights based on locked-test results. Any proposed calibration must be designed using validation data only, documented, and followed by one final untouched test evaluation.
