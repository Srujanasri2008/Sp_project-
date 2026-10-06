# VS Code Project Audit

Audit date: 2026-10-03

Scope: audit of research data/model artifacts, a Streamlit entrypoint import fix, and existing tests. No training, downloads, checkpoint changes, ARPS weight changes, preprocessing changes, or research-method changes were performed. The existing `scripts/predict.py` was inspected and left unchanged. After the audit, a local image supplied in the terminal context was run through that existing CLI.

## 1. Project status

The repository is an executable research/demo prototype with existing checkpoints, saved analysis reports, a CLI inference script, and a Streamlit UI. Streamlit's wrapper import was broken at audit start and was fixed without changing inference or research logic; the repaired entrypoint was then launched and checked. Patent readiness, article readiness, and clinical readiness are not established.

## 2. Environment status

- OS: Windows.
- Python: 3.12.10 from the project `.venv`.
- PyTorch: 2.14.1+cpu; CUDA unavailable.
- torchvision 0.29.1+cpu, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, OpenCV 5.0.0, Streamlit 1.64.0, pytest 9.1.1.
- Dependencies are also pinned in `requirements-lock.txt`.

## 3. Storage status

C: drive reported 132,405,514,240 bytes free (about 123.3 GiB). Audited folder sizes (recursive, excluding `.venv`): `data` 252,709,336 bytes; `models` 1,066,455,236 bytes; `reports` 310,525 bytes; `outputs` 400,215 bytes; `figures` 373,351 bytes; `logs` 30,886 bytes. Binary and five-class best checkpoints are approximately 61 MB each. No storage blocker found.

## 4. Dataset status

- Dataset code identifies the source as Hugging Face `bumbledeep/aptos`.
- Metadata and local image files exist. Audit found 3,662 manifest rows and zero referenced image files missing.
- Immutable source revision, primary dataset citation, license statement, and acquisition provenance are not fully recorded in project documentation.

## 5. Split status

- Manifest: `data/splits/official_split_manifest.csv`.
- Counts: train 2,563; validation 549; test 550; total 3,662.
- IDs are unique; `official_split_indices.json` exactly matches the manifest; all pairwise split overlaps are zero; seed is 42.
- No test-set tuning was performed in this audit.

## 6. ARPS status

- Existing implementation: `src/processing/arps.py`; frozen settings: `config/arps_config.yaml`.
- PATH_A, PATH_B, PATH_C evaluation, score, constraints, and fallback logic are implemented.
- Numeric weights and constraints are present. Their derivation/provenance and a validation-only calibration record are missing. No weights, preprocessing, or ARPS logic were changed.

## 7. Model status

- `models/binary/best_model.pt` exists (61,166,927 bytes), with a 2-class trained checkpoint, seed 42, recorded best epoch 3.
- `models/five_class/best_model.pt` exists (61,176,719 bytes), with a 5-class trained checkpoint, seed 42, recorded best epoch 5.
- Both checkpoints load through `DRInferenceEngine` on CPU. Missing checkpoints now fail closed rather than returning random-weight predictions.

## 8. Binary results

From `reports/binary_results.csv`, locked-test results: accuracy 0.9255, sensitivity 0.9032, specificity 0.9483, F1 0.9248, ROC-AUC 0.9678. These are saved project results; this audit did not retrain or retune the model.

## 9. Five-class status

The checkpoint exists and is loadable. From `reports/five_class_results.csv`, locked-test accuracy is 0.6345, macro-F1 0.4269, and QWK 0.6379. Severity inference is implemented, but these results are notably weaker than binary screening and should be treated cautiously.

## 10. Ablation status

`reports/ablation_results.csv` contains spatial-only, spatial+vessel, spatial+frequency, and full-fusion results. The CSV exists from completed experiment runs; no ablation was rerun in this audit.

## 11. Robustness status

`reports/robustness_results.csv` contains test-set quality-stratified results. A prior duplicate-quantile-edge failure was corrected before this audit; the current saved artifact exists. This is not cross-dataset robustness validation.

## 12. Statistics status

`reports/mcnemar_pairwise_results.csv` and `reports/quality_association_results.csv` exist from the completed analysis. They use the available APTOS-derived evaluation predictions and do not establish causal effects.

## 13. Error analysis status

`reports/error_analysis/error_summary.csv` and `full_error_audit_manifest.csv` exist from the completed comparison error analysis.

## 14. Grad-CAM status

Both binary-model and five-class-model Grad-CAM are implemented in the shared inference pipeline. On the supplied local image, both maps were generated and shown as distinct panels in `outputs/inference/testimg8fiveclspredprofdrsp_dbcc46ff99/inference_visualization.png`; `result.json` records both generation flags as true. Grad-CAM is a saliency visualization, not validated lesion localization.

## 15. Streamlit status

- `app/streamlit_app.py` exists. Its initial `from app.app import main` failed under Streamlit with `ModuleNotFoundError: 'app' is not a package`.
- The wrapper was repaired using a direct file-based import of `app/app.py`; `app/app.py` calls the same `DRInferenceEngine` as the CLI and loads the saved binary/five-class checkpoints. It does not create a separate predictor or random predictions.
- Verified launch: `python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501`; health and root endpoints both returned HTTP 200.
- Streamlit AppTest with a 30-second timeout rendered one title and one uploader with zero app exceptions. Its initial default 3-second attempt timed out during startup; the extended run passed.
- Exact normal launch command: `streamlit run app/streamlit_app.py`.
- Single/multiple upload UI is implemented, but browser-driven upload interactions were not exercised in this audit.

## 16. Test status

Existing tests were run with `python -m pytest -q`:

- Passed: 9
- Failed: 0
- Skipped: 0
- Errors: 0

Runtime: 7.51 seconds. Tests cover dataset tensor branches, split integrity, invalid/corrupt inputs, missing-checkpoint rejection, CPU two-image batch inference, both model outputs, ARPS candidates, consistency, both Grad-CAM overlays, and inference artifact writing. The test suite does not replace browser interaction tests or external validation.

## 17. Missing stages

- Browser-driven Streamlit single and batch upload tests.
- Independent cross-dataset validation.
- Literature/research-gap traceability and full dataset provenance/licensing record.
- Auditable ARPS weight provenance and validation-only calibration documentation.
- A complete, versioned experiment manifest binding commands/configuration/checkpoint hashes to each result.

## 18. Exact commands to run the existing project

From the project root in VS Code PowerShell:

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts/test_imports.py
python -m pytest -q
python scripts/generate_figures.py
streamlit run app/streamlit_app.py
```

Previously completed analysis scripts (do not train) can be run with:

```powershell
python scripts/run_robustness.py
python scripts/run_statistics.py
python scripts/run_error_analysis.py
```

Comparison and ablation scripts train models and are not needed to run inference; do not launch them for this audit.

## 19. Exact command for single-image inference

The existing equivalent inference script is `scripts/predict.py`; it was not modified in this task. After obtaining a local retinal fundus JPG/JPEG/PNG path, run:

```powershell
python scripts/predict.py --image "C:\Users\<user>\Pictures\fundus.jpg"
```

It loads both existing checkpoints, uses the shared preprocessing/inference pipeline, prints ARPS candidate scores, selected path, predictions/probabilities, confidence, and consistency, and saves JSON plus an original/processed/two-Grad-CAM visualization under `outputs/inference/`. The supplied local image completed successfully: PATH_A, ARPS 1.2591, binary DR 99.22%, five-class PDR 32.70%, consistency MATCH, both maps generated. This is the model output for one image, not a performance statistic or clinical assessment.

## 20. Research limitations

Evaluation is based on the APTOS-derived split only. No independent dataset evaluation, clinical validation, prospective testing, or documented calibration study was found. Five-class results are weaker than binary results. Softmax probabilities/confidence are not evidence of calibration. This software is for research/demo use and must not be treated as a diagnosis or used alone for clinical decisions.

## 21. Patent/article readiness status

- **Patent:** not assessed as patentable. No documented prior-art search, invention disclosure, claim set, inventorship/ownership review, or freedom-to-operate review was found. Do not claim novelty or patentability; seek qualified IP counsel.
- **Article:** not submission-ready. A research-gap/related-work review, complete dataset citation/license/revision, ARPS weight provenance, full methods and limitations narrative, external validation, and manuscript are missing.

## A. What is COMPLETE

- Existing checkpoints and saved APTOS-split test/analysis artifacts.
- CPU inference engine and CLI implementation; Streamlit wrapper import fix and HTTP startup check.
- Locked split integrity and complete local dataset file presence.
- Existing pytest suite: 9 passed, 0 failed/skipped/errors.

## B. What is INCOMPLETE

- User-provided local-image inference and browser-driven Streamlit upload verification.
- Dataset/weight provenance, cross-dataset validation, research-gap traceability, article preparation, and IP review.

## C. What is BROKEN

- The Streamlit wrapper import was broken and has now been repaired; the repaired server returned HTTP 200. No current failing pytest was observed.

## D. What I should run next

Open `outputs/inference/testimg8fiveclspredprofdrsp_dbcc46ff99/inference_visualization.png` and inspect the separate binary and five-class maps. For another local image, use the exact CLI command below. No training is needed.

## E. Exact command for single-image prediction

```powershell
python scripts/predict.py --image "C:\Users\SRUJANA BANDARU\Downloads\testimg8fiveclspredprofdrsp.jpeg" --device cpu
```
