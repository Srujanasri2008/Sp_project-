ADAPTIVE MULTI-DOMAIN SIGNAL PROCESSING AND DEEP LEARNING FOR
DIABETIC RETINOPATHY DETECTION FROM RETINAL FUNDUS IMAGES
================================================================

PROJECT AND TEAM
----------------
Project: DR fundus-image screening and five-class severity research/demo system.
Team members: [ADD EACH TEAM MEMBER'S FULL NAME, STUDENT ID, AND CONTRIBUTION]
Course/review: [ADD COURSE AND REVIEW DETAILS]

GitHub project:
https://github.com/Srujanasri2008/Sp_project-

IMPORTANT SCOPE NOTICE
----------------------
This is a research/demo prototype, not a medical device or diagnosis. Model
predictions can be wrong. Do not use them alone for clinical decisions. A
qualified eye-care professional must assess the patient and image.

WHAT IS INCLUDED / WHAT IS NOT
------------------------------
The repository includes application/source code, configuration, saved reports,
plots, split metadata, training histories, and the binary/five-class inference
checkpoints. The two inference checkpoints use Git LFS; install Git LFS and run
`git lfs pull` if they were not downloaded during clone.

The raw image files and Python virtual environment are intentionally excluded
from GitHub. Inference on a local JPG/JPEG/PNG image does not require the APTOS
images. Dataset loading, pytest's dataset tests, or retraining do require those
images; download them as described below. Comparison/ablation checkpoints and
per-image inference outputs are also excluded to keep the repository size
manageable.

1. REQUIREMENTS
---------------
Tested environment:
- Windows 10/11, PowerShell
- Python 3.12.10
- CPU PyTorch 2.14.1; CUDA was not available in the verified environment
- Git and Git LFS for cloning the checkpoints

Python packages are pinned in `requirements-lock.txt`. `requirements.txt`
contains minimum-version requirements. Use Python 3.12 for the closest match to
the verified setup. First install Python 3.12 and Git LFS if they are not
already installed.

2. CLONE AND INSTALL IN VS CODE
-------------------------------
In VS Code choose File > Open Folder and select the cloned `Sp_project-`
folder. Open Terminal > New Terminal (PowerShell), then run from the project
root:

    git lfs pull
    py -3.12 -m venv .venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    . .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements-lock.txt
    python scripts/test_imports.py

The execution-policy command applies only to the current PowerShell process.
If `.venv` already exists, skip the venv creation command and activate it.

3. QUICK INFERENCE (NO TRAINING OR DATASET DOWNLOAD)
----------------------------------------------------
The trained checkpoints are:
- `models/binary/best_model.pt`
- `models/five_class/best_model.pt`

Run prediction on your own local retinal fundus image:

    python scripts/predict.py --image "C:\full\path\to\fundus.jpg" --device cpu

Replace the example with a real existing JPG, JPEG, or PNG path. This command
loads both saved models and uses the shared inference pipeline. It does not
train or change model weights. It prints image validation, selected processing
path, ARPS score and candidate scores, binary prediction and probabilities,
five-class prediction and probabilities, consistency, and Grad-CAM status.
It generates separate binary-model and five-class-model Grad-CAM maps when
available.

Output is saved under:

    outputs/inference/<image-name>_<digest>/
        result.json
        inference_visualization.png

The visualization includes the original image, processed image, binary
Grad-CAM, and five-class Grad-CAM. `result.json` contains the detailed ARPS
candidate results and model outputs. The JSON includes the local source path;
review it before sharing. The model's probabilities are not established as
calibrated clinical risks.

4. RUN THE USER APPLICATION
---------------------------
From the project root with `.venv` active:

    streamlit run app/streamlit_app.py

Open the local URL printed by Streamlit, normally http://localhost:8501. Upload
one or more fundus images. The Streamlit interface calls the same
`DRInferenceEngine` as `scripts/predict.py`; it does not implement separate
preprocessing or random predictions. The interface shows prediction summaries,
probability bars, the original and processed images, both model-specific
Grad-CAMs, ARPS candidate/quality details, and a JSON download.

5. DATA SOURCE AND DOWNLOAD
---------------------------
The project preparation script uses the Hugging Face dataset:

    https://huggingface.co/datasets/bumbledeep/aptos

The dataset card identifies it as 3,662 resized APTOS retinal images and cites
the original APTOS 2019 Blindness Detection competition:

    https://www.kaggle.com/c/aptos2019-blindness-detection/data

The Hugging Face dataset card also lists a derived Kaggle source:

    https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data/

To retrieve/process the dataset using this project's code, install the Python
requirements, ensure an internet connection, then run from the root:

    python scripts/prepare_dataset.py

That script calls Hugging Face `datasets.load_dataset("bumbledeep/aptos")`,
expects 3,662 records, saves 224x224 PNG files under `data/raw/images/`, and
writes metadata and deterministic split files under `data/`. It may recreate
overwrite those metadata/split files. Check the current dataset-card and source
terms before redistribution; this repository does not redistribute the raw
images. Dataset license/provenance should be reviewed by the project team
before public redistribution.

6. FULL REPRODUCTION / EXECUTION ORDER
--------------------------------------
A. Environment setup and imports:

    python scripts/setup_environment.py
    python scripts/test_imports.py

B. Dataset preparation (downloads data; skip if already prepared):

    python scripts/prepare_dataset.py

C. Processing, ARPS and architecture checks:

    python scripts/verify_processing.py
    python scripts/cache_arps_selection.py
    python scripts/verify_models.py

D. Training (optional, CPU-expensive; updates checkpoints/reports):

    python scripts/train_binary.py
    python scripts/train_five_class.py

E. Comparison and ablation (also train models and can take many hours on CPU):

    python scripts/run_comparisons.py
    python scripts/run_ablation.py

F. Analysis from generated prediction files, plots and validation:

    python scripts/run_robustness.py
    python scripts/run_statistics.py
    python scripts/run_error_analysis.py
    python scripts/generate_figures.py
    python -m pytest -q

Do not run training merely to use the app. Existing inference checkpoints are
already provided. The test suite includes dataset-loading tests, so download
and prepare the dataset before running pytest in a fresh clone.

7. FILE ROLES AND HOW THE COMPONENTS CONNECT
--------------------------------------------
ROOT FILES
- `README.txt`: this plain-text setup, file-role, and execution guide.
- `README.md`: Markdown overview and quick usage.
- `requirements.txt`: package minimum versions.
- `requirements-lock.txt`: pinned package versions used for reproducibility.
- `.gitignore`: excludes the local environment, raw images, temporary outputs,
  and redundant experiment checkpoints.
- `.gitattributes`: sends the two inference checkpoints through Git LFS.

APPLICATION
- `app/streamlit_app.py`: Streamlit launch wrapper; delegates to `app/app.py`.
- `app/app.py`: upload/review interface, calls the shared inference engine,
  displays predictions/ARPS/Grad-CAM, and saves inference metadata locally.

CONFIGURATION
- `config/config.yaml`: project, dataset, model, training, device, and split
  defaults.
- `config/arps_config.yaml`: ARPS weights, constraints, fallback, and candidate
  processing settings. Do not tune using the locked test set.

DATA
- `data/metadata/dataset_metadata.csv`: dataset image IDs, labels and paths.
- `data/metadata/arps_selection_cache.csv`: cached ARPS selections and quality
  values used by processing/analysis.
- `data/splits/official_split_manifest.csv`: per-image split and labels.
- `data/splits/official_split_indices.json`: deterministic split IDs and counts.
- `data/raw/images/`: downloaded/resized retinal images; intentionally not in
  GitHub, generated by `scripts/prepare_dataset.py`.

SOURCE PACKAGE (`src/`)
- `src/data/dataset.py`: loads image files, split labels and spatial/vessel/
  frequency tensors for training/evaluation.
- `src/processing/validation.py`: validates paths, image bytes and image data.
- `src/processing/quality.py`: retinal mask and image-quality measurements.
- `src/processing/candidates.py`: PATH_A, PATH_B, PATH_C and AGCWD operations.
- `src/processing/arps.py`: candidate scoring, constraints and path selection.
- `src/processing/vessel.py`: vessel representation extraction.
- `src/processing/frequency.py`: frequency/FFT representation extraction.
- `src/models/network.py`: spatial encoder, optional domain branches and fusion.
- `src/training/trainer.py`: training, validation, checkpoint and evaluation flow.
- `src/evaluation/metrics.py`: classification metrics and output consistency.
- `src/explainability/gradcam.py`: Grad-CAM maps and image overlays.
- `src/inference/engine.py`: shared validation, preprocessing, ARPS, both models,
  consistency and both Grad-CAMs for CLI and Streamlit.
- `src/utils/reproducibility.py`: seeds and reproducibility helpers.

SCRIPTS (`scripts/`)
- `setup_environment.py`: system checks and creation of expected directories/
  virtual environment.
- `prepare_dataset.py`: obtains APTOS data and creates local images/manifest.
- `cache_arps_selection.py`: computes a saved ARPS choice for each dataset image.
- `verify_processing.py`: validates image processing, candidates, ARPS, vessel
  and frequency processing on an existing sample.
- `verify_models.py`: model-configuration forward-pass and Grad-CAM target check.
- `train_binary.py`: trains/evaluates the binary DR model; not a quick inference
  command.
- `train_five_class.py`: trains/evaluates the five-class model; not a quick
  inference command.
- `predict.py`: single local image inference using existing checkpoints.
- `run_comparisons.py`: trains and compares baseline, AGCWD and proposed setup.
- `run_ablation.py`: trains/evaluates feature-domain ablations.
- `run_robustness.py`: quality-stratified test-set analysis.
- `run_statistics.py`: paired statistical tests and quality association analysis.
- `run_error_analysis.py`: comparative error summary and image-level audit.
- `generate_figures.py`: plots saved comparison, ablation and robustness CSVs.
- `run_validation.py`: runs pytest and records output under `logs/validation/`.
- `test_imports.py`: checks dependency imports.
- `storage_audit.py`: reports storage use.

TESTS
- `tests/test_pipeline_smoke.py`: dataset tensors, split integrity, image
  validation, real CPU batch inference, checkpoint guard, ARPS, predictions,
  consistency, Grad-CAM and output writing.

MODEL FILES
- `models/binary/best_model.pt`: trained binary screening checkpoint.
- `models/five_class/best_model.pt`: trained five-class grading checkpoint.
- Other comparison/ablation checkpoints are not included in this GitHub package;
  their training scripts and report results are included.

REPORTS, FIGURES, LOGS, AND OUTPUTS
- `reports/binary_results.csv`, `reports/five_class_results.csv`: saved model
  test summaries.
- `reports/three_system_comparison.csv`: baseline/AGCWD/proposed comparison.
- `reports/ablation_results.csv`: domain ablation metrics.
- `reports/robustness_results.csv`: quality-stratified metrics.
- `reports/mcnemar_pairwise_results.csv`: paired system tests.
- `reports/quality_association_results.csv`: quality/correctness associations.
- `reports/error_analysis/`: error taxonomy and per-image audit.
- `reports/project_status.md`, `reports/vscode_project_audit.md`: audit and
  known gaps.
- `reports/reproducibility.json`: recorded environment/seed information.
- `figures/`: generated comparison, confusion-matrix, ROC, ablation and
  robustness charts; Grad-CAM images are generated locally during inference.
- `logs/training/`: saved training histories.
- `logs/inference/`, `logs/validation/`: generated locally when app/validation
  commands run; local per-image outputs are not committed.
- `outputs/inference/`: CLI JSON result and visualization, generated locally.

8. EXPECTED RESULTS AND WHERE THEY ARE SAVED
--------------------------------------------
For one image:
- Terminal: predictions, probabilities, selected ARPS path, candidate scores,
  consistency, and Grad-CAM availability.
- `outputs/inference/<image>_<digest>/result.json`: detailed inference result.
- `outputs/inference/<image>_<digest>/inference_visualization.png`: original,
  processed, binary Grad-CAM, and five-class Grad-CAM panels.

For model/experiment evaluation:
- `reports/binary_results.csv` and `reports/five_class_results.csv`.
- Comparison/ablation/robustness/statistical/error outputs under `reports/`.
- Generated plots under `figures/`.
- Training histories under `logs/training/`.

For validation:

    python -m pytest -q

The audited test suite passed 9 tests in the development workspace. A fresh
clone must first download/prepare the dataset because raw image files are not
tracked in GitHub.

9. KNOWN LIMITATIONS AND EXECUTION ISSUES
-----------------------------------------
- CPU training and full comparisons/ablations take a long time. Inference does
  not require retraining when the checkpoints are present.
- The current test set is an APTOS-derived split. Cross-dataset, prospective,
  and clinical validation have not been performed.
- The binary test metrics are stronger than five-class grading. Five-class
  locked-test accuracy was 0.6345 and macro-F1 was 0.4269; interpret severity
  estimates cautiously.
- Pairwise baseline/AGCWD/proposed results were not statistically significant
  after multiple-comparison correction.
- ARPS weight derivation/provenance is not fully documented; weights were not
  changed for this package. Never tune them against locked test data.
- Confidence scores are not established as calibrated clinical probabilities.
  Grad-CAM is not lesion segmentation or validated pathology localization.
- Dataset citation, revision and licensing/provenance should be checked before
  redistributing data. Raw images are not included in the GitHub repo.
- The app is a local research/demo interface, not a secured clinical service.

END OF README
