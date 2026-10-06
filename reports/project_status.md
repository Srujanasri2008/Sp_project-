# DR Project Status Audit

Audit date: 2026-10-03

This audit records only checked files, executed commands, and saved outputs. A successful inference on one image is not a model-performance estimate or a clinical validation.

## ✅ COMPLETE AND VERIFIED

- **Existing checkpoints:** `models/binary/best_model.pt` and `models/five_class/best_model.pt` exist, load, and contain trained state dictionaries. Recorded checkpoint metadata: 2/5 classes, seed 42; best epochs are 3 and 5 respectively.
- **Locked split:** `data/splits/official_split_manifest.csv` contains 3,662 unique IDs: train 2,563, validation 549, test 550. `official_split_indices.json` matches the manifest for all three partitions; pairwise overlap is zero.
- **Dataset loading and model branches:** dataset sample tests confirm spatial `(3,224,224)`, vessel `(1,224,224)`, and frequency `(1,224,224)` inputs. Real inference executes the branches.
- **ARPS execution:** a real image evaluated PATH_A, PATH_B, PATH_C, their configured constraints, and selected PATH_C. The code/config define normalized contrast, vessel similarity, detail, illumination and noise terms plus fallback behavior.
- **Saved-model CPU inference:** the CLI successfully loaded both checkpoints and predicted a real locked-test image. See `outputs/inference/aptos_2389_a80e77b7bb/`.
- **Exact CLI command used:** `python scripts/predict.py --image data/raw/images/aptos_2389.png`. For a user image: `python scripts/predict.py --image "C:\\path\\to\\fundus_image.jpg"`.
- **Full training requirement:** full training is not required to run inference or the app because both trained checkpoints already exist. Training scripts should only be run when intentionally reproducing or changing a model.
- **First verified real-image prediction:** image `aptos_2389.png` was valid; actual saved label was class 0. Binary output was Normal (98.63% Normal, 1.37% DR); five-class output was No DR (95.21%). Consistency was MATCH and Grad-CAM was generated. This is a single example only.
- **Inference outputs:** CLI writes candidate-level ARPS details and prediction probabilities to JSON and original/processed/binary-Grad-CAM/five-class-Grad-CAM panels to `outputs/inference/`.
- **Model test reports:** binary locked-test report `reports/binary_results.csv`: accuracy 0.9255, sensitivity 0.9032, specificity 0.9483, F1 0.9248, ROC-AUC 0.9678. Five-class locked-test report `reports/five_class_results.csv`: accuracy 0.6345, macro-F1 0.4269, QWK 0.6379.
- **Baseline/AGCWD/proposed comparison:** `reports/three_system_comparison.csv` contains locked-test results for all three. Accuracy/F1/ROC-AUC: baseline 0.9236/0.9236/0.9771; AGCWD 0.9218/0.9208/0.9695; proposed 0.9236/0.9247/0.9712.
- **Ablation, robustness, statistics and error analysis:** executed artifacts exist at `reports/ablation_results.csv`, `reports/robustness_results.csv`, `reports/mcnemar_pairwise_results.csv`, `reports/quality_association_results.csv`, and `reports/error_analysis/`.
- **Charts:** comparison ROC, three confusion matrices, ablation metrics and robustness plots were generated under `figures/` by `scripts/generate_figures.py`.
- **Grad-CAM:** separate binary and five-class maps were generated on the user-supplied image and are shown in `outputs/inference/testimg8fiveclspredprofdrsp_dbcc46ff99/inference_visualization.png`; both JSON flags are true.
- **Validation tests:** full command `python -m pytest -q` passed: 9 passed. Tests cover split integrity, dataset tensors, real CPU batch inference (two images), binary/five-class outputs, ARPS candidates, consistency, both Grad-CAM maps, invalid/corrupt inputs, missing-checkpoint rejection and inference artifact writing.
- **Streamlit startup:** requested command is `streamlit run app/streamlit_app.py`. Verified equivalent: `python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501`; `/_stcore/health` and `/` both returned HTTP 200.
- **Software versions:** `requirements-lock.txt` pins dependencies; `reports/reproducibility.json` records Python 3.12.10, Windows, CPU PyTorch 2.14.1, Streamlit 1.64.0 and other package versions.
- **Inference CLI and app share implementation:** both call `DRInferenceEngine`; app entrypoint is `app/streamlit_app.py` and delegates to `app/app.py`.

## 🟡 IMPLEMENTED BUT NOT FULLY VERIFIED

- **Streamlit single/batch upload interactions:** the app implements single and multiple image uploads through the shared engine, and the server starts successfully. Browser-driven upload clicks were not verified in this run. The two-image engine batch API itself is covered by pytest.
- **ARPS research specification:** formula, weights, constraints and candidate paths are executable in `src/processing/arps.py` and `config/arps_config.yaml`; the project lacks a complete paper-style derivation and independent validation of the weight choices.
- **Dataset source record:** code names Hugging Face `bumbledeep/aptos`, and local files/splits are present; an immutable dataset revision, primary dataset citation, license/provenance record and acquisition manifest were not found in the reviewed documentation.
- **Limitations disclosure:** the app has a research/demo disclaimer; a full limitations document covering class imbalance, external validity, calibration, and clinical-use restrictions is not complete.
- **Experiment reproducibility:** seed, package versions, training histories and scripts exist; a single immutable run manifest tying every result to source revision, full configuration, checkpoint hash and command is not established.

## 🟠 PARTIALLY IMPLEMENTED

- **Five-class model readiness:** checkpoint loads and inference works, but locked-test metrics are substantially weaker than binary screening (accuracy 0.6345, macro-F1 0.4269). Treat as a research prototype, not a dependable severity-grading product.
- **Publication evidence package:** internal comparison, ablation, robustness, statistical and error analyses exist, but the result set is single-dataset and lacks independent external validation.
- **Software usability:** CLI and Streamlit inference are implemented. CLI behavior, CPU path and server startup are verified; interactive browser flows are not.

## 🔴 MISSING

- **Research-gap traceability:** no documented literature review, research-gap matrix, or trace from cited prior work to claims and experiments.
- **ARPS weight provenance:** no auditable source/calibration report or validation-only calibration protocol establishing how the configured weights were selected. No ARPS weights were changed during this audit; do not tune on the locked test set.
- **Cross-dataset validation:** no independent dataset evaluation was found; available evaluations use the APTOS-derived split.
- **Patent readiness:** no prior-art search, invention disclosure, claim set, inventorship/ownership review, or freedom-to-operate assessment is present. No novelty or patentability conclusion is made; obtain qualified IP counsel before any filing or public disclosure decision.
- **Article readiness:** no manuscript, related-work citations, complete methods/provenance section, externally validated results, or publication-ready limitations discussion is present. The current artifacts are research results, not a submission-ready article.
- **Single next action:** run `streamlit run app/streamlit_app.py`, upload one retinal image you are authorized to use, and inspect its prediction and saved JSON/visualization under `outputs/inference/`. Do not start training unless intentionally reproducing or changing a model; do not calibrate ARPS weights against the locked test set.

## ❌ FAILING

- No current failing automated test or inference check was found in the checks listed above. A prior robustness run failed on duplicate quantile edges; the quartile handling was fixed and the current robustness report exists. Unsupported/non-image input is rejected, and missing checkpoints now fail closed instead of producing random-weight predictions.

