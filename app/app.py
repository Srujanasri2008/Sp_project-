from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import cv2
import numpy as np
import streamlit as st

from src.inference.engine import DRInferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def get_engine():
    return DRInferenceEngine(
        binary_model_path=PROJECT_ROOT / "models" / "binary" / "best_model.pt",
        five_class_model_path=PROJECT_ROOT / "models" / "five_class" / "best_model.pt",
        device="cpu",
    )


def preview_uploaded_image(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def save_inference_artifacts(result, image_digest):
    inference_dir = PROJECT_ROOT / "logs" / "inference"
    gradcam_dir = PROJECT_ROOT / "figures" / "gradcam"
    inference_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    gradcam_paths = []
    if result.gradcam_overlay_rgb is not None:
        gradcam_file = gradcam_dir / f"{image_digest}_binary.png"
        overlay_bgr = cv2.cvtColor(result.gradcam_overlay_rgb, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(str(gradcam_file), overlay_bgr):
            gradcam_paths.append(str(gradcam_file.relative_to(PROJECT_ROOT)))
    if result.five_class_gradcam_overlay_rgb is not None:
        gradcam_file = gradcam_dir / f"{image_digest}_five_class.png"
        overlay_bgr = cv2.cvtColor(result.five_class_gradcam_overlay_rgb, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(str(gradcam_file), overlay_bgr):
            gradcam_paths.append(str(gradcam_file.relative_to(PROJECT_ROOT)))

    log_path = inference_dir / "predictions.csv"
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_digest": image_digest,
        "is_valid": result.is_valid,
        "selected_path": result.selected_path,
        "binary_prediction": result.binary_prediction_str,
        "binary_probabilities": json.dumps(result.binary_probabilities, sort_keys=True),
        "severity_prediction": result.five_class_prediction_str,
        "severity_probabilities": json.dumps(result.five_class_probabilities, sort_keys=True),
        "confidence": result.confidence,
        "consistency": json.dumps(result.consistency, sort_keys=True),
        "gradcam_files": json.dumps(gradcam_paths),
        "error_message": result.error_message or "",
    }
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=row.keys())
        if log_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def build_result_payload(result, image_digest):
    candidates = {}
    if result.arps_details is not None:
        candidates = {
            name: {
                "score": detail.arps_score,
                "constraints_pass": detail.satisfies_constraints,
                "vessel_similarity": detail.vessel_similarity,
                "delta_noise": detail.delta_noise,
                "delta_contrast": detail.delta_contrast,
                "violations": detail.constraint_violations,
            }
            for name, detail in result.arps_details.candidates.items()
        }
    return {
        "image_digest": image_digest,
        "selected_path": result.selected_path,
        "arps_score": result.arps_score,
        "candidate_scores": candidates,
        "quality_metrics": result.quality_metrics,
        "binary_prediction": result.binary_prediction_str,
        "binary_probabilities_percent": result.binary_probabilities,
        "five_class_prediction": result.five_class_prediction_str,
        "five_class_probabilities_percent": result.five_class_probabilities,
        "confidence": result.confidence,
        "consistency": result.consistency,
        "binary_gradcam_generated": result.gradcam_overlay_rgb is not None,
        "five_class_gradcam_generated": result.five_class_gradcam_overlay_rgb is not None,
    }


def render_probability(label, value):
    st.progress(min(max(float(value) / 100.0, 0.0), 1.0), text=f"{label}  ·  {value:.2f}%")


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #183635;
            --muted: #617876;
            --line: #d7e4e1;
            --accent: #137f78;
            --accent-dark: #0c625e;
            --surface: #ffffff;
            --canvas: #f3f7f6;
        }
        [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(243,247,246,.92); }
        [data-testid="stSidebar"] { background: #eaf2f0; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
        .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h1 { font-size: 2.15rem; line-height: 1.12; margin-bottom: .25rem; }
        h2 { font-size: 1.25rem; margin-top: .35rem; }
        h3 { font-size: 1.02rem; }
        [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: .8rem .95rem; border-radius: 8px; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.45rem; }
        [data-testid="stFileUploader"] section { background: var(--surface); border: 1px dashed #8eb8b2; border-radius: 8px; }
        [data-testid="stFileUploader"] section:hover { border-color: var(--accent); }
        [data-testid="stProgressBar"] > div > div { background: var(--accent); }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent-dark); border-bottom-color: var(--accent); }
        [data-testid="stDownloadButton"] button { border-color: var(--accent); color: var(--accent-dark); }
        .workspace-kicker { color: var(--accent-dark); font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .45rem; }
        .workspace-subtitle { color: var(--muted); font-size: 1rem; margin-top: .1rem; }
        .safety-note { border-left: 3px solid #d49b32; padding: .55rem .8rem; color: #665329; background: #fff8e8; border-radius: 0 6px 6px 0; font-size: .86rem; }
        .case-heading { padding-bottom: .55rem; border-bottom: 1px solid var(--line); margin-bottom: .85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Retinal Image Review", page_icon=":material/visibility:", layout="wide")
    inject_styles()

    binary_checkpoint = PROJECT_ROOT / "models" / "binary" / "best_model.pt"
    five_checkpoint = PROJECT_ROOT / "models" / "five_class" / "best_model.pt"
    with st.sidebar:
        st.markdown("### RETINAL REVIEW")
        st.caption("Research workspace")
        st.divider()
        st.markdown("**Model availability**")
        st.write(f"{'Ready' if binary_checkpoint.is_file() else 'Missing'} · Binary checkpoint")
        st.write(f"{'Ready' if five_checkpoint.is_file() else 'Missing'} · Five-class checkpoint")
        st.write("CPU inference")
        st.divider()
        st.markdown(
            "<div class='safety-note'>Research/demo output only. Not a diagnosis or a substitute for assessment by a qualified eye-care professional.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='workspace-kicker'>Fundus image workspace</div>", unsafe_allow_html=True)
    st.title("Diabetic retinopathy review")
    st.markdown(
        "<div class='workspace-subtitle'>Screening and severity estimates with image-specific processing and model explanations.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    uploaded_files = st.file_uploader(
        "Select fundus images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Choose one image or several JPG, JPEG, or PNG fundus images.",
    )
    if uploaded_files:
        try:
            engine = get_engine()
        except FileNotFoundError as exc:
            st.error(f"Cannot run inference because a trained checkpoint is missing: {exc}")
            return

        cached_results = st.session_state.setdefault("inference_results", {})
        for uploaded_file in uploaded_files:
            image_bytes = uploaded_file.getvalue()
            image_digest = hashlib.sha256(image_bytes).hexdigest()[:16]
            image_rgb = preview_uploaded_image(image_bytes)
            st.markdown("<div class='case-heading'></div>", unsafe_allow_html=True)
            st.subheader(uploaded_file.name)
            if image_rgb is None:
                st.error("This file could not be decoded as an image. Other selected files will still be processed.")
                continue

            cache_key = f"bytes-input-v1:{image_digest}"
            if cache_key in cached_results:
                result = cached_results[cache_key]
            else:
                result = engine.predict_image(image_bytes, image_id=image_digest)
                save_inference_artifacts(result, image_digest)
                cached_results[cache_key] = result

            if not result.is_valid:
                st.error(result.error_message or "The file is not a valid fundus image.")
                continue

            summary_tab, image_tab, arps_tab = st.tabs(["Prediction", "Image review", "Processing details"])
            with summary_tab:
                top_metrics = st.columns(4)
                top_metrics[0].metric("Screening", result.binary_prediction_str)
                top_metrics[1].metric("Severity estimate", result.five_class_prediction_str)
                top_metrics[2].metric("Severity confidence", f"{result.confidence * 100:.1f}%")
                top_metrics[3].metric("Selected path", result.selected_path, f"ARPS {result.arps_score:.4f}")

                consistency_status = result.consistency.get("consistency_status", "UNKNOWN")
                if consistency_status == "MATCH":
                    st.success(f"Model outputs are internally consistent · {consistency_status}")
                else:
                    st.warning(f"Model outputs differ · {consistency_status}. Review both outputs.")

                probability_col, severity_col = st.columns(2)
                with probability_col:
                    st.markdown("#### Binary screening probabilities")
                    render_probability("Normal", result.binary_probabilities.get("Normal", 0.0))
                    render_probability("DR", result.binary_probabilities.get("DR", 0.0))
                with severity_col:
                    st.markdown("#### Five-class probabilities")
                    for class_name, probability in result.five_class_probabilities.items():
                        render_probability(class_name, probability)

                st.caption("Probabilities are model confidence scores, not calibrated clinical risk estimates.")
                payload = build_result_payload(result, image_digest)
                st.download_button(
                    "Download result JSON",
                    data=json.dumps(payload, indent=2, ensure_ascii=False),
                    file_name=f"retinal_review_{image_digest}.json",
                    mime="application/json",
                    icon=":material/download:",
                    key=f"download_{image_digest}",
                )

            with image_tab:
                original_col, processed_col = st.columns(2)
                with original_col:
                    st.image(image_rgb, caption="Original fundus image", width="stretch")
                with processed_col:
                    if result.processed_image_rgb is not None:
                        st.image(result.processed_image_rgb, caption=f"Processed image · {result.selected_path}", width="stretch")
                cam_binary_col, cam_five_col = st.columns(2)
                with cam_binary_col:
                    st.markdown("#### Binary-model Grad-CAM")
                    if result.gradcam_overlay_rgb is not None:
                        st.image(result.gradcam_overlay_rgb, caption=f"Target class: {result.binary_prediction_str}", width="stretch")
                    else:
                        st.info("Binary Grad-CAM could not be generated for this image.")
                with cam_five_col:
                    st.markdown("#### Five-class-model Grad-CAM")
                    if result.five_class_gradcam_overlay_rgb is not None:
                        st.image(result.five_class_gradcam_overlay_rgb, caption=f"Target class: {result.five_class_prediction_str}", width="stretch")
                    else:
                        st.info("Five-class Grad-CAM could not be generated for this image.")
                st.caption("Grad-CAM highlights image regions that influenced each model output; it is not lesion segmentation or proof of pathology.")

            with arps_tab:
                st.markdown("#### ARPS candidate comparison")
                if result.arps_details is not None:
                    candidate_rows = [
                        {
                            "Path": name,
                            "Score": detail.arps_score,
                            "Constraints": "Pass" if detail.satisfies_constraints else "Fail",
                            "Vessel similarity": detail.vessel_similarity,
                            "Noise change": detail.delta_noise,
                            "Contrast change": detail.delta_contrast,
                        }
                        for name, detail in result.arps_details.candidates.items()
                    ]
                    st.dataframe(candidate_rows, hide_index=True, width="stretch")
                    st.caption(result.arps_details.selection_reason)
                if result.quality_metrics:
                    st.markdown("#### Input quality measurements")
                    quality_columns = st.columns(len(result.quality_metrics))
                    for column, (metric_name, metric_value) in zip(quality_columns, result.quality_metrics.items()):
                        column.metric(metric_name.replace("_", " ").title(), f"{metric_value:.2f}")
    else:
        st.info("Select a retinal fundus image above to begin review.")


if __name__ == "__main__":
    main()
