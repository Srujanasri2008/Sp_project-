"""Run trained binary and five-class models on one fundus image."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.inference.engine import DRInferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a JPG, JPEG, or PNG fundus image")
    parser.add_argument("--device", default="cpu", choices=("cpu",), help="Inference device (CPU only in this project setup)")
    parser.add_argument(
        "--binary-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "binary" / "best_model.pt",
    )
    parser.add_argument(
        "--five-class-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "five_class" / "best_model.pt",
    )
    return parser.parse_args()


def save_visualization(result, output_path: Path):
    panels = [
        ("Original image", result.original_image_rgb),
        (f"Selected path: {result.selected_path}", result.processed_image_rgb),
        (f"Binary-model Grad-CAM: {result.binary_prediction_str}", result.gradcam_overlay_rgb),
        (f"Five-class Grad-CAM: {result.five_class_prediction_str}", result.five_class_gradcam_overlay_rgb),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.8), constrained_layout=True)
    for axis, (title, image) in zip(axes, panels):
        axis.axis("off")
        axis.set_title(title)
        if image is not None:
            axis.imshow(image)
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def result_payload(result, source_path: Path):
    candidates = {}
    if result.arps_details is not None:
        candidates = {
            name: asdict(candidate)
            for name, candidate in result.arps_details.candidates.items()
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_image": str(source_path),
        "is_valid": result.is_valid,
        "error_message": result.error_message,
        "selected_processing_path": result.selected_path,
        "arps_score": result.arps_score,
        "arps_selection_reason": result.arps_details.selection_reason if result.arps_details else None,
        "arps_fallback_used": result.arps_details.fallback_used if result.arps_details else None,
        "candidate_scores": candidates,
        "original_quality_metrics": result.quality_metrics,
        "binary_prediction": result.binary_prediction_str,
        "binary_probabilities_percent": result.binary_probabilities,
        "five_class_prediction": result.five_class_prediction_str,
        "five_class_probabilities_percent": result.five_class_probabilities,
        "five_class_confidence": result.confidence,
        "consistency": result.consistency,
        "binary_gradcam_generated": result.gradcam_overlay_rgb is not None,
        "five_class_gradcam_generated": result.five_class_gradcam_overlay_rgb is not None,
    }


def main():
    args = parse_args()
    source_path = Path(args.image).expanduser().resolve()
    if not source_path.is_file():
        print(f"ERROR: image file does not exist: {source_path}", file=sys.stderr)
        return 2

    try:
        engine = DRInferenceEngine(
            binary_model_path=args.binary_checkpoint,
            five_class_model_path=args.five_class_checkpoint,
            device=args.device,
        )
        result = engine.predict_image(source_path, image_id=source_path.name)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: inference could not start: {exc}", file=sys.stderr)
        return 2

    image_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:10]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_") or "image"
    output_dir = PROJECT_ROOT / "outputs" / "inference" / f"{safe_stem}_{image_digest}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = result_payload(result, source_path)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Original image: {source_path}")
    print(f"Valid fundus image: {result.is_valid}")
    if not result.is_valid:
        print(f"Validation error: {result.error_message}")
        print(f"Full result: {result_path.relative_to(PROJECT_ROOT)}")
        return 1

    visualization_path = output_dir / "inference_visualization.png"
    save_visualization(result, visualization_path)
    print(f"Selected processing path: {result.selected_path}")
    print(f"ARPS score: {result.arps_score:.4f}")
    print("Candidate scores:")
    for name, candidate in candidates_for_print(result).items():
        print(f"  {name}: {candidate['arps_score']:.4f} (constraints_pass={candidate['satisfies_constraints']})")
    print(f"Binary prediction: {result.binary_prediction_str}")
    print(f"Binary probabilities (%): {result.binary_probabilities}")
    print(f"Five-class prediction: {result.five_class_prediction_str}")
    print(f"Five-class probabilities (%): {result.five_class_probabilities}")
    print(f"Consistency: {result.consistency}")
    print(f"Binary Grad-CAM generated: {result.gradcam_overlay_rgb is not None}")
    print(f"Five-class Grad-CAM generated: {result.five_class_gradcam_overlay_rgb is not None}")
    print(f"Visualization: {visualization_path.relative_to(PROJECT_ROOT)}")
    print(f"Full result: {result_path.relative_to(PROJECT_ROOT)}")
    return 0


def candidates_for_print(result):
    if result.arps_details is None:
        return {}
    return {name: asdict(candidate) for name, candidate in result.arps_details.candidates.items()}


if __name__ == "__main__":
    raise SystemExit(main())