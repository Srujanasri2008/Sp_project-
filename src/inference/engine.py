"""
engine.py
Unified, production-grade inference engine for Diabetic Retinopathy detection.
Integrates the complete end-to-end pipeline:
1. Image validation (format, dimensions, channels, finiteness, retinal FOV)
2. Signal/Quality analysis (contrast, sharpness, noise, illumination, vessels)
3. ARPS candidate evaluation and image-specific path selection (A, B, C)
4. Multi-domain feature representations (Spatial, Vessel structural, Frequency spectral)
5. Binary DR screening prediction
6. Five-class ordinal DR severity prediction
7. Semantic consistency check (MATCH / WARNING)
8. Grad-CAM visual explanation overlay
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.validation import validate_fundus_image, ImageValidationResult
from src.processing.quality import extract_retinal_mask, analyze_image_quality
from src.processing.arps import select_processing_path, ARPSResult
from src.processing.vessel import extract_vessel_representation
from src.processing.frequency import extract_frequency_representation
from src.evaluation.metrics import consistency_check
from src.models.network import DRMultiDomainModel
from src.explainability.gradcam import GradCAM


SEVERITY_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (PDR)"
}


@dataclass
class SingleInferenceResult:
    image_id: str
    is_valid: bool
    error_message: Optional[str]
    original_image_rgb: Optional[np.ndarray]
    processed_image_rgb: Optional[np.ndarray]
    selected_path: str
    arps_score: float
    arps_details: Optional[ARPSResult]
    quality_metrics: Optional[Dict[str, float]]
    binary_label: int
    binary_prediction_str: str
    binary_probabilities: Dict[str, float]
    five_class_label: int
    five_class_prediction_str: str
    five_class_probabilities: Dict[str, float]
    confidence: float
    consistency: Dict[str, Any]
    gradcam_heatmap: Optional[np.ndarray]
    gradcam_overlay_rgb: Optional[np.ndarray]
    five_class_gradcam_heatmap: Optional[np.ndarray]
    five_class_gradcam_overlay_rgb: Optional[np.ndarray]


class DRInferenceEngine:
    """
    Main inference controller for single-image and batch processing.
    """
    def __init__(
        self,
        binary_model_path: Optional[Union[str, Path]] = None,
        five_class_model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.binary_model = None
        self.five_class_model = None
        self.gradcam = None
        self.five_class_gradcam = None

        bin_path = Path(binary_model_path) if binary_model_path else (PROJECT_ROOT / "models" / "binary" / "best_model.pt")
        five_path = Path(five_class_model_path) if five_class_model_path else (PROJECT_ROOT / "models" / "five_class" / "best_model.pt")

        self.load_models(bin_path, five_path)

    def load_models(self, binary_path: Path, five_class_path: Path):
        """Loads weights for binary and five-class models."""
        missing_paths = [path for path in (binary_path, five_class_path) if not path.is_file()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                "Required trained checkpoint(s) are missing; refusing random-weight inference: "
                f"{missing}"
            )

        # 1. Binary Model
        self.binary_model = DRMultiDomainModel(
            num_classes=2,
            include_vessel=True,
            include_frequency=True
        ).to(self.device)

        ckpt = torch.load(binary_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.binary_model.load_state_dict(state_dict)
        self.binary_model.eval()
        print(f"[+] Loaded binary model from: {binary_path}")

        # 2. Five-Class Model
        self.five_class_model = DRMultiDomainModel(
            num_classes=5,
            include_vessel=True,
            include_frequency=True
        ).to(self.device)

        ckpt = torch.load(five_class_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.five_class_model.load_state_dict(state_dict)
        self.five_class_model.eval()
        print(f"[+] Loaded five-class model from: {five_class_path}")

        # Initialize Grad-CAM on spatial branch of binary model
        self.gradcam = GradCAM(self.binary_model)
        self.five_class_gradcam = GradCAM(self.five_class_model)

    def predict_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        image_id: str = "query_image",
        generate_cam: bool = True
    ) -> SingleInferenceResult:
        """
        Executes complete frozen pipeline on a single retinal fundus image.
        """
        # Step 1: Validation
        val_res = validate_fundus_image(image_input)
        if not val_res.is_valid:
            return SingleInferenceResult(
                image_id=image_id,
                is_valid=False,
                error_message=val_res.error_message,
                original_image_rgb=None,
                processed_image_rgb=None,
                selected_path="INVALID",
                arps_score=0.0,
                arps_details=None,
                quality_metrics=None,
                binary_label=-1,
                binary_prediction_str="Invalid Image",
                binary_probabilities={},
                five_class_label=-1,
                five_class_prediction_str="Invalid Image",
                five_class_probabilities={},
                confidence=0.0,
                consistency={"consistency_status": "INVALID", "clinical_interpretation": val_res.error_message},
                gradcam_heatmap=None,
                gradcam_overlay_rgb=None,
                five_class_gradcam_heatmap=None,
                five_class_gradcam_overlay_rgb=None,
            )

        raw_bgr = val_res.image_bgr
        raw_bgr_resized = cv2.resize(raw_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        original_rgb = cv2.cvtColor(raw_bgr_resized, cv2.COLOR_BGR2RGB)

        # Step 2 & 3: Signal Quality & ARPS Selection
        arps_res = select_processing_path(raw_bgr_resized)
        proc_bgr = arps_res.selected_image
        proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

        # Step 4: Multi-Domain Representations
        spatial_tensor = torch.from_numpy(proc_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        vessel_map = extract_vessel_representation(proc_bgr, target_size=(224, 224))
        vessel_tensor = torch.from_numpy(vessel_map).permute(2, 0, 1).unsqueeze(0).to(self.device)

        freq_map = extract_frequency_representation(proc_bgr, target_size=(224, 224))
        freq_tensor = torch.from_numpy(freq_map).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Step 5: Binary Model Inference
        with torch.no_grad():
            bin_logits = self.binary_model(spatial_tensor, vessel_tensor, freq_tensor)
            bin_probs = torch.softmax(bin_logits, dim=1).cpu().numpy()[0]
            bin_pred = int(np.argmax(bin_probs))

        bin_prob_dict = {
            "Normal": round(float(bin_probs[0]) * 100.0, 2),
            "DR": round(float(bin_probs[1]) * 100.0, 2)
        }
        bin_str = "DR" if bin_pred == 1 else "Normal"

        # Step 6: Five-Class Model Inference
        with torch.no_grad():
            five_logits = self.five_class_model(spatial_tensor, vessel_tensor, freq_tensor)
            five_probs = torch.softmax(five_logits, dim=1).cpu().numpy()[0]
            five_pred = int(np.argmax(five_probs))

        five_prob_dict = {
            SEVERITY_NAMES[i]: round(float(five_probs[i]) * 100.0, 2) for i in range(5)
        }
        five_str = SEVERITY_NAMES[five_pred]

        # Overall confidence: probability of the predicted severity
        confidence = float(five_probs[five_pred])

        # Step 7: Semantic Consistency Check
        consistency_info = consistency_check(
            binary_pred=bin_pred,
            five_class_pred=five_pred,
            binary_prob=float(bin_probs[bin_pred]),
            five_class_prob=five_probs.tolist()
        )

        # Step 8: Grad-CAM Explainability
        heatmap = None
        overlay_rgb = None
        five_class_heatmap = None
        five_class_overlay_rgb = None
        if generate_cam and self.gradcam is not None:
            try:
                cam_target = bin_pred
                heatmap, _, _ = self.gradcam.generate(
                    spatial_tensor,
                    vessel_tensor,
                    freq_tensor,
                    target_class=cam_target
                )
                overlay_rgb = self.gradcam.generate_overlay(original_rgb, heatmap, alpha=0.45)
            except Exception as e:
                print(f"[!] Binary Grad-CAM generation warning: {e}")

        if generate_cam and self.five_class_gradcam is not None:
            try:
                five_class_heatmap, _, _ = self.five_class_gradcam.generate(
                    spatial_tensor,
                    vessel_tensor,
                    freq_tensor,
                    target_class=five_pred,
                )
                five_class_overlay_rgb = self.five_class_gradcam.generate_overlay(
                    original_rgb, five_class_heatmap, alpha=0.45
                )
            except Exception as e:
                print(f"[!] Five-class Grad-CAM generation warning: {e}")

        return SingleInferenceResult(
            image_id=image_id,
            is_valid=True,
            error_message=None,
            original_image_rgb=original_rgb,
            processed_image_rgb=proc_rgb,
            selected_path=arps_res.selected_path,
            arps_score=round(arps_res.arps_score, 4),
            arps_details=arps_res,
            quality_metrics=arps_res.original_quality,
            binary_label=bin_pred,
            binary_prediction_str=bin_str,
            binary_probabilities=bin_prob_dict,
            five_class_label=five_pred,
            five_class_prediction_str=five_str,
            five_class_probabilities=five_prob_dict,
            confidence=round(confidence, 4),
            consistency=consistency_info,
            gradcam_heatmap=heatmap,
            gradcam_overlay_rgb=overlay_rgb,
            five_class_gradcam_heatmap=five_class_heatmap,
            five_class_gradcam_overlay_rgb=five_class_overlay_rgb,
        )

    def predict_batch(self, image_inputs: List[Union[str, Path, bytes, np.ndarray]]) -> List[SingleInferenceResult]:
        """
        Executes the identical inference pipeline sequentially over a batch of images.
        """
        results = []
        for idx, inp in enumerate(image_inputs):
            img_id = f"batch_item_{idx+1:03d}"
            res = self.predict_image(inp, image_id=img_id, generate_cam=True)
            results.append(res)
        return results
