"""
arps.py
Adaptive Retinal Signal-Preservation Score (ARPS) module.
Calculates candidate signal quality, checks frozen constraints, and selects the
image-specific optimal processing path.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import yaml
from pathlib import Path

from src.processing.quality import (
    extract_retinal_mask,
    compute_contrast,
    compute_sharpness,
    compute_noise_indicator,
    compute_illumination_variation,
    compute_vessel_visibility
)
from src.processing.candidates import (
    apply_path_a,
    apply_path_b,
    apply_path_c,
    generate_candidate_paths
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "arps_config.yaml"


@dataclass
class CandidateEvaluation:
    path_name: str
    arps_score: float
    vessel_similarity: float
    delta_noise: float
    delta_contrast: float
    contrast: float
    sharpness: float
    noise: float
    illumination_variation: float
    satisfies_constraints: bool
    constraint_violations: List[str]


@dataclass
class ARPSResult:
    selected_path: str
    selected_image: np.ndarray
    arps_score: float
    candidates: Dict[str, CandidateEvaluation]
    selection_reason: str
    fallback_used: bool
    original_quality: Dict[str, float]


def load_arps_config() -> Dict[str, Any]:
    """Loads frozen ARPS weights and constraints from config/arps_config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Default frozen constants if config file is unavailable
    return {
        "arps": {
            "weights": {
                "contrast": 0.08230652499670243,
                "vessel": 0.0689232338703772,
                "detail": 0.42840850268781994,
                "illumination": 0.28237726704874355,
                "noise": 0.13798447139635686
            },
            "constraints": {
                "min_vessel_similarity": 0.85,
                "max_delta_noise": 2.5,
                "min_delta_contrast": -5.0
            },
            "default_fallback_path": "PATH_A"
        }
    }


def compute_vessel_structural_similarity(vessel_orig: np.ndarray, vessel_cand: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes normalized cosine similarity between candidate and original vessel maps inside mask.
    Produces a value in [0.0, 1.0].
    """
    v_orig = vessel_orig[mask > 0].astype(np.float64)
    v_cand = vessel_cand[mask > 0].astype(np.float64)
    
    norm_orig = np.linalg.norm(v_orig)
    norm_cand = np.linalg.norm(v_cand)
    
    if norm_orig < 1e-7 or norm_cand < 1e-7:
        return 1.0
        
    similarity = np.dot(v_orig, v_cand) / (norm_orig * norm_cand)
    return float(np.clip(similarity, 0.0, 1.0))


def evaluate_candidate(
    cand_name: str,
    cand_img: np.ndarray,
    orig_gray: np.ndarray,
    orig_vessel: np.ndarray,
    orig_quality: Dict[str, float],
    mask: np.ndarray,
    weights: Dict[str, float],
    constraints: Dict[str, float]
) -> CandidateEvaluation:
    """Evaluates a single candidate path under the frozen ARPS formula and constraints."""
    cand_gray = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY)
    
    # Candidate metrics
    c_cand = compute_contrast(cand_gray, mask)
    s_cand = compute_sharpness(cand_gray, mask)
    n_cand = compute_noise_indicator(cand_gray, mask)
    i_cand = compute_illumination_variation(cand_gray, mask)
    _, v_cand = compute_vessel_visibility(cand_img, mask)
    
    # Baseline comparison metrics
    delta_contrast = c_cand - orig_quality["contrast"]
    delta_noise = n_cand - orig_quality["noise"]
    vessel_sim = compute_vessel_structural_similarity(orig_vessel, v_cand, mask)
    
    # Normalized components for ARPS formula
    # C_norm: contrast relative to baseline
    c_norm = float(np.clip(c_cand / (orig_quality["contrast"] + 1e-5), 0.0, 2.0))
    # V_norm: vessel preservation similarity [0, 1]
    v_norm = vessel_sim
    # D_norm: sharpness / detail relative to baseline
    d_norm = float(np.clip(s_cand / (orig_quality["sharpness"] + 1e-5), 0.0, 3.0))
    # I_norm: illumination uniformity quality factor [0, 1]
    i_norm = float(np.clip(1.0 - (i_cand / 100.0), 0.0, 1.0))
    # N_norm: noise relative to baseline
    n_norm = float(np.clip(n_cand / (orig_quality["noise"] + 1e-5), 0.0, 3.0))
    
    # ARPS = w1(C) + w2(V) + w3(D) + w4(I) - w5(N)
    arps_score = (
        weights["contrast"] * c_norm +
        weights["vessel"] * v_norm +
        weights["detail"] * d_norm +
        weights["illumination"] * i_norm -
        weights["noise"] * n_norm
    )
    
    # Check frozen constraints
    violations = []
    if vessel_sim < constraints["min_vessel_similarity"]:
        violations.append(f"Vessel similarity {vessel_sim:.3f} < {constraints['min_vessel_similarity']}")
    if delta_noise > constraints["max_delta_noise"]:
        violations.append(f"Delta noise {delta_noise:.3f} > {constraints['max_delta_noise']}")
    if delta_contrast < constraints["min_delta_contrast"]:
        violations.append(f"Delta contrast {delta_contrast:.3f} < {constraints['min_delta_contrast']}")
        
    satisfies = len(violations) == 0
    
    return CandidateEvaluation(
        path_name=cand_name,
        arps_score=float(arps_score),
        vessel_similarity=round(vessel_sim, 4),
        delta_noise=round(delta_noise, 4),
        delta_contrast=round(delta_contrast, 4),
        contrast=round(c_cand, 4),
        sharpness=round(s_cand, 4),
        noise=round(n_cand, 4),
        illumination_variation=round(i_cand, 4),
        satisfies_constraints=satisfies,
        constraint_violations=violations
    )


def select_processing_path(image_bgr: np.ndarray) -> ARPSResult:
    """
    Core ARPS Interface:
    Takes an input retinal fundus image, evaluates candidate DSP paths,
    checks retinal preservation constraints, and selects the optimal path.
    
    Args:
        image_bgr: Input BGR image (H, W, 3).
        
    Returns:
        ARPSResult containing selected path name, enhanced image, scores, and rationale.
    """
    cfg = load_arps_config()
    weights = cfg["arps"]["weights"]
    constraints = cfg["arps"]["constraints"]
    default_fallback = cfg["arps"].get("default_fallback_path", "PATH_A")
    
    # 1. Image mask & baseline quality
    mask = extract_retinal_mask(image_bgr)
    orig_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    orig_contrast = compute_contrast(orig_gray, mask)
    orig_sharpness = compute_sharpness(orig_gray, mask)
    orig_noise = compute_noise_indicator(orig_gray, mask)
    orig_illum = compute_illumination_variation(orig_gray, mask)
    orig_vessel_vis, orig_vessel_map = compute_vessel_visibility(image_bgr, mask)
    
    orig_quality = {
        "contrast": orig_contrast,
        "sharpness": orig_sharpness,
        "noise": orig_noise,
        "illumination_variation": orig_illum,
        "vessel_visibility": orig_vessel_vis
    }
    
    # 2. Generate candidates
    candidates = generate_candidate_paths(image_bgr)
    
    # 3. Evaluate each candidate
    evaluations: Dict[str, CandidateEvaluation] = {}
    for name, cand_img in candidates.items():
        evaluations[name] = evaluate_candidate(
            cand_name=name,
            cand_img=cand_img,
            orig_gray=orig_gray,
            orig_vessel=orig_vessel_map,
            orig_quality=orig_quality,
            mask=mask,
            weights=weights,
            constraints=constraints
        )
        
    # 4. Selection logic based on preservation constraints
    valid_candidates = [ev for ev in evaluations.values() if ev.satisfies_constraints]
    
    if valid_candidates:
        # Select valid candidate with maximum ARPS score
        best = max(valid_candidates, key=lambda x: x.arps_score)
        selected_path = best.path_name
        selection_reason = (
            f"Selected {best.path_name} with highest valid ARPS score ({best.arps_score:.4f}) "
            f"satisfying all retinal preservation constraints."
        )
        fallback_used = False
    else:
        # Documented fallback rule: minimal constraint violation penalty
        def constraint_penalty(ev: CandidateEvaluation) -> float:
            p_v = max(0.0, constraints["min_vessel_similarity"] - ev.vessel_similarity) * 10.0
            p_n = max(0.0, ev.delta_noise - constraints["max_delta_noise"])
            p_c = max(0.0, constraints["min_delta_contrast"] - ev.delta_contrast)
            return p_v + p_n + p_c

        best = min(evaluations.values(), key=constraint_penalty)
        if best.path_name in evaluations:
            selected_path = best.path_name
            selection_reason = (
                f"No candidate met all strict constraints. Selected {selected_path} "
                f"under documented fallback policy (least constraint penalty violation)."
            )
            fallback_used = True
        else:
            selected_path = default_fallback
            selection_reason = f"Fallback to default standard path {default_fallback}."
            fallback_used = True
            
    return ARPSResult(
        selected_path=selected_path,
        selected_image=candidates[selected_path],
        arps_score=evaluations[selected_path].arps_score,
        candidates=evaluations,
        selection_reason=selection_reason,
        fallback_used=fallback_used,
        original_quality=orig_quality
    )
