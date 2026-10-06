"""
verify_processing.py
Verifies Phases 3 & 4:
- Quality analysis & retinal masking
- Candidate DSP paths (A, B, C) & AGCWD
- ARPS scoring & constraint-guided path selection
- Vessel structural representation
- Frequency FFT log-magnitude representation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.processing.validation import validate_fundus_image
from src.processing.quality import extract_retinal_mask, analyze_image_quality
from src.processing.candidates import apply_path_a, apply_path_b, apply_path_c, apply_agcwd
from src.processing.arps import select_processing_path
from src.processing.vessel import extract_vessel_representation
from src.processing.frequency import extract_frequency_representation

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_verification():
    print("=" * 65)
    print("PHASES 3 & 4: IMAGE PROCESSING & ARPS PIPELINE VERIFICATION")
    print("=" * 65)

    manifest_path = PROJECT_ROOT / "data" / "splits" / "official_split_manifest.csv"
    df = pd.read_csv(manifest_path)
    sample_path = PROJECT_ROOT / df.iloc[0]["file_path"]
    print(f"Sample test image: {sample_path.name}")

    # 1. Validation test
    print("\n[1] Testing image validation...")
    val_res = validate_fundus_image(sample_path)
    assert val_res.is_valid, f"Validation failed: {val_res.error_message}"
    print(f"  Validation: PASSED ({val_res.metadata})")

    # Invalid image test
    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
    invalid_res = validate_fundus_image(fake_img)
    assert not invalid_res.is_valid, "Failed to reject invalid small image!"
    print(f"  Invalid rejection test: PASSED (Correctly rejected: {invalid_res.error_message})")

    img_bgr = cv2.imread(str(sample_path))

    # 2. Retinal mask & quality analysis
    print("\n[2] Testing retinal mask & signal quality...")
    mask = extract_retinal_mask(img_bgr)
    assert mask.shape == (224, 224), f"Mask shape mismatch: {mask.shape}"
    quality = analyze_image_quality(img_bgr)
    print(f"  Quality metrics: {quality}")
    assert quality["contrast"] > 0, "Contrast should be positive"
    assert quality["sharpness"] > 0, "Sharpness should be positive"

    # 3. Candidate DSP paths & AGCWD
    print("\n[3] Testing candidate paths (A, B, C) and AGCWD...")
    out_a = apply_path_a(img_bgr)
    out_b = apply_path_b(img_bgr)
    out_c = apply_path_c(img_bgr)
    out_agcwd = apply_agcwd(img_bgr)
    assert out_a.shape == (224, 224, 3), "Path A shape mismatch"
    assert out_b.shape == (224, 224, 3), "Path B shape mismatch"
    assert out_c.shape == (224, 224, 3), "Path C shape mismatch"
    assert out_agcwd.shape == (224, 224, 3), "AGCWD shape mismatch"
    print("  All candidate paths & AGCWD generated successfully.")

    # 4. ARPS selection
    print("\n[4] Testing ARPS candidate evaluation and selection...")
    arps_res = select_processing_path(img_bgr)
    print(f"  Selected Path   : {arps_res.selected_path}")
    print(f"  ARPS Score      : {arps_res.arps_score:.4f}")
    print(f"  Selection Reason: {arps_res.selection_reason}")
    print(f"  Fallback Used   : {arps_res.fallback_used}")
    for name, cand in arps_res.candidates.items():
        print(f"    - {name}: ARPS={cand.arps_score:.4f}, VesselSim={cand.vessel_similarity:.4f}, "
              f"dNoise={cand.delta_noise:.4f}, dContrast={cand.delta_contrast:.4f}, Valid={cand.satisfies_constraints}")
    assert arps_res.selected_path in ["PATH_A", "PATH_B", "PATH_C"], "Invalid path selected"

    # 5. Vessel branch
    print("\n[5] Testing retinal vessel representation...")
    vessel_map = extract_vessel_representation(img_bgr)
    assert vessel_map.shape == (224, 224, 1), f"Vessel map shape mismatch: {vessel_map.shape}"
    assert np.all(vessel_map >= 0.0) and np.all(vessel_map <= 1.0), "Vessel map out of range [0, 1]"
    print(f"  Vessel branch: PASSED (Shape: {vessel_map.shape}, Max: {np.max(vessel_map):.3f})")

    # 6. Frequency branch
    print("\n[6] Testing 2D FFT frequency representation...")
    freq_map = extract_frequency_representation(img_bgr)
    assert freq_map.shape == (224, 224, 1), f"Frequency map shape mismatch: {freq_map.shape}"
    assert np.all(np.isfinite(freq_map)), "Frequency map has non-finite values!"
    assert np.all(freq_map >= 0.0) and np.all(freq_map <= 1.0), "Frequency map out of range [0, 1]"
    print(f"  Frequency branch: PASSED (Shape: {freq_map.shape}, Max: {np.max(freq_map):.3f})")

    print("\n" + "=" * 65)
    print("ALL IMAGE-PROCESSING AND ARPS MODULES FULLY VERIFIED!")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
