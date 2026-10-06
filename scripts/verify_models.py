"""
verify_models.py
Verifies Phase 5: Deep learning architecture instantiation and forward pass across all configurations:
- Binary screening (num_classes=2)
- Five-class grading (num_classes=5)
- Spatial only ablation
- Spatial + Vessel ablation
- Spatial + Frequency ablation
- Full proposed multi-domain fusion
- Grad-CAM target layer accessibility
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.network import DRMultiDomainModel


def run_model_verification():
    print("=" * 65)
    print("PHASE 5: DEEP LEARNING MODEL & FORWARD PASS VERIFICATION")
    print("=" * 65)

    batch_size = 2
    x_spatial = torch.randn(batch_size, 3, 224, 224)
    x_vessel = torch.randn(batch_size, 1, 224, 224)
    x_freq = torch.randn(batch_size, 1, 224, 224)

    # 1. Full Proposed Binary Model
    print("\n[1] Testing Full Proposed Binary Model (Spatial + Vessel + Frequency)...")
    model_bin = DRMultiDomainModel(num_classes=2, include_vessel=True, include_frequency=True)
    out_bin = model_bin(x_spatial, x_vessel, x_freq)
    print(f"  Output shape: {out_bin.shape} (Expected: ({batch_size}, 2))")
    assert out_bin.shape == (batch_size, 2), "Binary output shape mismatch"

    # 2. Full Proposed Five-Class Model
    print("\n[2] Testing Full Proposed Five-Class Model...")
    model_five = DRMultiDomainModel(num_classes=5, include_vessel=True, include_frequency=True)
    out_five = model_five(x_spatial, x_vessel, x_freq)
    print(f"  Output shape: {out_five.shape} (Expected: ({batch_size}, 5))")
    assert out_five.shape == (batch_size, 5), "Five-class output shape mismatch"

    # 3. Spatial Only (Ablation / Baseline / AGCWD)
    print("\n[3] Testing Spatial-Only Model (Ablation / Baseline / AGCWD)...")
    model_spatial = DRMultiDomainModel(num_classes=2, include_vessel=False, include_frequency=False)
    out_spatial = model_spatial(x_spatial)
    print(f"  Output shape: {out_spatial.shape} (Expected: ({batch_size}, 2))")
    assert out_spatial.shape == (batch_size, 2), "Spatial-only output shape mismatch"

    # 4. Spatial + Vessel Ablation
    print("\n[4] Testing Spatial + Vessel Model...")
    model_sv = DRMultiDomainModel(num_classes=2, include_vessel=True, include_frequency=False)
    out_sv = model_sv(x_spatial, x_vessel=x_vessel)
    print(f"  Output shape: {out_sv.shape} (Expected: ({batch_size}, 2))")
    assert out_sv.shape == (batch_size, 2), "Spatial+Vessel output shape mismatch"

    # 5. Spatial + Frequency Ablation
    print("\n[5] Testing Spatial + Frequency Model...")
    model_sf = DRMultiDomainModel(num_classes=2, include_vessel=False, include_frequency=True)
    out_sf = model_sf(x_spatial, x_freq=x_freq)
    print(f"  Output shape: {out_sf.shape} (Expected: ({batch_size}, 2))")
    assert out_sf.shape == (batch_size, 2), "Spatial+Freq output shape mismatch"

    # 6. Grad-CAM Target Layer
    print("\n[6] Checking Grad-CAM Target Layer...")
    target_layer = model_bin.get_gradcam_target_layer()
    print(f"  Target Layer: {target_layer}")
    assert target_layer is not None, "Target layer for Grad-CAM is None"

    print("\n" + "=" * 65)
    print("ALL MODEL ARCHITECTURES AND FORWARD PASSES VERIFIED!")
    print("=" * 65)


if __name__ == "__main__":
    run_model_verification()
