"""
cache_arps_selection.py
Performs offline ARPS path selection and quality metric analysis for all dataset images.
Stores the deterministic routing decision (PATH_A, PATH_B, or PATH_C), ARPS score,
and baseline quality measurements in data/metadata/arps_selection_cache.csv.
Eliminates redundant candidate DSP evaluations during iterative neural network training.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
from tqdm import tqdm
from src.processing.arps import select_processing_path

MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "official_split_manifest.csv"
OUTPUT_CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "arps_selection_cache.csv"


def build_arps_cache():
    print("=" * 65)
    print("PRECOMPUTING DETERMINISTIC ARPS SELECTION CACHE")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} images.")

    records = []
    path_distribution = {"PATH_A": 0, "PATH_B": 0, "PATH_C": 0}
    fallback_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing ARPS decisions"):
        img_id = row["image_id"]
        rel_path = row["file_path"]
        abs_path = PROJECT_ROOT / rel_path

        img_bgr = cv2.imread(str(abs_path))
        if img_bgr is None:
            raise IOError(f"Could not read image at {abs_path}")

        arps_res = select_processing_path(img_bgr)
        sel_path = arps_res.selected_path
        path_distribution[sel_path] = path_distribution.get(sel_path, 0) + 1
        if arps_res.fallback_used:
            fallback_count += 1

        eval_sel = arps_res.candidates[sel_path]

        records.append({
            "image_id": img_id,
            "selected_path": sel_path,
            "arps_score": round(arps_res.arps_score, 4),
            "fallback_used": arps_res.fallback_used,
            "contrast": eval_sel.contrast,
            "sharpness": eval_sel.sharpness,
            "noise": eval_sel.noise,
            "illumination_variation": eval_sel.illumination_variation,
            "vessel_similarity": eval_sel.vessel_similarity,
            "delta_noise": eval_sel.delta_noise,
            "delta_contrast": eval_sel.delta_contrast,
            "selection_reason": arps_res.selection_reason
        })

    cache_df = pd.DataFrame(records)
    OUTPUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(OUTPUT_CACHE_PATH, index=False)
    print(f"\n[+] ARPS selection cache saved to: {OUTPUT_CACHE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    File size: {OUTPUT_CACHE_PATH.stat().st_size / 1024:.2f} KB")

    print("\nARPS Path Selection Summary across all 3,662 images:")
    for p, cnt in path_distribution.items():
        pct = (cnt / len(df)) * 100.0
        print(f"  {p:<10}: {cnt:>5} images ({pct:>5.1f}%)")
    print(f"  Fallbacks used: {fallback_count} images ({(fallback_count/len(df))*100.1:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    build_arps_cache()
