"""
run_error_analysis.py
Performs comprehensive error taxonomy and comparative error analysis across:
- Baseline (Fixed Preprocessing)
- AGCWD (Literature Comparator)
- Proposed ARPS System
Analyzes:
- False Positives (FP) & False Negatives (FN)
- Inter-system Error Venn / Overlap:
  * Cases where all 3 systems fail (Universal Hard Cases)
  * Cases where Proposed succeeds while comparators fail (Proposed Advantage)
  * System-specific unique errors
- Correlates errors with image signal quality
Saves results to reports/error_analysis/error_summary.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def run_error_analysis():
    print("=" * 65)
    print("PHASE 13: COMPARATIVE ERROR ANALYSIS & SYSTEM TAXONOMY")
    print("=" * 65)

    base_path = PROJECT_ROOT / "reports" / "baseline_fixed_test_predictions.csv"
    agcwd_path = PROJECT_ROOT / "reports" / "agcwd_adaptive_test_predictions.csv"
    prop_path = PROJECT_ROOT / "reports" / "binary_test_predictions.csv"
    cache_path = PROJECT_ROOT / "data" / "metadata" / "arps_selection_cache.csv"

    if not (base_path.exists() and agcwd_path.exists() and prop_path.exists()):
        print("[ERROR] Predictions for all 3 systems must exist to perform comparative error analysis.")
        return

    df_base = pd.read_csv(base_path).rename(columns={"y_pred": "pred_base", "prob_dr": "prob_base"})
    df_agcwd = pd.read_csv(agcwd_path).rename(columns={"y_pred": "pred_agcwd", "prob_dr": "prob_agcwd"})
    df_prop = pd.read_csv(prop_path).rename(columns={"y_pred": "pred_prop", "prob_dr": "prob_prop"})
    cache_df = pd.read_csv(cache_path)

    merged = pd.merge(df_base, df_agcwd[["image_id", "pred_agcwd", "prob_agcwd"]], on="image_id")
    merged = pd.merge(merged, df_prop[["image_id", "pred_prop", "prob_prop"]], on="image_id")
    merged = pd.merge(merged, cache_df, on="image_id", how="left")

    y_true = merged["y_true"]
    merged["base_correct"] = (merged["pred_base"] == y_true)
    merged["agcwd_correct"] = (merged["pred_agcwd"] == y_true)
    merged["prop_correct"] = (merged["pred_prop"] == y_true)

    # Error taxonomy subsets
    all_fail = merged[(~merged["base_correct"]) & (~merged["agcwd_correct"]) & (~merged["prop_correct"])]
    all_pass = merged[merged["base_correct"] & merged["agcwd_correct"] & merged["prop_correct"]]
    prop_only_correct = merged[(~merged["base_correct"]) & (~merged["agcwd_correct"]) & merged["prop_correct"]]
    prop_only_fail = merged[merged["base_correct"] & merged["agcwd_correct"] & (~merged["prop_correct"])]
    base_only_fail = merged[(~merged["base_correct"]) & merged["agcwd_correct"] & merged["prop_correct"]]
    agcwd_only_fail = merged[merged["base_correct"] & (~merged["agcwd_correct"]) & merged["prop_correct"]]

    print("\nSystem Error Categorization (Test Set = 550 images):")
    print("-" * 65)
    print(f"  All 3 systems correct                     : {len(all_pass):>4} ({len(all_pass)/len(merged)*100:.1f}%)")
    print(f"  All 3 systems fail (Universal Hard Cases) : {len(all_fail):>4} ({len(all_fail)/len(merged)*100:.1f}%)")
    print(f"  Proposed succeeds where both others fail  : {len(prop_only_correct):>4} ({len(prop_only_correct)/len(merged)*100:.1f}%)")
    print(f"  Proposed unique errors (both others pass) : {len(prop_only_fail):>4} ({len(prop_only_fail)/len(merged)*100:.1f}%)")
    print(f"  Baseline unique errors (both others pass) : {len(base_only_fail):>4} ({len(base_only_fail)/len(merged)*100:.1f}%)")
    print(f"  AGCWD unique errors (both others pass)    : {len(agcwd_only_fail):>4} ({len(agcwd_only_fail)/len(merged)*100:.1f}%)")
    print("-" * 65)

    # Proposed model detailed FP and FN breakdown
    prop_fp = merged[(merged["pred_prop"] == 1) & (y_true == 0)]
    prop_fn = merged[(merged["pred_prop"] == 0) & (y_true == 1)]

    print("\nProposed System Error Decomposition:")
    print(f"  False Positives (FP) [Predicted DR, Actually Normal] : {len(prop_fp)}")
    print(f"  False Negatives (FN) [Predicted Normal, Actually DR] : {len(prop_fn)}")

    if len(prop_fp) > 0:
        print(f"    FP Quality Profile -> Avg Contrast: {prop_fp['contrast'].mean():.2f}, "
              f"Avg Noise: {prop_fp['noise'].mean():.2f}, Avg IllumVar: {prop_fp['illumination_variation'].mean():.2f}")
    if len(prop_fn) > 0:
        print(f"    FN Quality Profile -> Avg Contrast: {prop_fn['contrast'].mean():.2f}, "
              f"Avg Noise: {prop_fn['noise'].mean():.2f}, Avg IllumVar: {prop_fn['illumination_variation'].mean():.2f}")

    # Save summary table
    out_dir = PROJECT_ROOT / "reports" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = [
        {"category": "All 3 Systems Correct", "count": len(all_pass), "percentage": round(len(all_pass)/len(merged)*100, 2)},
        {"category": "All 3 Systems Fail (Universal Hard)", "count": len(all_fail), "percentage": round(len(all_fail)/len(merged)*100, 2)},
        {"category": "Proposed Advantage (Proposed Right, Comparators Wrong)", "count": len(prop_only_correct), "percentage": round(len(prop_only_correct)/len(merged)*100, 2)},
        {"category": "Proposed Unique Error (Proposed Wrong, Comparators Right)", "count": len(prop_only_fail), "percentage": round(len(prop_only_fail)/len(merged)*100, 2)},
        {"category": "Proposed False Positives", "count": len(prop_fp), "percentage": round(len(prop_fp)/len(merged)*100, 2)},
        {"category": "Proposed False Negatives", "count": len(prop_fn), "percentage": round(len(prop_fn)/len(merged)*100, 2)}
    ]
    pd.DataFrame(summary_data).to_csv(out_dir / "error_summary.csv", index=False)
    
    # Save full image-level error audit manifest
    merged.to_csv(out_dir / "full_error_audit_manifest.csv", index=False)
    print(f"\nArtifacts saved to:\n  - {out_dir / 'error_summary.csv'}\n  - {out_dir / 'full_error_audit_manifest.csv'}")
    print("=" * 65)


if __name__ == "__main__":
    run_error_analysis()
