"""
run_robustness.py
Evaluates performance robustness across image quality quartiles (Q1, Q2, Q3, Q4)
for contrast, sharpness, noise, and illumination variation on the test set.
Compares:
- Baseline (Fixed Preprocessing)
- AGCWD (Existing Adaptive Enhancement)
- Proposed ARPS System
Saves results to reports/robustness_results.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_binary_metrics


def make_quality_quartiles(values: pd.Series, labels=None):
    """Create stable quartile labels without crashing when quality scores repeat."""
    if labels is None:
        labels = ["Q1", "Q2", "Q3", "Q4"]

    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return pd.Series(pd.Categorical([], categories=labels), index=values.index)

    if values.nunique() == 1:
        return pd.Series(pd.Categorical([labels[0]] * len(values), categories=labels), index=values.index)

    try:
        quartiles = pd.qcut(values, q=4, labels=labels, duplicates="drop")
        return quartiles
    except ValueError:
        ranked = values.rank(method="first", na_option="keep")
        q_count = len(values)
        labels_by_rank = [labels[0]] * q_count
        for i in range(q_count):
            rank = ranked.iloc[i]
            if rank <= q_count / 4:
                labels_by_rank[i] = labels[0]
            elif rank <= (q_count / 4) * 2:
                labels_by_rank[i] = labels[1]
            elif rank <= (q_count / 4) * 3:
                labels_by_rank[i] = labels[2]
            else:
                labels_by_rank[i] = labels[3]
        return pd.Series(pd.Categorical(labels_by_rank, categories=labels), index=values.index)


def run_robustness_analysis():
    print("=" * 65)
    print("PHASE 11: QUALITY-STRATIFIED ROBUSTNESS ANALYSIS (TEST SET)")
    print("=" * 65)

    manifest_path = PROJECT_ROOT / "data" / "splits" / "official_split_manifest.csv"
    arps_cache_path = PROJECT_ROOT / "data" / "metadata" / "arps_selection_cache.csv"

    manifest_df = pd.read_csv(manifest_path)
    test_manifest = manifest_df[manifest_df["split"] == "test"].copy()

    cache_df = pd.read_csv(arps_cache_path)
    merged_test = pd.merge(test_manifest, cache_df, on="image_id", how="inner")

    # Load test predictions from the 3 systems
    systems = {
        "Baseline": PROJECT_ROOT / "reports" / "baseline_fixed_test_predictions.csv",
        "AGCWD": PROJECT_ROOT / "reports" / "agcwd_adaptive_test_predictions.csv",
        "Proposed_ARPS": PROJECT_ROOT / "reports" / "binary_test_predictions.csv"
    }

    preds_data = {}
    for sys_name, ppath in systems.items():
        if ppath.exists():
            preds_data[sys_name] = pd.read_csv(ppath)
        else:
            print(f"[WARNING] Prediction file not found for {sys_name} at {ppath}")

    if not preds_data:
        print("[ERROR] No system predictions found. Run training/comparisons first.")
        return

    quality_attributes = ["contrast", "sharpness", "noise", "illumination_variation"]
    records = []

    for attr in quality_attributes:
        print(f"\nAnalyzing robustness across quartiles for: {attr.upper()}...")
        quartiles = make_quality_quartiles(merged_test[attr])
        merged_test[f"{attr}_quartile"] = quartiles

        for q_label in ["Q1", "Q2", "Q3", "Q4"]:
            q_ids = set(merged_test[merged_test[f"{attr}_quartile"] == q_label]["image_id"])
            q_subset = merged_test[merged_test["image_id"].isin(q_ids)]
            attr_min = q_subset[attr].min()
            attr_max = q_subset[attr].max()

            for sys_name, df_preds in preds_data.items():
                sys_q = df_preds[df_preds["image_id"].isin(q_ids)]
                if len(sys_q) == 0:
                    continue

                y_true = sys_q["y_true"].values
                y_pred = sys_q["y_pred"].values
                y_prob = sys_q["prob_dr"].values if "prob_dr" in sys_q else None

                metrics = compute_binary_metrics(y_true, y_pred, y_prob)

                records.append({
                    "attribute": attr,
                    "quartile": q_label,
                    "val_min": round(float(attr_min), 3),
                    "val_max": round(float(attr_max), 3),
                    "system": sys_name,
                    "count": len(sys_q),
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"]
                })

    robustness_df = pd.DataFrame(records)
    out_file = PROJECT_ROOT / "reports" / "robustness_results.csv"
    robustness_df.to_csv(out_file, index=False)
    print("\n" + "=" * 65)
    print(f"Robustness results saved to: {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    run_robustness_analysis()
