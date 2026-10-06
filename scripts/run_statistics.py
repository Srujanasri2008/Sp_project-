"""
run_statistics.py
Statistical evaluation on frozen test predictions:
1. Pairwise McNemar tests with continuity correction:
   - Baseline vs AGCWD
   - Baseline vs Proposed ARPS
   - AGCWD vs Proposed ARPS
2. Holm-Bonferroni correction for multiple hypothesis testing
3. Quality-performance association analysis via Logistic Regression:
   correctness ~ standardized_quality + true_label
Saves results to reports/statistical_results.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar


def compute_mcnemar_pairwise(preds_a: pd.DataFrame, preds_b: pd.DataFrame, name_a: str, name_b: str):
    """
    Computes McNemar test contingency table and p-value between two models.
    Contingency table:
      Both correct | A correct, B wrong
      B correct, A wrong | Both wrong
    """
    merged = pd.merge(preds_a[["image_id", "y_true", "y_pred"]],
                      preds_b[["image_id", "y_pred"]],
                      on="image_id", suffixes=(f"_{name_a}", f"_{name_b}"))
    
    corr_a = (merged[f"y_pred_{name_a}"] == merged["y_true"])
    corr_b = (merged[f"y_pred_{name_b}"] == merged["y_true"])

    # Contingency matrix
    n11 = int(np.sum(corr_a & corr_b))
    n10 = int(np.sum(corr_a & (~corr_b)))  # A right, B wrong
    n01 = int(np.sum((~corr_a) & corr_b))  # A wrong, B right
    n00 = int(np.sum((~corr_a) & (~corr_b)))

    table = [[n11, n10],
             [n01, n00]]

    # Exact McNemar if off-diagonal sum < 25, else chi-square with continuity correction
    exact = (n10 + n01) < 25
    res = mcnemar(table, exact=exact, correction=True)

    stat = float(getattr(res, "statistic", 0.0))
    pval = float(res.pvalue)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "both_correct (n11)": n11,
        f"{name_a}_only_correct (n10)": n10,
        f"{name_b}_only_correct (n01)": n01,
        "both_incorrect (n00)": n00,
        "statistic": round(stat, 4),
        "raw_p_value": pval,
        "exact_test": exact
    }


def holm_bonferroni_correction(p_values: list) -> list:
    """Applies step-down Holm-Bonferroni correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = p * (m - rank)
        adj_p = min(1.0, max(running_max, adj_p))
        running_max = adj_p
        corrected[orig_idx] = adj_p

    return corrected


def run_statistical_analysis():
    print("=" * 65)
    print("PHASE 12: STATISTICAL TESTING & ASSOCIATION ANALYSIS")
    print("=" * 65)

    base_path = PROJECT_ROOT / "reports" / "baseline_fixed_test_predictions.csv"
    agcwd_path = PROJECT_ROOT / "reports" / "agcwd_adaptive_test_predictions.csv"
    prop_path = PROJECT_ROOT / "reports" / "binary_test_predictions.csv"

    if not (base_path.exists() and agcwd_path.exists() and prop_path.exists()):
        print("[ERROR] Test predictions for all 3 systems must exist to compute statistics.")
        return

    df_base = pd.read_csv(base_path)
    df_agcwd = pd.read_csv(agcwd_path)
    df_prop = pd.read_csv(prop_path)

    print("\n[1] Running Pairwise McNemar Tests...")
    comps = [
        compute_mcnemar_pairwise(df_base, df_agcwd, "Baseline", "AGCWD"),
        compute_mcnemar_pairwise(df_base, df_prop, "Baseline", "Proposed"),
        compute_mcnemar_pairwise(df_agcwd, df_prop, "AGCWD", "Proposed")
    ]

    p_vals = [c["raw_p_value"] for c in comps]
    adj_p_vals = holm_bonferroni_correction(p_vals)

    for c, adj_p in zip(comps, adj_p_vals):
        c["holm_corrected_p_value"] = round(adj_p, 5)
        c["significant_at_0.05"] = adj_p < 0.05
        print(f"  {c['comparison']:<25}: stat={c['statistic']:.3f}, raw_p={c['raw_p_value']:.4f}, "
              f"Holm_adj_p={adj_p:.4f} (Sig: {c['significant_at_0.05']})")

    # [2] Quality-Performance Association Analysis (Logistic Regression)
    # correctness ~ standardized_quality + true_label
    print("\n[2] Quality-Performance Association Analysis (Proposed Model)...")
    cache_path = PROJECT_ROOT / "data" / "metadata" / "arps_selection_cache.csv"
    cache_df = pd.read_csv(cache_path)

    merged = pd.merge(df_prop, cache_df, on="image_id", how="inner")
    merged["correct"] = (merged["y_pred"] == merged["y_true"]).astype(int)

    assoc_records = []
    qual_attrs = ["contrast", "sharpness", "noise", "illumination_variation"]

    for attr in qual_attrs:
        # Standardize quality attribute (Z-score)
        val = merged[attr].values
        std_val = (val - np.mean(val)) / (np.std(val) + 1e-7)
        merged[f"{attr}_z"] = std_val

        X = sm.add_constant(merged[[f"{attr}_z", "y_true"]])
        y = merged["correct"]

        logit_model = sm.Logit(y, X).fit(disp=False)
        params = logit_model.params
        conf = logit_model.conf_int()
        pvals = logit_model.pvalues

        odds_ratio = float(np.exp(params[f"{attr}_z"]))
        ci_lower = float(np.exp(conf.loc[f"{attr}_z"][0]))
        ci_upper = float(np.exp(conf.loc[f"{attr}_z"][1]))
        pval = float(pvals[f"{attr}_z"])

        assoc_records.append({
            "quality_metric": attr,
            "effect_size_coef": round(float(params[f"{attr}_z"]), 4),
            "odds_ratio": round(odds_ratio, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "raw_p_value": round(pval, 5),
            "holm_adj_p_value": 0.0  # will fill below
        })

    assoc_pvals = [r["raw_p_value"] for r in assoc_records]
    assoc_adj = holm_bonferroni_correction(assoc_pvals)
    for r, adj in zip(assoc_records, assoc_adj):
        r["holm_adj_p_value"] = round(adj, 5)
        print(f"  {r['quality_metric']:<24}: OR={r['odds_ratio']:.3f} (95% CI: [{r['ci_95_lower']:.3f}, {r['ci_95_upper']:.3f}]), "
              f"p={r['raw_p_value']:.4f}, Holm_p={adj:.4f}")

    # Save to reports/statistical_results.csv
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mcnemar_df = pd.DataFrame(comps)
    mcnemar_df.to_csv(out_dir / "mcnemar_pairwise_results.csv", index=False)
    
    assoc_df = pd.DataFrame(assoc_records)
    assoc_df.to_csv(out_dir / "quality_association_results.csv", index=False)

    print("\n" + "=" * 65)
    print("Statistical analysis completed. Artifacts saved:")
    print(f"  - reports/mcnemar_pairwise_results.csv")
    print(f"  - reports/quality_association_results.csv")
    print("Note: Associational models do not imply causal relationships.")
    print("=" * 65)


if __name__ == "__main__":
    run_statistical_analysis()
