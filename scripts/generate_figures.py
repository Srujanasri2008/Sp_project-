"""Generate presentation figures from saved test-set analysis reports."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


SYSTEMS = {
    "Baseline": "baseline_fixed_test_predictions.csv",
    "AGCWD": "agcwd_adaptive_test_predictions.csv",
    "Proposed ARPS": "binary_test_predictions.csv",
}


def load_prediction_reports():
    reports_dir = PROJECT_ROOT / "reports"
    predictions = {}
    for system, filename in SYSTEMS.items():
        path = reports_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required test predictions not found: {path}")
        predictions[system] = pd.read_csv(path)
    return predictions


def plot_confusion_matrices(predictions):
    output_dir = PROJECT_ROOT / "figures" / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    for system, frame in predictions.items():
        fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
        ConfusionMatrixDisplay.from_predictions(
            frame["y_true"],
            frame["y_pred"],
            display_labels=["No DR", "DR"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"{system} - Locked Test Set")
        fig.savefig(output_dir / f"{system.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)


def plot_roc_curves(predictions):
    output_path = PROJECT_ROOT / "figures" / "roc_curves" / "locked_test_roc.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for system, frame in predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            frame["y_true"], frame["prob_dr"]
        )
        score = auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, linewidth=2,
                label=f"{system} (AUC {score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title="Locked Test Set ROC Curves")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ablation_metrics():
    report_path = PROJECT_ROOT / "reports" / "ablation_results.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Ablation report not found: {report_path}")
    frame = pd.read_csv(report_path)
    output_path = PROJECT_ROOT / "figures" / "ablations" / "ablation_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]
    labels = frame["ablation"].str.replace("_", " ").str.title()
    x = np.arange(len(frame))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - (len(metrics) - 1) / 2) * width,
               frame[metric], width, label=metric.replace("_", " ").title())
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Locked Test Set Ablation Results")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_robustness_metrics():
    report_path = PROJECT_ROOT / "reports" / "robustness_results.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Robustness report not found: {report_path}")
    frame = pd.read_csv(report_path)
    output_path = PROJECT_ROOT / "figures" / "robustness" / "robustness_f1_by_quality.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attributes = frame["attribute"].dropna().unique().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, constrained_layout=True)
    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    for ax, attribute in zip(axes.flat, attributes):
        subset = frame[frame["attribute"] == attribute]
        for system, system_frame in subset.groupby("system"):
            grouped = system_frame.set_index("quartile")["f1_score"]
            values = [grouped.get(label, np.nan) for label in quartile_order]
            ax.plot(quartile_order, values, marker="o", linewidth=1.8, label=system)
        ax.set_title(attribute.replace("_", " ").title())
        ax.set_xlabel("Image quality quartile")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 score")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Quality-Stratified Test Set Robustness", y=1.04)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    predictions = load_prediction_reports()
    plot_confusion_matrices(predictions)
    plot_roc_curves(predictions)
    plot_ablation_metrics()
    plot_robustness_metrics()
    print("Generated figures:")
    for folder in ("confusion_matrices", "roc_curves", "ablations", "robustness"):
        paths = sorted((PROJECT_ROOT / "figures" / folder).glob("*.png"))
        for path in paths:
            print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()