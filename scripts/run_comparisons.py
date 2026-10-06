"""
run_comparisons.py
Fair controlled comparison across three systems (Binary Screening):
- System 1: Baseline Fixed Preprocessing (Path A CLAHE + Spatial EfficientNet-B0)
- System 2: Existing Adaptive Enhancement (AGCWD + Spatial EfficientNet-B0)
- System 3: Proposed ARPS Adaptive Processing Selection (Spatial EfficientNet-B0)
Maintains identical split, model family, optimizer, epochs, and random seed.
Saves comparative evaluation results to reports/three_system_comparison.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pandas as pd

from src.utils.reproducibility import set_seed
from src.data.dataset import create_dataloader
from src.models.network import DRMultiDomainModel
from src.training.trainer import DRTrainer


def train_and_eval_system(system_name: str, mode: str, epochs: int = 5, batch_size: int = 16):
    print("\n" + "=" * 65)
    print(f"RUNNING COMPARISON EXPERIMENT: {system_name.upper()} (mode='{mode}')")
    print("=" * 65)

    set_seed(42)
    torch.set_num_threads(4)

    train_loader = create_dataloader(
        split="train",
        mode=mode,
        include_vessel=False,
        include_frequency=False,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        split="val",
        mode=mode,
        include_vessel=False,
        include_frequency=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = create_dataloader(
        split="test",
        mode=mode,
        include_vessel=False,
        include_frequency=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = DRMultiDomainModel(
        num_classes=2,
        include_vessel=False,
        include_frequency=False,
        dropout_rate=0.3
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    ckpt_dir = PROJECT_ROOT / "models" / "comparison" / system_name
    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=ckpt_dir,
        experiment_name=f"comparison_{system_name}",
        device=torch.device("cpu"),
        num_classes=2,
        include_vessel=False,
        include_frequency=False,
        patience=3,
        seed=42
    )

    trainer.fit(max_epochs=epochs, resume=True)

    # Evaluate best model on Test set
    best_ckpt = torch.load(trainer.best_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics, y_true, y_pred, y_prob = trainer.evaluate(test_loader)

    # Save test predictions for statistical comparison (McNemar tests)
    preds_df = pd.DataFrame({
        "image_id": test_loader.dataset.df["image_id"],
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_dr": y_prob[:, 1]
    })
    preds_path = PROJECT_ROOT / "reports" / f"{system_name}_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path.relative_to(PROJECT_ROOT)}")

    return {
        "system": system_name,
        "mode": mode,
        "test_loss": round(test_loss, 4),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall_sensitivity": test_metrics["recall"],
        "specificity": test_metrics["specificity"],
        "f1_score": test_metrics["f1_score"],
        "roc_auc": test_metrics["roc_auc"],
        "tp": test_metrics["tp"],
        "fp": test_metrics["fp"],
        "tn": test_metrics["tn"],
        "fn": test_metrics["fn"]
    }


def run_all_comparisons(epochs: int = 5, batch_size: int = 16):
    print("=" * 65)
    print("PHASE 9: THREE-SYSTEM FAIR CONTROLLED COMPARISON")
    print("=" * 65)

    systems = [
        ("baseline_fixed", "baseline"),
        ("agcwd_adaptive", "agcwd"),
        ("proposed_arps_spatial", "proposed_arps")
    ]

    results = []
    for name, mode in systems:
        res = train_and_eval_system(name, mode, epochs=epochs, batch_size=batch_size)
        results.append(res)

    results_df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "reports" / "three_system_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("THREE-SYSTEM COMPARISON SUMMARY (LOCKED TEST SET):")
    print("-" * 65)
    print(results_df[["system", "accuracy", "sensitivity" if "sensitivity" in results_df else "recall_sensitivity",
                      "specificity", "f1_score", "roc_auc"]].to_string(index=False))
    print("-" * 65)
    print(f"Results report saved to: {output_path.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_comparisons()
