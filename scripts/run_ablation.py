"""
run_ablation.py
Domain contribution ablations under ARPS candidate selection:
- Ablation 1: Proposed ARPS + Spatial Only (1280-dim)
- Ablation 2: Proposed ARPS + Spatial + Vessel Structural (1408-dim)
- Ablation 3: Proposed ARPS + Spatial + Frequency Spectral (1408-dim)
- Ablation 4: Full Proposed ARPS + Spatial + Vessel + Frequency Fusion (1536-dim)
Saves results to reports/ablation_results.csv.
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


def train_and_eval_ablation(
    name: str,
    include_vessel: bool,
    include_frequency: bool,
    epochs: int = 5,
    batch_size: int = 16
):
    print("\n" + "=" * 65)
    print(f"RUNNING ABLATION: {name.upper()}")
    print(f"Vessel: {include_vessel} | Frequency: {include_frequency}")
    print("=" * 65)

    set_seed(42)
    torch.set_num_threads(4)

    train_loader = create_dataloader(
        split="train",
        mode="proposed_arps",
        include_vessel=include_vessel,
        include_frequency=include_frequency,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        split="val",
        mode="proposed_arps",
        include_vessel=include_vessel,
        include_frequency=include_frequency,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = create_dataloader(
        split="test",
        mode="proposed_arps",
        include_vessel=include_vessel,
        include_frequency=include_frequency,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = DRMultiDomainModel(
        num_classes=2,
        include_vessel=include_vessel,
        include_frequency=include_frequency,
        dropout_rate=0.3
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    ckpt_dir = PROJECT_ROOT / "models" / "comparison" / f"ablation_{name}"
    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=ckpt_dir,
        experiment_name=f"ablation_{name}",
        device=torch.device("cpu"),
        num_classes=2,
        include_vessel=include_vessel,
        include_frequency=include_frequency,
        patience=3,
        seed=42
    )

    trainer.fit(max_epochs=epochs, resume=True)

    best_ckpt = torch.load(trainer.best_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics, y_true, y_pred, y_prob = trainer.evaluate(test_loader)

    # Save test predictions
    preds_df = pd.DataFrame({
        "image_id": test_loader.dataset.df["image_id"],
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_dr": y_prob[:, 1]
    })
    preds_path = PROJECT_ROOT / "reports" / f"ablation_{name}_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    return {
        "ablation": name,
        "vessel_branch": include_vessel,
        "frequency_branch": include_frequency,
        "feature_dim": 1280 + (128 if include_vessel else 0) + (128 if include_frequency else 0),
        "test_loss": round(test_loss, 4),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "sensitivity": test_metrics["recall"],
        "specificity": test_metrics["specificity"],
        "f1_score": test_metrics["f1_score"],
        "roc_auc": test_metrics["roc_auc"]
    }


def run_all_ablations(epochs: int = 5, batch_size: int = 16):
    print("=" * 65)
    print("PHASE 10: PROPOSED MULTI-DOMAIN ABLATION STUDY")
    print("=" * 65)

    configurations = [
        ("spatial_only", False, False),
        ("spatial_plus_vessel", True, False),
        ("spatial_plus_frequency", False, True),
        ("full_multidomain_fusion", True, True)
    ]

    results = []
    for name, v_flag, f_flag in configurations:
        res = train_and_eval_ablation(name, v_flag, f_flag, epochs=epochs, batch_size=batch_size)
        results.append(res)

    results_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "reports" / "ablation_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("DOMAIN ABLATION STUDY SUMMARY (LOCKED TEST SET):")
    print("-" * 65)
    print(results_df[["ablation", "feature_dim", "accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]].to_string(index=False))
    print("-" * 65)
    print(f"Ablation results saved to: {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_ablations()
