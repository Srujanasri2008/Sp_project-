"""
train_five_class.py
Trains the proposed multi-domain five-class ordinal DR severity model:
0 = No DR, 1 = Mild DR, 2 = Moderate DR, 3 = Severe DR, 4 = Proliferative DR (PDR).
Features:
- EfficientNet-B0 Spatial + Vessel Structural + Frequency Spectral Fusion
- Explicit class-imbalance aware weighted Cross-Entropy Loss
- Quadratic Weighted Kappa (QWK) & Macro-F1 metric tracking
Saves best checkpoint to models/five_class/best_model.pt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.utils.reproducibility import set_seed
from src.data.dataset import create_dataloader
from src.models.network import DRMultiDomainModel
from src.training.trainer import DRTrainer


def run_five_class_training(epochs: int = 5, batch_size: int = 16):
    print("=" * 65)
    print("PHASE 7: PROPOSED MULTI-DOMAIN FIVE-CLASS DR SEVERITY TRAINING")
    print("=" * 65)

    set_seed(42)
    torch.set_num_threads(4)

    train_loader = create_dataloader(
        split="train",
        mode="proposed_arps",
        include_vessel=True,
        include_frequency=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        split="val",
        mode="proposed_arps",
        include_vessel=True,
        include_frequency=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Class imbalance handling
    train_counts = train_loader.dataset.df["five_class"].value_counts().sort_index().values
    total_train = len(train_loader.dataset)
    class_weights = total_train / (5.0 * train_counts.astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Class Imbalance Analysis (Train Set):")
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    for c, (cnt, w) in enumerate(zip(train_counts, class_weights)):
        print(f"  Class {c} ({class_names[c]:<16}): {cnt:>5} samples | Weight: {w:.3f}")

    model = DRMultiDomainModel(
        num_classes=5,
        include_vessel=True,
        include_frequency=True,
        dropout_rate=0.3
    )

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=PROJECT_ROOT / "models" / "five_class",
        experiment_name="proposed_five_class_full",
        device=torch.device("cpu"),
        num_classes=5,
        include_vessel=True,
        include_frequency=True,
        patience=3,
        seed=42
    )

    result = trainer.fit(max_epochs=epochs, resume=True)

    # Locked final evaluation on Test set
    print("\nRunning locked final evaluation on Test set...")
    test_loader = create_dataloader(
        split="test",
        mode="proposed_arps",
        include_vessel=True,
        include_frequency=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    best_ckpt = torch.load(trainer.best_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics, y_true, y_pred, y_prob = trainer.evaluate(test_loader)
    print("\nPROPOSED FIVE-CLASS MODEL TEST RESULTS:")
    print("-" * 65)
    print(f"  Accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {test_metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score : {test_metrics['macro_f1']:.4f}")
    print(f"  QWK (Kappa)    : {test_metrics['qwk']:.4f}")
    print(f"  Macro ROC-AUC  : {test_metrics['roc_auc_macro']:.4f}")
    print("\nPer-Class Breakdown:")
    for cname, stats in test_metrics["per_class"].items():
        print(f"  {cname:<18}: True={stats['true_count']:>3}, Pred={stats['pred_count']:>3} | "
              f"P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1_score']:.3f}")
    print("-" * 65)

    # Save to reports/five_class_results.csv
    report_file = PROJECT_ROOT / "reports" / "five_class_results.csv"
    summary_record = {
        "model": "Proposed Multi-Domain (ARPS + Spatial + Vessel + Frequency)",
        "num_classes": 5,
        "test_loss": round(test_loss, 4),
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "qwk": test_metrics["qwk"],
        "roc_auc_macro": test_metrics["roc_auc_macro"]
    }
    pd.DataFrame([summary_record]).to_csv(report_file, index=False)
    print(f"Five-class severity results saved to: {report_file.relative_to(PROJECT_ROOT)}")

    # Save test predictions for downstream statistical testing & consistency check
    preds_df = pd.DataFrame({
        "image_id": test_loader.dataset.df["image_id"],
        "y_true": y_true,
        "y_pred": y_pred,
        **{f"prob_c{i}": y_prob[:, i] for i in range(5)}
    })
    preds_path = PROJECT_ROOT / "reports" / "five_class_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Five-class test predictions saved to: {preds_path.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    run_five_class_training()
