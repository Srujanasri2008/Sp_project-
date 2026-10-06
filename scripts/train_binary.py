"""
train_binary.py
Trains the proposed multi-domain binary DR screening model:
Architecture: EfficientNet-B0 Spatial + Vessel Structural + Frequency Spectral Fusion
Saves best checkpoint to models/binary/best_model.pt.
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


def run_binary_training(epochs: int = 5, batch_size: int = 16):
    print("=" * 65)
    print("PHASE 6: PROPOSED MULTI-DOMAIN BINARY DR SCREENING TRAINING")
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

    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = DRMultiDomainModel(
        num_classes=2,
        include_vessel=True,
        include_frequency=True,
        dropout_rate=0.3
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=PROJECT_ROOT / "models" / "binary",
        experiment_name="proposed_binary_full",
        device=torch.device("cpu"),
        num_classes=2,
        include_vessel=True,
        include_frequency=True,
        patience=3,
        seed=42
    )

    result = trainer.fit(max_epochs=epochs, resume=True)

    # Final evaluation on locked Test set
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

    # Load best checkpoint
    best_ckpt = torch.load(trainer.best_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_loss, test_metrics, y_true, y_pred, y_prob = trainer.evaluate(test_loader)
    print("\nPROPOSED BINARY MODEL TEST RESULTS:")
    print("-" * 65)
    print(f"  Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  Precision  : {test_metrics['precision']:.4f}")
    print(f"  Recall/Sens: {test_metrics['recall']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1-Score   : {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix (TN, FP, FN, TP): [{test_metrics['tn']}, {test_metrics['fp']}, {test_metrics['fn']}, {test_metrics['tp']}]")
    print("-" * 65)

    # Save results to reports/binary_results.csv
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "binary_results.csv"
    
    record = {
        "model": "Proposed Multi-Domain (ARPS + Spatial + Vessel + Frequency)",
        "num_classes": 2,
        "test_loss": round(test_loss, 4),
        **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
    }
    pd.DataFrame([record]).to_csv(report_file, index=False)
    print(f"Binary screening results saved to: {report_file.relative_to(PROJECT_ROOT)}")

    # Save test predictions for downstream statistical testing & consistency check
    preds_df = pd.DataFrame({
        "image_id": test_loader.dataset.df["image_id"],
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_dr": y_prob[:, 1]
    })
    preds_path = PROJECT_ROOT / "reports" / "binary_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Binary test predictions saved to: {preds_path.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    run_binary_training()
