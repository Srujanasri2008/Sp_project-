"""
trainer.py
Reusable, resource-safe training engine for Binary and Five-Class Diabetic Retinopathy models.
Features:
- Conservative memory management and CPU thread control
- Early stopping & validation checkpointing (saves best_model.pt and latest_model.pt only)
- Complete checkpoint state: epoch, weights, optimizer, validation metrics, seed, config
- Resumable training
- Detailed training history logs
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_binary_metrics, compute_five_class_metrics


class DRTrainer:
    """
    Standardized trainer for DR models across all experiments.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: Path,
        experiment_name: str,
        device: torch.device = torch.device("cpu"),
        num_classes: int = 2,
        include_vessel: bool = True,
        include_frequency: bool = True,
        patience: int = 3,
        seed: int = 42
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.device = device
        self.num_classes = num_classes
        self.include_vessel = include_vessel
        self.include_frequency = include_frequency
        self.patience = patience
        self.seed = seed

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_model.pt"

    def _prepare_batch_inputs(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        x_spatial = batch["spatial"].to(self.device)
        x_vessel = batch["vessel"].to(self.device) if self.include_vessel else None
        x_freq = batch["frequency"].to(self.device) if self.include_frequency else None
        
        if self.num_classes == 2:
            y = batch["binary_label"].to(self.device)
        else:
            y = batch["five_class_label"].to(self.device)
            
        return x_spatial, x_vessel, x_freq, y

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x_spatial, x_vessel, x_freq, y = self._prepare_batch_inputs(batch)
            self.optimizer.zero_grad()

            logits = self.model(x_spatial, x_vessel, x_freq)
            loss = self.criterion(logits, y)
            loss.backward()

            # Gradient clipping to maintain numerical stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total_samples += len(y)

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)
        return avg_loss, acc

    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        eval_loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                x_spatial, x_vessel, x_freq, y = self._prepare_batch_inputs(batch)
                logits = self.model(x_spatial, x_vessel, x_freq)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * len(y)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_targets.extend(y.cpu().numpy())
                total_samples += len(y)

        avg_loss = total_loss / max(1, total_samples)
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        if self.num_classes == 2:
            metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        else:
            metrics = compute_five_class_metrics(y_true, y_pred, y_prob)

        metrics["val_loss"] = round(avg_loss, 4)
        return avg_loss, metrics, y_true, y_pred, y_prob

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float, metrics: Dict[str, Any], is_best: bool = False):
        state = {
            "epoch": epoch,
            "experiment_name": self.experiment_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": metrics,
            "num_classes": self.num_classes,
            "include_vessel": self.include_vessel,
            "include_frequency": self.include_frequency,
            "random_seed": self.seed,
            "is_best": is_best
        }
        torch.save(state, path)

    def fit(self, max_epochs: int = 5, resume: bool = True) -> Dict[str, Any]:
        start_epoch = 1
        best_metric = -float("inf")
        patience_counter = 0
        history = []

        # Resume if requested and latest exists
        if resume and self.latest_checkpoint_path.exists():
            print(f"[+] Resuming from existing checkpoint: {self.latest_checkpoint_path}")
            checkpoint = torch.load(self.latest_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            if self.best_checkpoint_path.exists():
                best_ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
                best_metric = best_ckpt["val_metrics"].get("f1_score", best_ckpt["val_metrics"].get("qwk", 0.0))
            print(f"  Resumed at epoch {start_epoch}, previous best metric: {best_metric:.4f}")

        print(f"\nBeginning training for '{self.experiment_name}' ({max_epochs} epochs max)...")
        print("-" * 65)

        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_metrics, _, _, _ = self.evaluate()
            elapsed = time.time() - t0

            # Target metric: F1-score for binary, QWK for five-class
            current_metric = val_metrics.get("f1_score", val_metrics.get("qwk", val_metrics["accuracy"]))

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "time_sec": round(elapsed, 1)
            }
            history.append(epoch_record)

            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                patience_counter = 0
                self.save_checkpoint(self.best_checkpoint_path, epoch, val_loss, val_metrics, is_best=True)
                mark = "[BEST]"
            else:
                patience_counter += 1
                mark = "      "

            # Always save latest checkpoint
            self.save_checkpoint(self.latest_checkpoint_path, epoch, val_loss, val_metrics, is_best=False)

            metric_name = "F1" if self.num_classes == 2 else "QWK"
            print(
                f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f}, Val {metric_name}: {current_metric:.4f} | {mark} ({elapsed:.1f}s)"
            )

            if patience_counter >= self.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch} (no improvement for {self.patience} epochs).")
                break

        print("-" * 65)
        print(f"Training completed. Best model saved to: {self.best_checkpoint_path.relative_to(PROJECT_ROOT)}")

        # Save history log
        log_dir = PROJECT_ROOT / "logs" / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.experiment_name}_history.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": self.experiment_name, "best_metric": best_metric, "history": history}, f, indent=2)

        return {"best_metric": best_metric, "history": history}
