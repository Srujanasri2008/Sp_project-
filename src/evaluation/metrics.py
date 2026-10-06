"""
metrics.py
Comprehensive evaluation metrics and consistency checking:
- Binary Screening Metrics (Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, Confusion Matrix)
- Five-Class Severity Metrics (Accuracy, Macro/Per-class Precision, Recall, F1, QWK, Confusion Matrix)
- Binary / Five-Class Consistency Checker
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes full clinical screening metrics for binary DR classification.
    0: Normal, 1: DR
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    sensitivity = rec
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                prob_dr = y_prob[:, 1]
            else:
                prob_dr = y_prob
            auc = float(roc_auc_score(y_true, prob_dr))
        except Exception:
            auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def compute_five_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes ordinal DR severity grading metrics.
    0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: PDR
    """
    classes = [0, 1, 2, 3, 4]
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))

    # Quadratic Weighted Kappa (QWK) is standard for ordinal DR severity grading
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    # Per-class metrics
    per_prec = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    per_class_report = {}
    for c, name in enumerate(class_names):
        count_true = int(np.sum(y_true == c))
        count_pred = int(np.sum(y_pred == c))
        per_class_report[name] = {
            "class_id": c,
            "true_count": count_true,
            "pred_count": count_pred,
            "precision": round(float(per_prec[c]), 4),
            "recall": round(float(per_rec[c]), 4),
            "f1_score": round(float(per_f1[c]), 4)
        }

    # Multiclass ROC-AUC (One-vs-Rest)
    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "qwk": round(qwk, 4),
        "roc_auc_macro": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report
    }


def consistency_check(
    binary_pred: int,
    five_class_pred: int,
    binary_prob: Optional[float] = None,
    five_class_prob: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluates semantic consistency between independent binary and five-class models:
    Rule:
      five_class == 0 -> derived binary = 0 (Normal)
      five_class in {1, 2, 3, 4} -> derived binary = 1 (DR)
      
    Returns:
      Dict with match/warning status, derived labels, and resolved clinical interpretation.
    """
    derived_binary = 0 if five_class_pred == 0 else 1
    is_match = (binary_pred == derived_binary)
    status = "MATCH" if is_match else "WARNING"

    severity_names = {
        0: "No DR (Normal)",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR (PDR)"
    }
    
    # Clinically consistent final interpretation:
    # Severity grading governs the final interpretation, and any binary disagreement is reported explicitly.
    if five_class_pred == 0:
        clinical_interpretation = "Normal (No Diabetic Retinopathy detected)"
    else:
        clinical_interpretation = f"Diabetic Retinopathy Detected - Severity: {severity_names[five_class_pred]}"

    explanation = None
    if not is_match:
        explanation = (
            f"Model Consistency Warning: Independent binary classifier predicted "
            f"{'DR' if binary_pred == 1 else 'Normal'}, whereas ordinal five-class model graded "
            f"{severity_names[five_class_pred]} (derived binary: {'DR' if derived_binary == 1 else 'Normal'}). "
            f"The final interpretation reflects the five-class severity grading for logical consistency."
        )

    return {
        "binary_prediction": binary_pred,
        "five_class_prediction": five_class_pred,
        "derived_binary_from_five_class": derived_binary,
        "is_consistent": is_match,
        "consistency_status": status,
        "clinical_interpretation": clinical_interpretation,
        "diagnostic_note": explanation
    }


def evaluate_consistency_across_dataset(
    binary_preds: np.ndarray,
    five_class_preds: np.ndarray
) -> Dict[str, Any]:
    """
    Computes dataset-wide consistency rate between binary and five-class model predictions.
    """
    derived_binaries = np.where(five_class_preds == 0, 0, 1)
    matches = (binary_preds == derived_binaries)
    consistency_rate = float(np.mean(matches))
    total = len(binary_preds)
    mismatch_indices = np.where(~matches)[0].tolist()

    return {
        "total_evaluated": total,
        "consistent_count": int(np.sum(matches)),
        "inconsistent_count": int(total - np.sum(matches)),
        "consistency_rate": round(consistency_rate, 4),
        "mismatch_indices_sample": mismatch_indices[:20]
    }
