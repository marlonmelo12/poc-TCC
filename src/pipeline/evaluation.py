"""
Performance evaluation metrics module.
Calculates Accuracy, Precision, Recall, F1, and ROC-AUC.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = np.nan
    if y_prob is not None:
        try:
            # If 2D prob array, take prob of positive class (class 1)
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                scores = y_prob[:, 1]
            else:
                scores = y_prob.flatten()
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }
