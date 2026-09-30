"""Classifier metrics on a forward-only holdout."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Labels and probabilities must have the same nonzero length")
    if not np.isfinite(probability).all() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Invalid labels or probabilities")
    if np.unique(y_true).size != 2:
        raise ValueError("Both classes are needed for held-out ranking metrics")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be between zero and one")

    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, predicted)),
        "precision_at_0_5": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
