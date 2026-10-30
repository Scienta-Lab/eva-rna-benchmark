"""Metrics emitted by the benchmark harness."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def supervised_metrics(
    y_true: np.ndarray, prediction: np.ndarray, task_type: str
) -> dict[str, float]:
    """Compute the supervised metric set."""
    if task_type == "classification":
        labels = prediction.argmax(axis=1)
        scores = prediction[:, 1] if prediction.shape[1] == 2 else prediction
        metrics = {
            "accuracy": float(accuracy_score(y_true, labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
            "f1_macro": float(f1_score(y_true, labels, average="macro")),
        }
        if prediction.shape[1] == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, scores))
            metrics["auprc"] = float(average_precision_score(y_true, scores))
        else:
            metrics["auroc"] = float(
                roc_auc_score(y_true, scores, multi_class="ovr", average="macro")
            )
        return metrics
    return {
        "pearson": float(pearsonr(y_true, prediction).statistic),
        "spearman": float(spearmanr(y_true, prediction).statistic),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def perturbation_metrics(
    observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    """Compare expression changes using mean sample-wise correlations."""

    def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        numerator = (left_centered * right_centered).sum(axis=1)
        denominator = np.sqrt(
            (left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1)
        )
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return float(correlations.mean())

    def stable_ranks(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values, axis=1, kind="stable"), axis=1).astype(
            np.float32
        )

    observed_delta = observed - baseline
    predicted_delta = predicted - baseline
    return {
        "pearson": rowwise_pearson(observed_delta, predicted_delta),
        "spearman": rowwise_pearson(
            stable_ranks(observed_delta),
            stable_ranks(predicted_delta),
        ),
        "mae": float(mean_absolute_error(observed.reshape(-1), predicted.reshape(-1))),
    }
