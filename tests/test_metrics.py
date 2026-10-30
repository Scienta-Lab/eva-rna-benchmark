"""Tests for the supervised metric set."""

import numpy as np

from benchmark.metrics import supervised_metrics


def test_multiclass_auroc_is_reported_as_auroc() -> None:
    """tasks.csv declares primary_metric=auroc for multi-class tasks too."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probability = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = supervised_metrics(y_true, probability, "classification")
    assert "auroc" in metrics
    assert "auroc_ovr_macro" not in metrics
    assert metrics["auroc"] == 1.0


def test_binary_auroc_keeps_auprc() -> None:
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    metrics = supervised_metrics(y_true, probability, "classification")
    assert metrics["auroc"] == 1.0
    assert "auprc" in metrics
