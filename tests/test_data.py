"""Tests for release preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.data import (
    DatasetSplit,
    load_task_split,
    log2_cpm,
    select_de_genes,
    select_supervised_genes,
    subset_genes,
)
from benchmark.metrics import perturbation_metrics


def test_log2_cpm_normalizes_before_log() -> None:
    counts = np.array([[1, 1], [0, 0]], dtype=float)
    transformed = log2_cpm(counts)
    np.testing.assert_allclose(transformed[0], np.log2([500_001, 500_001]))
    np.testing.assert_array_equal(transformed[1], [0, 0])


def test_supervised_selection_uses_requested_task_score() -> None:
    features = np.array([[0, 5], [1, 5], [2, 5], [3, 5]], dtype=float)
    targets = np.array([0.0, 1.0, 2.0, 3.0])
    selected = select_supervised_genes(
        features, targets, ["signal", "constant"], task_type="regression", k=1
    )
    assert selected == ["signal"]


def test_subset_genes_reorders_every_split_consistently() -> None:
    split = DatasetSplit(
        gene_ids=["a", "b", "c"],
        train_x=np.array([[1.0, 2.0, 3.0]]),
        val_x=np.array([[4.0, 5.0, 6.0]]),
        test_x=np.array([[7.0, 8.0, 9.0]]),
        train_y=np.array([0.0]),
        val_y=np.array([1.0]),
        test_y=np.array([2.0]),
    )
    reduced = subset_genes(split, ["c", "a"])
    assert reduced.gene_ids == ["c", "a"]
    np.testing.assert_array_equal(reduced.train_x, [[3.0, 1.0]])
    np.testing.assert_array_equal(reduced.val_x, [[6.0, 4.0]])
    np.testing.assert_array_equal(reduced.test_x, [[9.0, 7.0]])
    np.testing.assert_array_equal(reduced.train_y, split.train_y)


def test_subset_genes_rejects_unknown_genes() -> None:
    split = DatasetSplit(
        gene_ids=["a"],
        train_x=np.zeros((1, 1)),
        val_x=np.zeros((1, 1)),
        test_x=np.zeros((1, 1)),
        train_y=np.zeros(1),
        val_y=np.zeros(1),
        test_y=np.zeros(1),
    )
    with pytest.raises(ValueError, match="missing 1 genes"):
        subset_genes(split, ["b"])


def test_de_selection_appends_required_target() -> None:
    baseline = np.zeros((2, 3))
    target = np.array([[5, 1, 0], [5, 1, 0]])
    selected = select_de_genes(
        baseline,
        target,
        ["1", "2", "3"],
        k=1,
        required_gene_ids={"3"},
    )
    assert selected == ["1", "3"]


def test_task_split_uses_seed_feature_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv.gz"
    feature_sets = tmp_path / "feature_sets.csv.gz"
    pd.DataFrame(
        {
            "gene__1": [1.0, 2.0, 3.0],
            "gene__2": [4.0, 5.0, 6.0],
            "target__score": [1.0, 2.0, 3.0],
            "split__task__seed_10": ["train", "val", "test"],
        }
    ).to_csv(dataset, index=False)
    pd.DataFrame(
        {
            "task_id": ["task"],
            "seed": [10],
            "gene_id": ["2"],
            "rank": [0],
        }
    ).to_csv(feature_sets, index=False)
    split = load_task_split(
        dataset,
        task_id="task",
        target="score",
        seed=10,
        feature_sets_path=feature_sets,
    )
    assert split.gene_ids == ["2"]
    np.testing.assert_array_equal(split.train_x, [[4.0]])


def test_perturbation_pearson_is_mean_samplewise_correlation() -> None:
    baseline = np.zeros((2, 3), dtype=np.float32)
    observed = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    predicted = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32)
    metrics = perturbation_metrics(observed, predicted, baseline)
    assert metrics["pearson"] == 0.0


def test_task_split_canonicalizes_label_casing(tmp_path: Path) -> None:
    """Mixed-case labels must collapse to one class, not several."""
    path = tmp_path / "endotype.csv.gz"
    pd.DataFrame(
        {
            "sample_id": [str(index) for index in range(6)],
            "gene__1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "target__endotype": [
                "Lymphoid",
                "lymphoid",
                " Myeloid ",
                "myeloid",
                "Lymphoid",
                "myeloid",
            ],
            "split__endo__seed_10": [
                "train",
                "train",
                "train",
                "train",
                "val",
                "test",
            ],
        }
    ).to_csv(path, index=False)
    split = load_task_split(path, task_id="endo", target="endotype", seed=10)
    assert set(np.unique(split.train_y)) == {"lymphoid", "myeloid"}
    assert split.val_y.tolist() == ["lymphoid"]
    assert split.test_y.tolist() == ["myeloid"]


def test_regression_targets_are_left_numeric(tmp_path: Path) -> None:
    path = tmp_path / "score.csv.gz"
    pd.DataFrame(
        {
            "sample_id": ["0", "1", "2"],
            "gene__1": [0.0, 1.0, 2.0],
            "target__score": [1.5, 2.5, 3.5],
            "split__task__seed_10": ["train", "val", "test"],
        }
    ).to_csv(path, index=False)
    split = load_task_split(path, task_id="task", target="score", seed=10)
    assert split.train_y.dtype.kind == "f"
    assert split.train_y.tolist() == [1.5]
