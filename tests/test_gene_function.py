"""Tests for the gene-function probe protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.runner import gene_embeddings_with_zero_fill, run_gene_function


class _PartialVocabularyModel:
    """Encoder that represents only even-numbered genes."""

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        return np.array([int(gene_id) % 2 == 0 for gene_id in gene_ids], dtype=bool)

    def gene_embeddings(self, gene_ids: list[str]) -> np.ndarray:
        return np.array(
            [[float(gene_id), 1.0] for gene_id in gene_ids], dtype=np.float32
        )


def test_out_of_vocabulary_genes_are_zero_filled_not_dropped() -> None:
    embeddings = gene_embeddings_with_zero_fill(
        _PartialVocabularyModel(), ["2", "3", "4"]
    )
    assert embeddings.shape == (3, 2)
    assert embeddings[0].tolist() == [2.0, 1.0]
    assert embeddings[1].tolist() == [0.0, 0.0]
    assert embeddings[2].tolist() == [4.0, 1.0]


def _release(root: Path) -> Path:
    """Write a task whose second class has no positive gene in its test split."""
    data_dir = root / "datasets"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "family": "gene_function",
                "task_id": "genes",
                "dataset_id": "genes",
                "file": "genes.csv.gz",
                "target": "*",
                "seeds": "42",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    pd.DataFrame(
        {
            "task_id": ["genes", "genes", "genes"],
            "dataset_key": ["genes", "genes", "genes"],
            "class": ["scored", "also_scored", "degenerate"],
        }
    ).to_csv(data_dir / "gene_function_classes.csv", index=False)
    n_genes = 40
    scored = np.zeros(n_genes, dtype=int)
    scored[:12] = 1
    scored[30:33] = 1
    also_scored = np.zeros(n_genes, dtype=int)
    also_scored[6:18] = 1
    also_scored[33:36] = 1
    degenerate = np.zeros(n_genes, dtype=int)
    degenerate[:6] = 1
    split = ["train"] * 30 + ["test"] * 10
    pd.DataFrame(
        {
            "gene_id": [str(index) for index in range(n_genes)],
            "target__scored": scored,
            "split__scored__seed_42": split,
            "target__also_scored": also_scored,
            "split__also_scored__seed_42": split,
            "target__degenerate": degenerate,
            "split__degenerate__seed_42": split,
        }
    ).to_csv(data_dir / "genes.csv.gz", index=False)
    return data_dir


def test_class_with_a_single_labelled_test_split_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from benchmark import runner

    monkeypatch.setattr(
        runner, "_load_model", lambda *_, **__: _PartialVocabularyModel()
    )
    data_dir = _release(tmp_path)
    destination = run_gene_function(
        task_id="genes",
        model_name="stub",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=42,
        device="cpu",
        vocab_map=None,
    )
    reported = capsys.readouterr().out
    assert "degenerate: test split holds a single label, excluded" in reported
    assert "scored:" not in reported
    written = pd.read_csv(destination)
    assert set(written["aggregation"]) == {"mean", "std"}


def test_results_are_aggregated_across_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmark import runner

    monkeypatch.setattr(
        runner, "_load_model", lambda *_, **__: _PartialVocabularyModel()
    )
    data_dir = _release(tmp_path)
    destination = run_gene_function(
        task_id="genes",
        model_name="stub",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=42,
        device="cpu",
        vocab_map=None,
    )
    written = pd.read_csv(destination)
    auroc = written[written["metric"].eq("auroc")]
    assert sorted(auroc["aggregation"]) == ["mean", "std"]
    assert auroc["seed"].unique().tolist() == [42]
