"""CSV loading and benchmark preprocessing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

GENE_PREFIX = "gene__"
TARGET_PREFIX = "target__"


@dataclass(frozen=True)
class DatasetSplit:
    """A materialized task split."""

    gene_ids: list[str]
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log2_cpm(counts: np.ndarray, target_sum: float = 1_000_000.0) -> np.ndarray:
    """Normalize non-negative counts to log2 counts-per-million."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix")
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")
    totals = values.sum(axis=1, keepdims=True)
    scale = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals),
        where=totals != 0,
    )
    return np.log2(values * scale + 1.0).astype(np.float32)


def select_supervised_genes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    gene_ids: list[str],
    *,
    task_type: str,
    k: int = 8_000,
) -> list[str]:
    """Fit the K-best selector on training samples only."""
    if len(gene_ids) != train_x.shape[1]:
        raise ValueError("gene_ids and train_x have inconsistent dimensions")
    score_func = f_classif if task_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=min(k, train_x.shape[1]))
    selector.fit(train_x, train_y)
    return [
        gene
        for gene, keep in zip(gene_ids, selector.get_support(), strict=False)
        if keep
    ]


def subset_genes(split: DatasetSplit, gene_ids: list[str]) -> DatasetSplit:
    """Restrict every split matrix to gene_ids, preserving the given order."""
    positions = {gene: index for index, gene in enumerate(split.gene_ids)}
    missing = [gene for gene in gene_ids if gene not in positions]
    if missing:
        raise ValueError(f"Split is missing {len(missing)} genes: {missing[:3]}")
    columns = [positions[gene] for gene in gene_ids]
    return DatasetSplit(
        gene_ids=list(gene_ids),
        train_x=split.train_x[:, columns],
        val_x=split.val_x[:, columns],
        test_x=split.test_x[:, columns],
        train_y=split.train_y,
        val_y=split.val_y,
        test_y=split.test_y,
    )


def select_de_genes(
    baseline_x: np.ndarray,
    target_x: np.ndarray,
    gene_ids: list[str],
    *,
    k: int,
    required_gene_ids: set[str] | None = None,
) -> list[str]:
    """Rank genes by absolute mean log2-expression change."""
    if baseline_x.shape[1] != target_x.shape[1] or len(gene_ids) != baseline_x.shape[1]:
        raise ValueError("expression matrices and gene_ids are inconsistent")
    scores = np.abs(target_x.mean(axis=0) - baseline_x.mean(axis=0))
    order = np.argsort(-scores, kind="stable")[: min(k, len(gene_ids))]
    selected = [gene_ids[index] for index in order]
    required = required_gene_ids or set()
    selected_set = set(selected)
    selected.extend(gene for gene in gene_ids if gene in required - selected_set)
    return selected


def load_task_split(
    path: Path,
    *,
    task_id: str,
    target: str,
    seed: int,
    feature_sets_path: Path | None = None,
) -> DatasetSplit:
    """Load one task/seed split from a release CSV."""
    frame = pd.read_csv(path)
    gene_columns = [column for column in frame if column.startswith(GENE_PREFIX)]
    if feature_sets_path is not None and feature_sets_path.is_file():
        feature_sets = pd.read_csv(
            feature_sets_path,
            dtype={"task_id": str, "gene_id": str},
        )
        selected = feature_sets[
            feature_sets["task_id"].eq(task_id) & feature_sets["seed"].eq(seed)
        ].sort_values("rank")
        if selected.empty:
            raise ValueError(f"No feature set found for {task_id} seed {seed}")
        gene_columns = [f"{GENE_PREFIX}{gene_id}" for gene_id in selected["gene_id"]]
        missing_genes = set(gene_columns) - set(frame.columns)
        if missing_genes:
            raise ValueError(f"{path} is missing {len(missing_genes)} selected genes")
    split_column = f"split__{task_id}__seed_{seed}"
    target_column = f"{TARGET_PREFIX}{target}"
    missing = {split_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[target_column].dtype == object:
        frame[target_column] = frame[target_column].astype(str).str.strip().str.lower()

    def take(split: str) -> tuple[np.ndarray, np.ndarray]:
        rows = frame[split_column] == split
        return (
            frame.loc[rows, gene_columns].to_numpy(dtype=np.float32),
            frame.loc[rows, target_column].to_numpy(),
        )

    train_x, train_y = take("train")
    val_x, val_y = take("val")
    test_x, test_y = take("test")
    if min(len(train_x), len(val_x), len(test_x)) == 0:
        raise ValueError(f"{task_id} seed {seed} has an empty split")
    return DatasetSplit(
        gene_ids=[column.removeprefix(GENE_PREFIX) for column in gene_columns],
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
    )
