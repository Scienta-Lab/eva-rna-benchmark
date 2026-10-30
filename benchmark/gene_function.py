"""Shared validation for gene-function class manifests and matrices."""

from pathlib import Path

import pandas as pd


def load_gene_function_classes(
    data_dir: Path,
    task_id: str,
    dataset_id: str,
) -> list[str]:
    """Load the authoritative class list for a gene-function task.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``gene_function_classes.csv``.
    task_id : str
        Gene-function task identifier.
    dataset_id : str
        Dataset identifier declared by the task.

    Returns
    -------
    list[str]
        Classes in their declared evaluation order.
    """
    manifest = pd.read_csv(data_dir / "gene_function_classes.csv", dtype=str)
    selected = manifest[manifest["task_id"].eq(task_id)]
    if selected.empty:
        raise ValueError(f"No gene-function classes declared for {task_id!r}")
    dataset_ids = set(selected["dataset_key"])
    if dataset_ids != {dataset_id}:
        raise ValueError(
            f"{task_id}: expected dataset {dataset_id!r}, "
            f"class manifest declares {sorted(dataset_ids)!r}"
        )
    classes = selected["class"].tolist()
    if len(classes) != len(set(classes)):
        raise ValueError(f"{task_id}: class manifest contains duplicate classes")
    return classes


def validate_gene_function_layout(
    frame: pd.DataFrame,
    *,
    task_id: str,
    classes: list[str],
    seeds: list[int],
) -> None:
    """Require a matrix to match its declared targets and splits exactly.

    Parameters
    ----------
    frame : pd.DataFrame
        Gene-function matrix or its zero-row header.
    task_id : str
        Task identifier used in validation errors.
    classes : list[str]
        Authoritative evaluated classes.
    seeds : list[int]
        Seeds requiring one class-specific split column each.
    """
    expected_targets = {f"target__{class_name}" for class_name in classes}
    actual_targets = {column for column in frame if column.startswith("target__")}
    expected_splits = {
        f"split__{class_name}__seed_{seed}" for class_name in classes for seed in seeds
    }
    actual_splits = {column for column in frame if column.startswith("split__")}
    errors = []
    for kind, expected, actual in (
        ("targets", expected_targets, actual_targets),
        ("splits", expected_splits, actual_splits),
    ):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            errors.append(f"{kind}: missing={missing!r}, unexpected={unexpected!r}")
    if errors:
        raise ValueError(
            f"{task_id}: gene-function layout mismatch; " + "; ".join(errors)
        )
