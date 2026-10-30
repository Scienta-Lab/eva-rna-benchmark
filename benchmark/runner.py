"""Task runner shared by the local CLI and Metaflow."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.data import (
    DatasetSplit,
    load_task_split,
    select_supervised_genes,
    subset_genes,
)
from benchmark.gene_function import (
    load_gene_function_classes,
    validate_gene_function_layout,
)
from benchmark.metrics import supervised_metrics
from models.checkpoints import resolve_checkpoint

PROBE_MAX_ITER = 1_000
PROBE_C = 1.0
PROBE_TOLERANCE = 1e-4
PROBE_SOLVER = "lbfgs"
PROBE_CLASS_WEIGHT = "balanced"
"""Settings of the gene-function logistic probe.

An L2-regularized logistic classifier with C=1, the LBFGS solver and tolerance
1e-4, over a 72/8/20 split. Class weights are balanced because gene annotations
are heavily imbalanced, and an unweighted fit is dominated by the negatives.
"""


def gene_embeddings_with_zero_fill(model: Any, gene_ids: list[str]) -> np.ndarray:
    """Embed every gene, leaving out-of-vocabulary rows at zero.

    Gene-function splits are released per gene and cannot depend on which
    encoder is being evaluated, so the full gene set is kept and unmappable
    genes are represented by a zero vector rather than dropped. Dropping them
    would score each encoder on a different test set.
    """
    known = model.known_gene_mask(gene_ids)
    usable = [gene_id for gene_id, ok in zip(gene_ids, known, strict=True) if ok]
    if not usable:
        raise ValueError("No gene is represented in the encoder vocabulary")
    embedded = model.gene_embeddings(usable)
    embeddings = np.zeros((len(gene_ids), embedded.shape[1]), dtype=embedded.dtype)
    embeddings[known] = embedded
    return embeddings


MODEL_MODULES = {
    "eva": "eva_rna",
    "scgpt": "scgpt",
    "bulkrnabert": "bulkrnabert",
    "linear": "linear",
}
"""Model names accepted on the command line, and the module implementing each."""

MULTI_CHECKPOINT_MODELS = {"eva"}
"""Families distributed as several checkpoints, whose results must stay apart."""


def result_file_name(model_name: str, checkpoint: str, override: str | None) -> str:
    """Name the result file so two checkpoints of one model cannot merge.

    EVA ships as 7M, 60M and 300M. Results are keyed by task and seed, so
    writing all three to `eva.csv` interleaves them and yields a mean over
    whichever ran last -- silently, since nothing records which checkpoint a row
    came from. Naming those by checkpoint keeps them apart. Every other family
    has one checkpoint and keeps its name.
    """
    if override:
        return override
    if model_name in MULTI_CHECKPOINT_MODELS and checkpoint:
        return Path(checkpoint).name.replace("-", "_")
    return model_name


def _task_row(data_dir: Path, task_id: str) -> pd.Series:
    tasks = pd.read_csv(data_dir / "tasks.csv", dtype=str).fillna("")
    selected = tasks[tasks["task_id"] == task_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one task row for {task_id!r}")
    return selected.iloc[0]


def _load_model(
    name: str,
    checkpoint: str,
    device: str,
    vocab_map: str | None,
    batch_size: int | None = None,
) -> Any:
    if name not in MODEL_MODULES:
        raise ValueError(
            f"Unknown model: {name}; expected one of {sorted(MODEL_MODULES)}"
        )
    module = importlib.import_module(f"models.{MODEL_MODULES[name]}")
    kwargs: dict[str, Any] = {
        "checkpoint": resolve_checkpoint(checkpoint),
        "device": device,
        "vocab_map": vocab_map,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return module.create_model(**kwargs)


def _write_results(
    output: Path,
    *,
    family: str,
    task_id: str,
    target: str,
    model_name: str,
    seed: int,
    metrics: dict[str, float],
) -> Path:
    rows = [
        {
            "family": family,
            "task_id": task_id,
            "target": target,
            "model": model_name,
            "seed": seed,
            "split": "test",
            "aggregation": "seed",
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]
    category_dir = output / family
    category_dir.mkdir(parents=True, exist_ok=True)
    destination = category_dir / f"{model_name}.csv"
    new_rows = pd.DataFrame(rows)
    if destination.exists():
        current = pd.read_csv(destination)
        same_task = current["task_id"].eq(task_id)
        replaced = same_task & (
            current["seed"].eq(seed) | ~current["aggregation"].eq("seed")
        )
        new_rows = pd.concat([current.loc[~replaced], new_rows], ignore_index=True)
    task_seed_rows = new_rows[
        new_rows["task_id"].eq(task_id) & new_rows["aggregation"].eq("seed")
    ]
    if task_seed_rows["seed"].nunique() > 1:
        aggregate_rows = []
        group_columns = ["family", "task_id", "target", "model", "split", "metric"]
        for keys, group in task_seed_rows.groupby(group_columns, sort=False):
            base = dict(zip(group_columns, keys, strict=True))
            for aggregation, value in (
                ("mean", group["value"].mean()),
                ("std", group["value"].std(ddof=1)),
            ):
                aggregate_rows.append(
                    {
                        **base,
                        "seed": "",
                        "aggregation": aggregation,
                        "value": float(value),
                    }
                )
        new_rows = pd.concat(
            [new_rows, pd.DataFrame(aggregate_rows)],
            ignore_index=True,
        )
    new_rows = new_rows.sort_values(["task_id", "metric", "aggregation", "seed"])
    new_rows.to_csv(destination, index=False)
    return destination


def _write_class_aggregate_results(
    output: Path,
    *,
    family: str,
    task_id: str,
    target: str,
    model_name: str,
    seed: int,
    class_metrics: list[dict[str, float]],
) -> Path:
    """Record a gene-function score as its mean and spread across classes.

    Gene-function tasks probe each class separately, so the reported score is a
    distribution over classes rather than over seeds. The publication reference
    stores exactly these two aggregates.
    """
    frame = pd.DataFrame(class_metrics)
    rows = [
        {
            "family": family,
            "task_id": task_id,
            "target": target,
            "model": model_name,
            "split": "test",
            "seed": seed,
            "aggregation": aggregation,
            "metric": metric,
            "value": float(value),
        }
        for aggregation, values in (
            ("mean", frame.mean()),
            ("std", frame.std(ddof=1)),
        )
        for metric, value in values.items()
    ]
    category_dir = output / family
    category_dir.mkdir(parents=True, exist_ok=True)
    destination = category_dir / f"{model_name}.csv"
    new_rows = pd.DataFrame(rows)
    if destination.exists():
        current = pd.read_csv(destination)
        replaced = current["task_id"].eq(task_id) & current["seed"].eq(seed)
        new_rows = pd.concat([current.loc[~replaced], new_rows], ignore_index=True)
    new_rows = new_rows.sort_values(["task_id", "metric", "aggregation", "seed"])
    new_rows.to_csv(destination, index=False)
    return destination


def _tensorboard_dir(
    results_dir: Path,
    *,
    model_name: str,
    task_id: str,
    seed: int,
    enabled: bool,
) -> Path | None:
    """Return the per-run scalar log directory, or None when logging is off."""
    if not enabled:
        return None
    return results_dir / "tensorboard" / model_name / task_id / f"seed_{seed}"


def _restrict_to_known_genes(split: DatasetSplit, model: Any, k: int) -> DatasetSplit:
    """Drop genes the encoder cannot tokenize, so selection yields k usable genes."""
    known_gene_mask = getattr(model, "known_gene_mask", None)
    if known_gene_mask is None:
        return split
    known = known_gene_mask(split.gene_ids)
    candidates = [
        gene for gene, usable in zip(split.gene_ids, known, strict=True) if usable
    ]
    if not candidates:
        raise ValueError("No input gene is present in the encoder vocabulary")
    if len(candidates) < k:
        print(
            f"Only {len(candidates)} of {len(split.gene_ids)} genes are in the encoder "
            f"vocabulary; selecting {len(candidates)} instead of {k}",
            flush=True,
        )
    return subset_genes(split, candidates)


def log2_cpm_values(values: np.ndarray) -> np.ndarray:
    """Return released CPM values as log2(CPM+1)."""
    return np.log2(1.0 + np.asarray(values, dtype=np.float64)).astype(np.float32)


def _vocabulary_split(split: DatasetSplit, model: Any) -> DatasetSplit:
    """Drop panel genes the encoder has no token for, as the paper's runs did."""
    known_gene_mask = getattr(model, "known_gene_mask", None)
    if known_gene_mask is None:
        return split
    known = known_gene_mask(split.gene_ids)
    kept = [gene for gene, ok in zip(split.gene_ids, known, strict=True) if ok]
    if not kept:
        raise ValueError("No panel gene is present in the encoder vocabulary")
    if len(kept) < len(split.gene_ids):
        print(
            f"Filtering genes to match encoder vocabulary: "
            f"{len(kept)}/{len(split.gene_ids)} genes retained",
            flush=True,
        )
    return subset_genes(split, kept)


def apply_model_preprocessing(model: Any, expression: np.ndarray) -> np.ndarray:
    """Apply the model's released-CPM transform, or pass values through.

    Every path that hands expression to an encoder must call this: datasets ship
    CPM, so skipping it silently feeds unnormalized values to the model.
    """
    preprocess = getattr(model, "preprocess_expression", None)
    if preprocess is None:
        return expression
    return preprocess(expression)


def _preprocess_split(split: DatasetSplit, model: Any) -> DatasetSplit:
    """Apply the model's own normalization to the released CPM matrices.

    Datasets ship as CPM so each encoder applies its published transform:
    log2(CPM+1) for EVA and the linear baseline, log10(1+CPM) for BulkRNABert,
    quantile binning for scGPT.

    This runs after the released panel has been loaded, so that panel stays as
    published. It runs before `_reselect_genes`, so a run that passes an explicit
    gene count instead fits K-best on the values the encoder consumes.
    """
    if getattr(model, "preprocess_expression", None) is None:
        return split
    return DatasetSplit(
        gene_ids=split.gene_ids,
        train_x=apply_model_preprocessing(model, split.train_x),
        val_x=apply_model_preprocessing(model, split.val_x),
        test_x=apply_model_preprocessing(model, split.test_x),
        train_y=split.train_y,
        val_y=split.val_y,
        test_y=split.test_y,
    )


def _reselect_genes(
    split: DatasetSplit,
    model: Any,
    *,
    task_type: str,
    k: int,
) -> DatasetSplit:
    """Refit the K-best panel over genes the encoder can actually consume."""
    split = _restrict_to_known_genes(split, model, k)
    selected = select_supervised_genes(
        split.train_x,
        split.train_y,
        split.gene_ids,
        task_type=task_type,
        k=k,
    )
    return subset_genes(split, selected)


def run_supervised(
    *,
    task_id: str,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    seed: int,
    device: str,
    vocab_map: str | None,
    result_model_name: str | None = None,
    tensorboard: bool = False,
    feature_count: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Run a regression or classification task."""
    task = _task_row(data_dir, task_id)
    family = task["family"]
    if family not in {"classification", "regression"}:
        raise ValueError(f"{task_id} is a {family} task, not a supervised task")
    dataset_path = data_dir / task["file"]
    model = _load_model(model_name, checkpoint, device, vocab_map, batch_size)
    released_panel = feature_count is None
    split = load_task_split(
        dataset_path,
        task_id=task_id,
        target=task["target"],
        seed=seed,
        feature_sets_path=(
            data_dir / "feature_sets.csv.gz" if released_panel else None
        ),
    )
    if not released_panel:
        selected = _reselect_genes(
            _preprocess_split(split, model),
            model,
            task_type=family,
            k=feature_count,
        )
        split = subset_genes(split, selected.gene_ids)
    if getattr(model, "n_bins", None) is not None:
        model.inputs_prebinned = True

    if model_name == "linear":
        split = _preprocess_split(split, model)
        train_x = model.encode_samples(split.train_x, split.gene_ids)
        test_x = model.encode_samples(split.test_x, split.gene_ids)
        estimator = model.fit(train_x, split.train_y, family, seed=seed)
        if family == "classification":
            probability = estimator.predict_proba(test_x)
            class_to_index = {
                value: index for index, value in enumerate(estimator.classes_)
            }
            y_true = np.array([class_to_index[value] for value in split.test_y])
            metrics = supervised_metrics(y_true, probability, family)
        else:
            metrics = supervised_metrics(
                split.test_y, estimator.predict(test_x), family
            )
    else:
        from benchmark.supervised_head import fit_head, predict_head, protocol_for

        split = _vocabulary_split(split, model)
        log_x = {
            name: log2_cpm_values(getattr(split, f"{name}_x"))
            for name in ("train", "val", "test")
        }
        fitted = fit_head(
            model,
            (log_x["train"], split.train_y),
            (log_x["val"], split.val_y),
            split.gene_ids,
            task_type=family,
            seed=seed,
            protocol=protocol_for(task_id),
            log_dir=_tensorboard_dir(
                results_dir,
                model_name=result_file_name(model_name, checkpoint, result_model_name),
                task_id=task_id,
                seed=seed,
                enabled=tensorboard,
            ),
        )
        prediction = predict_head(
            model, fitted, log_x["test"], split.gene_ids, task_type=family
        )
        if family == "classification":
            class_to_index = {
                value: index for index, value in enumerate(fitted.classes)
            }
            y_true = np.array([class_to_index[value] for value in split.test_y])
        else:
            y_true = split.test_y
        metrics = supervised_metrics(y_true, prediction, family)

    return _write_results(
        results_dir,
        family=family,
        task_id=task_id,
        target=task["target"],
        model_name=result_file_name(model_name, checkpoint, result_model_name),
        seed=seed,
        metrics=metrics,
    )


def run_gene_function(
    *,
    task_id: str,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    seed: int,
    device: str,
    vocab_map: str | None,
    result_model_name: str | None = None,
) -> Path:
    """Run a gene-function logistic probe on frozen gene embeddings.

    Reads the train and test splits only. The released matrices carry the
    published 72/8/20 split, but the probe has no tuned hyperparameter, so the
    8% validation slice is never selected on and those genes go unused.
    """
    from sklearn.linear_model import LogisticRegression

    task = _task_row(data_dir, task_id)
    dataset_path = data_dir / task["file"]
    frame = pd.read_csv(dataset_path, dtype={"gene_id": str})
    classes = load_gene_function_classes(data_dir, task_id, task["dataset_id"])
    seeds = [int(value) for value in task["seeds"].split("|")]
    validate_gene_function_layout(
        frame,
        task_id=task_id,
        classes=classes,
        seeds=seeds,
    )
    model = _load_model(model_name, checkpoint, device, vocab_map)
    embeddings = gene_embeddings_with_zero_fill(model, frame["gene_id"].tolist())
    class_metrics: list[dict[str, float]] = []
    for class_name in classes:
        target_column = f"target__{class_name}"
        split_column = f"split__{class_name}__seed_{seed}"
        if split_column not in frame:
            raise ValueError(f"Missing gene-function split column {split_column}")
        train = frame[split_column] == "train"
        test = frame[split_column] == "test"
        y_true = frame.loc[test, target_column].to_numpy(dtype=int)
        if len(np.unique(y_true)) < 2:
            print(
                f"{class_name}: test split holds a single label, excluded",
                flush=True,
            )
            continue
        probe = LogisticRegression(
            C=PROBE_C,
            solver=PROBE_SOLVER,
            tol=PROBE_TOLERANCE,
            max_iter=PROBE_MAX_ITER,
            random_state=seed,
            class_weight=PROBE_CLASS_WEIGHT,
        ).fit(embeddings[train], frame.loc[train, target_column].astype(int))
        probability = probe.predict_proba(embeddings[test])
        class_metrics.append(supervised_metrics(y_true, probability, "classification"))
    if not class_metrics:
        raise ValueError(f"{dataset_path} has no scorable target__<class> columns")
    return _write_class_aggregate_results(
        results_dir,
        family=task["family"],
        task_id=task_id,
        target=task["target"],
        model_name=result_file_name(model_name, checkpoint, result_model_name),
        seed=seed,
        class_metrics=class_metrics,
    )


def _check_seed_is_available(task: pd.Series, seed: int) -> None:
    """Reject a seed the task's released matrix carries no split for.

    Seeds are not uniform across families: supervised and perturbation tasks
    ship 10-14, gene function ships only 42. Without this the mismatch surfaces
    deep in the runner as a missing column.
    """
    available = [int(value) for value in task["seeds"].split("|")]
    if seed not in available:
        raise ValueError(
            f"{task['task_id']} has no split for seed {seed}; "
            f"it ships seeds {', '.join(str(value) for value in available)}"
        )


def run_task(**kwargs: Any) -> Path:
    """Dispatch a manifest task to the corresponding runner."""
    kwargs["results_dir"] = Path(kwargs["results_dir"])
    data_dir = Path(kwargs["data_dir"])
    task = _task_row(data_dir, kwargs["task_id"])
    _check_seed_is_available(task, kwargs["seed"])
    kwargs["data_dir"] = data_dir
    if task["family"] in {"classification", "regression"}:
        return run_supervised(**kwargs)
    if task["family"] == "perturbation":
        from benchmark.perturbation import run_perturbation

        kwargs.pop("feature_count", None)
        return run_perturbation(**kwargs)
    kwargs.pop("tensorboard", None)
    kwargs.pop("feature_count", None)
    kwargs.pop("batch_size", None)
    if task["family"] == "gene_function":
        return run_gene_function(**kwargs)
    raise NotImplementedError(
        json.dumps(
            {
                "task_id": kwargs["task_id"],
                "family": task["family"],
                "message": "Use benchmark.perturbation or benchmark.zsp directly.",
            }
        )
    )
