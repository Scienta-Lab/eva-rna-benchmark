"""Run a complete manifest family and write one consolidated result CSV."""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from benchmark.metrics import supervised_metrics
from benchmark.runner import (
    _load_model,
    apply_model_preprocessing,
)


def _selected_gene_columns(
    feature_sets: pd.DataFrame,
    task_id: str,
    seed: int,
) -> list[str]:
    selected = feature_sets[
        feature_sets["task_id"].eq(task_id) & feature_sets["seed"].eq(seed)
    ].sort_values("rank")
    if selected.empty:
        raise ValueError(f"No features found for {task_id} seed {seed}")
    if selected["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate features found for {task_id} seed {seed}")
    return [f"gene__{gene_id}" for gene_id in selected["gene_id"]]


def _append_aggregates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seed_rows = pd.DataFrame(rows)
    group_columns = ["family", "task_id", "target", "model", "split", "metric"]
    aggregates: list[dict[str, object]] = []
    for keys, group in seed_rows.groupby(group_columns, sort=False):
        base = dict(zip(group_columns, keys, strict=True))
        for aggregation, value in (
            ("mean", group["value"].mean()),
            ("std", group["value"].std(ddof=1)),
        ):
            aggregates.append(
                {
                    **base,
                    "seed": "",
                    "aggregation": aggregation,
                    "value": float(value),
                }
            )
    return rows + aggregates


def _run_regression_task(
    task: dict[str, str],
    *,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    device: str,
    vocab_map: str | None,
) -> list[dict[str, object]]:
    feature_sets = pd.read_csv(
        data_dir / "feature_sets.csv.gz",
        dtype={"task_id": str, "gene_id": str},
    )
    dataset_path = data_dir / task["file"]
    frame = pd.read_csv(dataset_path)
    target_column = f"target__{task['target']}"
    model = _load_model(model_name, checkpoint, device, vocab_map)
    rows: list[dict[str, object]] = []
    for seed_text in task["seeds"].split("|"):
        seed = int(seed_text)
        gene_columns = _selected_gene_columns(
            feature_sets,
            task["task_id"],
            seed,
        )
        missing = set(gene_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{task['task_id']} seed {seed}: {len(missing)} genes are missing"
            )
        split_column = f"split__{task['task_id']}__seed_{seed}"
        train = frame[split_column].eq("train")
        test = frame[split_column].eq("test")
        train_expression = frame.loc[train, gene_columns].to_numpy(dtype=np.float32)
        test_expression = frame.loc[test, gene_columns].to_numpy(dtype=np.float32)
        gene_ids = [column.removeprefix("gene__") for column in gene_columns]
        train_features = model.encode_samples(
            apply_model_preprocessing(model, train_expression), gene_ids
        )
        test_features = model.encode_samples(
            apply_model_preprocessing(model, test_expression), gene_ids
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator = model.fit(
                train_features,
                frame.loc[train, target_column].to_numpy(),
                "regression",
                seed=seed,
            )
        prediction = estimator.predict(test_features)
        metrics = supervised_metrics(
            frame.loc[test, target_column].to_numpy(),
            prediction,
            "regression",
        )
        for metric, value in metrics.items():
            rows.append(
                {
                    "family": "regression",
                    "task_id": task["task_id"],
                    "target": task["target"],
                    "model": model_name,
                    "split": "test",
                    "seed": seed,
                    "aggregation": "seed",
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def run_regression_suite(
    *,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    device: str = "cpu",
    vocab_map: str | None = None,
    workers: int = 1,
) -> Path:
    """Run every released regression task and seed."""
    if model_name != "linear":
        raise ValueError("The consolidated regression suite currently supports linear")
    if workers < 1:
        raise ValueError("workers must be at least one")
    tasks = pd.read_csv(data_dir / "tasks.csv", dtype=str).fillna("")
    tasks = tasks[tasks["family"].eq("regression")]
    rows: list[dict[str, object]] = []
    task_records = tasks.to_dict(orient="records")
    results_dir.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        for task in task_records:
            rows.extend(
                _run_regression_task(
                    task,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    data_dir=data_dir,
                    device=device,
                    vocab_map=vocab_map,
                )
            )
            _write_results(rows, results_dir, model_name)
            print(f"Completed {task['task_id']}", flush=True)
        return results_dir / "regression" / f"{model_name}.csv"

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_regression_task,
                task,
                model_name=model_name,
                checkpoint=checkpoint,
                data_dir=data_dir,
                device=device,
                vocab_map=vocab_map,
            ): task["task_id"]
            for task in task_records
        }
        for future in as_completed(futures):
            task_id = futures[future]
            rows.extend(future.result())
            _write_results(rows, results_dir, model_name)
            print(f"Completed {task_id}", flush=True)
    return results_dir / "regression" / f"{model_name}.csv"


def _write_results(
    rows: list[dict[str, object]],
    results_dir: Path,
    model_name: str,
) -> None:
    partial = pd.DataFrame(_append_aggregates(rows)).sort_values(
        ["task_id", "metric", "aggregation", "seed"]
    )
    category_dir = results_dir / "regression"
    category_dir.mkdir(parents=True, exist_ok=True)
    partial.to_csv(category_dir / f"{model_name}.csv", index=False)


def main() -> None:
    """Run a regression suite from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="linear")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-map")
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    output = run_regression_suite(
        model_name=arguments.model,
        checkpoint=arguments.checkpoint,
        data_dir=arguments.data_dir,
        results_dir=arguments.results_dir,
        device=arguments.device,
        vocab_map=arguments.vocab_map,
        workers=arguments.workers,
    )
    print(output)


if __name__ == "__main__":
    main()
