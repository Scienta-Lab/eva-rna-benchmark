"""Tests for consolidated suite execution."""

from pathlib import Path

import pandas as pd

from benchmark.suite import run_regression_suite


def test_regression_suite_writes_seed_and_aggregate_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "regression",
                "task_id": "task",
                "dataset_id": "dataset",
                "file": "dataset.csv.gz",
                "target": "score",
                "seeds": "10|11",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    frame = pd.DataFrame(
        {
            "gene__1": list(range(10)),
            "target__score": list(range(10)),
            "split__task__seed_10": [
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "test",
                "test",
                "test",
                "test",
            ],
            "split__task__seed_11": [
                "test",
                "test",
                "test",
                "test",
                "val",
                "val",
                "train",
                "train",
                "train",
                "train",
            ],
        }
    )
    frame.to_csv(data_dir / "dataset.csv.gz", index=False)
    pd.DataFrame(
        {
            "task_id": ["task", "task"],
            "seed": [10, 11],
            "gene_id": ["1", "1"],
            "rank": [0, 0],
        }
    ).to_csv(data_dir / "feature_sets.csv.gz", index=False)
    output = run_regression_suite(
        model_name="linear",
        checkpoint="",
        data_dir=data_dir,
        results_dir=results_dir,
    )
    results = pd.read_csv(output)
    assert output == results_dir / "regression" / "linear.csv"
    assert list(results.columns) == [
        "family",
        "task_id",
        "target",
        "model",
        "split",
        "seed",
        "aggregation",
        "metric",
        "value",
    ]
    assert set(results["aggregation"]) == {"seed", "mean", "std"}
    assert len(results[results["aggregation"].eq("seed")]) == 10
    assert len(results[results["aggregation"].eq("mean")]) == 5
    assert len(results[results["aggregation"].eq("std")]) == 5
