"""End-to-end smoke test for the manifest-driven harness."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.runner import (
    _write_results,
    result_file_name,
    run_task,
)


def test_linear_regression_task_writes_tidy_results(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "regression",
                "task_id": "smoke",
                "dataset_id": "smoke",
                "file": "smoke.csv.gz",
                "target": "score",
                "seeds": "10|11",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": [str(index) for index in range(9)],
            "gene__1": list(range(9)),
            "gene__2": [1] * 9,
            "target__score": list(range(9)),
            "split__smoke__seed_10": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "test",
                "test",
            ],
        }
    ).to_csv(data_dir / "smoke.csv.gz", index=False)
    destination = run_task(
        task_id="smoke",
        model_name="linear",
        checkpoint="",
        data_dir=data_dir,
        results_dir=results_dir,
        seed=10,
        device="cpu",
        vocab_map=None,
    )
    result = pd.read_csv(destination)
    assert destination == results_dir / "regression" / "linear.csv"
    assert list(result.columns) == [
        "family",
        "task_id",
        "target",
        "model",
        "seed",
        "split",
        "aggregation",
        "metric",
        "value",
    ]
    assert set(result["metric"]) == {"pearson", "spearman", "rmse", "mae", "r2"}
    assert set(result["task_id"]) == {"smoke"}


def test_result_writer_merges_tasks_and_recomputes_aggregates(
    tmp_path: Path,
) -> None:
    common = {
        "output": tmp_path,
        "family": "regression",
        "target": "score",
        "model_name": "model",
    }
    _write_results(
        **common,
        task_id="first",
        seed=10,
        metrics={"pearson": 0.2},
    )
    destination = _write_results(
        **common,
        task_id="first",
        seed=11,
        metrics={"pearson": 0.4},
    )
    _write_results(
        **common,
        task_id="second",
        seed=10,
        metrics={"pearson": 0.6},
    )

    result = pd.read_csv(destination)
    first = result[result["task_id"].eq("first")]
    assert len(result[result["aggregation"].eq("seed")]) == 3
    assert first.loc[first["aggregation"].eq("mean"), "value"].item() == pytest.approx(
        0.3
    )
    assert round(first.loc[first["aggregation"].eq("std"), "value"].item(), 3) == 0.141


def _write_cross_species_task(data_dir: Path) -> None:
    """Write a two-panel perturbation task, with its manifest and gene panels."""
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "perturbation",
                "task_id": "cross_species",
                "dataset_id": "cross_species",
                "file": "cross_species.csv.gz",
                "target": "expression",
                "seeds": "10|11",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    pd.DataFrame(
        {
            "task_id": ["cross_species"] * 4,
            "seed": [10] * 4,
            "scope": ["train", "train", "test", "test"],
            "gene_id": ["1", "2", "2", "3"],
            "rank": [0, 1, 0, 1],
        }
    ).to_csv(data_dir / "perturbation_feature_sets.csv.gz", index=False)
    pd.DataFrame(
        {
            "sample_id": ["train_1", "train_2", "val", "test"],
            "baseline__gene__1": [1.0, 2.0, 3.0, float("nan")],
            "baseline__gene__2": [2.0, 3.0, 4.0, 5.0],
            "baseline__gene__3": [float("nan"), float("nan"), float("nan"), 6.0],
            "target__gene__1": [2.0, 3.0, 4.0, float("nan")],
            "target__gene__2": [4.0, 5.0, 6.0, 7.0],
            "target__gene__3": [float("nan"), float("nan"), float("nan"), 7.0],
            "split__cross_species__seed_10": ["train", "train", "val", "test"],
        }
    ).to_csv(data_dir / "cross_species.csv.gz", index=False)


def test_average_perturbation_supports_distinct_gene_panels(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "datasets"
    results_dir = tmp_path / "results"
    _write_cross_species_task(data_dir)

    destination = run_task(
        task_id="cross_species",
        model_name="linear",
        checkpoint="",
        data_dir=data_dir,
        results_dir=results_dir,
        seed=10,
        device="cpu",
        vocab_map=None,
    )

    result = pd.read_csv(destination)
    pearson = result.loc[result["metric"].eq("pearson"), "value"].item()
    assert pearson == pytest.approx(1.0)


def test_perturbation_names_results_like_the_supervised_runner(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "datasets"
    _write_cross_species_task(data_dir)

    destination = run_task(
        task_id="cross_species",
        model_name="linear",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=10,
        device="cpu",
        vocab_map=None,
        result_model_name="renamed",
    )

    assert destination.name == "renamed.csv"
    assert set(pd.read_csv(destination)["model"]) == {"renamed"}


@pytest.mark.parametrize(
    ("model_name", "checkpoint", "override", "expected"),
    [
        ("eva", "eva-60m", None, "eva_60m"),
        ("eva", "/checkpoints/eva-300m", None, "eva_300m"),
        ("eva", "eva-60m", "custom", "custom"),
        ("scgpt", "scgpt-human", None, "scgpt"),
        ("linear", "linear", None, "linear"),
    ],
)
def test_result_file_name_keeps_eva_checkpoints_apart(
    model_name: str, checkpoint: str, override: str | None, expected: str
) -> None:
    assert result_file_name(model_name, checkpoint, override) == expected


def test_preprocessing_runs_before_the_panel_is_fitted() -> None:
    """K-best must see the encoder's own transform, not the released CPM."""
    from benchmark.data import DatasetSplit
    from benchmark.runner import _preprocess_split, _reselect_genes

    class ColumnSwapper:
        """Stand-in adapter whose transform inverts the raw gene ranking."""

        def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
            return expression[:, ::-1].copy()

    split = DatasetSplit(
        gene_ids=["a", "b"],
        train_x=np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]),
        val_x=np.array([[0.0, 1.0]]),
        test_x=np.array([[0.0, 2.0]]),
        train_y=np.array([1.0, 2.0, 3.0, 4.0]),
        val_y=np.array([1.0]),
        test_y=np.array([2.0]),
    )
    model = ColumnSwapper()
    reduced = _reselect_genes(
        _preprocess_split(split, model), model, task_type="regression", k=1
    )
    assert reduced.gene_ids == ["a"]
    np.testing.assert_array_equal(reduced.train_x, [[1.0], [2.0], [3.0], [4.0]])


def test_preprocessing_is_skipped_for_models_without_a_transform() -> None:
    from benchmark.data import DatasetSplit
    from benchmark.runner import _preprocess_split

    split = DatasetSplit(
        gene_ids=["a"],
        train_x=np.array([[3.0]]),
        val_x=np.array([[4.0]]),
        test_x=np.array([[5.0]]),
        train_y=np.zeros(1),
        val_y=np.zeros(1),
        test_y=np.zeros(1),
    )

    class NoTransform:
        pass

    assert _preprocess_split(split, NoTransform()) is split


def test_prebinned_inputs_are_not_binned_again() -> None:
    """Setting inputs_prebinned must disable the adapter's own binning."""
    from models.scgpt import prepare_scgpt_batch

    class FakeVocab:
        values = {"<cls>": 0, "<pad>": 1, "7": 2}

        def __contains__(self, key: str) -> bool:
            return key in self.values

        def __getitem__(self, key: str) -> int:
            return self.values[key]

    expression = np.array([[4.0]], dtype=np.float32)
    _, unbinned, _ = prepare_scgpt_batch(
        expression, ["7"], FakeVocab(), None, n_bins=None
    )
    assert float(unbinned[0, 1]) == 4.0


def test_reselect_genes_without_binning_uses_raw_values() -> None:
    """Non-binning encoders must still honour an explicit feature_count."""
    from benchmark.data import DatasetSplit
    from benchmark.runner import _reselect_genes

    class NeverBins:
        def bin_expression(self, expression: np.ndarray) -> np.ndarray:
            raise AssertionError("bin_expression must not be called")

    split = DatasetSplit(
        gene_ids=["signal", "noise"],
        train_x=np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]]),
        val_x=np.array([[1.0, 5.0]]),
        test_x=np.array([[2.0, 5.0]]),
        train_y=np.array([1.0, 2.0, 3.0, 4.0]),
        val_y=np.array([1.0]),
        test_y=np.array([2.0]),
    )
    reduced = _reselect_genes(split, NeverBins(), task_type="regression", k=1)
    assert reduced.gene_ids == ["signal"]


class _VocabularyEncoder:
    """Encoder that only tokenizes an explicit gene whitelist."""

    def __init__(self, known: set[str]) -> None:
        self.known = known

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        return np.array([gene in self.known for gene in gene_ids], dtype=bool)


def _linear_split(gene_ids: list[str]) -> "object":
    from benchmark.data import DatasetSplit

    rows = np.arange(1, 5, dtype=float)
    train_x = np.stack([rows * (index + 1) for index in range(len(gene_ids))], axis=1)
    return DatasetSplit(
        gene_ids=list(gene_ids),
        train_x=train_x,
        val_x=train_x[:1],
        test_x=train_x[:1],
        train_y=rows,
        val_y=rows[:1],
        test_y=rows[:1],
    )


def test_selection_only_considers_genes_in_the_encoder_vocabulary() -> None:
    """The panel must be filled from tokenizable genes, not silently shrunk."""
    from benchmark.runner import _reselect_genes

    split = _linear_split(["a", "b", "c", "d"])
    encoder = _VocabularyEncoder({"c", "d"})
    reduced = _reselect_genes(split, encoder, task_type="regression", k=2)
    assert reduced.gene_ids == ["c", "d"]


def test_selection_warns_when_the_vocabulary_cannot_fill_the_panel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from benchmark.runner import _reselect_genes

    split = _linear_split(["a", "b", "c"])
    reduced = _reselect_genes(
        split, _VocabularyEncoder({"b"}), task_type="regression", k=3
    )
    assert reduced.gene_ids == ["b"]
    assert "selecting 1 instead of 3" in capsys.readouterr().out


def test_selection_rejects_a_disjoint_vocabulary() -> None:
    from benchmark.runner import _reselect_genes

    split = _linear_split(["a", "b"])
    with pytest.raises(ValueError, match="No input gene is present"):
        _reselect_genes(
            split,
            _VocabularyEncoder(set()),
            task_type="regression",
            k=1,
        )


def test_models_without_a_vocabulary_keep_every_gene_as_a_candidate() -> None:
    """The linear baseline exposes no known_gene_mask and must not be filtered."""
    from benchmark.runner import _reselect_genes

    class NoVocabulary:
        pass

    split = _linear_split(["a", "b", "c"])
    reduced = _reselect_genes(split, NoVocabulary(), task_type="regression", k=3)
    assert reduced.gene_ids == ["a", "b", "c"]


def test_apply_model_preprocessing_is_the_single_normalization_entry_point() -> None:
    """Every path feeding an encoder must route through this helper."""
    from benchmark.runner import apply_model_preprocessing

    class Doubler:
        def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
            return expression * 2.0

    class NoTransform:
        pass

    values = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(
        apply_model_preprocessing(Doubler(), values), [[2.0, 4.0]]
    )
    assert apply_model_preprocessing(NoTransform(), values) is values


def test_direct_frame_readers_normalize_before_encoding() -> None:
    """suite and perturbation read matrices themselves and once bypassed it.

    runner.py is exempt: it normalizes the whole split up front in
    _preprocess_split, so its encode_samples arguments are already transformed.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "benchmark"
    offenders = []
    for name in ("suite.py", "perturbation.py"):
        source = (root / name).read_text()
        for match in re.finditer(r"\.encode_samples\(\s*([^,\n]*)", source):
            argument = match.group(1).strip()
            if "apply_model_preprocessing" not in argument:
                offenders.append(f"{name}: encode_samples({argument}...)")
    assert not offenders, "unnormalized encoder calls: " + "; ".join(offenders)


def test_load_model_forwards_batch_size_only_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch_size must reach create_model, and stay absent when unset.

    Passing batch_size=None explicitly would shadow each adapter's own
    DEFAULT_BATCH_SIZE fallback (kwargs.get("batch_size", DEFAULT) sees the
    key and returns None instead of the default), so _load_model must omit
    the key entirely rather than pass it as None.
    """
    import types

    from benchmark.runner import _load_model

    calls: list[dict[str, object]] = []

    def fake_create_model(**kwargs: object) -> str:
        calls.append(kwargs)
        return "model"

    stub_module = types.SimpleNamespace(create_model=fake_create_model)
    monkeypatch.setattr(
        "benchmark.runner.importlib.import_module", lambda name: stub_module
    )

    _load_model("bulkrnabert", "checkpoint", "cpu", "vocab.json")
    assert "batch_size" not in calls[-1]

    _load_model("bulkrnabert", "checkpoint", "cpu", "vocab.json", batch_size=2)
    assert calls[-1]["batch_size"] == 2


def test_run_task_dispatch_pops_batch_size_only_for_gene_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """batch_size is meaningless for gene-function's unbatched embedding lookup.

    run_supervised and run_perturbation both batch through encode_samples and
    must receive it; run_gene_function's strict signature has no such
    parameter and would raise TypeError if it were not popped first.
    """
    import benchmark.runner as runner

    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    captured: dict[str, dict[str, object]] = {}

    def record(name: str) -> Callable[..., str]:
        def _fake(**kwargs: object) -> str:
            captured[name] = kwargs
            return "path"

        return _fake

    monkeypatch.setattr(runner, "run_supervised", record("supervised"))
    monkeypatch.setattr(runner, "run_gene_function", record("gene_function"))
    monkeypatch.setattr(
        "benchmark.perturbation.run_perturbation", record("perturbation")
    )

    for family in ("regression", "gene_function", "perturbation"):
        pd.DataFrame(
            [
                {
                    "family": family,
                    "task_id": family,
                    "file": "x.csv.gz",
                    "seeds": "10|11",
                }
            ]
        ).to_csv(data_dir / "tasks.csv", index=False)
        runner.run_task(
            task_id=family,
            model_name="linear",
            checkpoint="",
            data_dir=data_dir,
            results_dir=tmp_path / "results",
            seed=10,
            device="cpu",
            vocab_map=None,
            batch_size=2,
        )

    assert captured["supervised"]["batch_size"] == 2
    assert captured["perturbation"]["batch_size"] == 2
    assert "batch_size" not in captured["gene_function"]


def test_run_task_forwards_tensorboard_to_perturbation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Perturbation logs scalars too, so the flag must survive dispatch."""
    import benchmark.runner as runner

    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    captured: dict[str, dict[str, object]] = {}

    def record(name: str) -> Callable[..., str]:
        def _fake(**kwargs: object) -> str:
            captured[name] = kwargs
            return "path"

        return _fake

    monkeypatch.setattr(runner, "run_gene_function", record("gene_function"))
    monkeypatch.setattr(
        "benchmark.perturbation.run_perturbation", record("perturbation")
    )

    for family in ("perturbation", "gene_function"):
        pd.DataFrame(
            [
                {
                    "family": family,
                    "task_id": family,
                    "file": "x.csv.gz",
                    "seeds": "10|11",
                }
            ]
        ).to_csv(data_dir / "tasks.csv", index=False)
        runner.run_task(
            task_id=family,
            model_name="linear",
            checkpoint="",
            data_dir=data_dir,
            results_dir=tmp_path / "results",
            seed=10,
            device="cpu",
            vocab_map=None,
            tensorboard=False,
        )

    assert captured["perturbation"]["tensorboard"] is False
    assert "tensorboard" not in captured["gene_function"]


def test_run_task_rejects_a_seed_the_task_does_not_ship(tmp_path: Path) -> None:
    """Gene function ships seed 42 only, so the flow default has to say so."""
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "gene_function",
                "task_id": "gene_function_go",
                "file": "x.csv.gz",
                "seeds": "42",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)

    with pytest.raises(ValueError, match="no split for seed 10.*ships seeds 42"):
        run_task(
            task_id="gene_function_go",
            model_name="linear",
            checkpoint="",
            data_dir=data_dir,
            results_dir=tmp_path / "results",
            seed=10,
            device="cpu",
            vocab_map=None,
        )


def test_load_model_rejects_module_names() -> None:
    """Only the public model names are accepted; `eva_rna` is a module, not a model."""
    from benchmark.runner import _load_model

    with pytest.raises(ValueError, match="Unknown model: eva_rna"):
        _load_model("eva_rna", "eva-60m", "cpu", None)
