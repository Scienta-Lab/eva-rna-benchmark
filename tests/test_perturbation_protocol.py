"""Tests for the perturbation decoder and its fine-tuning loop."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from benchmark import perturbation
from benchmark.perturbation import (
    PROTOCOLS,
    PerturbationDecoder,
    PerturbationProtocol,
    _decoder_input,
    _encoder_input,
    _finetune_decoder,
    _mixup,
)
from benchmark.runner import run_task


class FakeEncoder:
    """Tiny per-gene encoder: one linear layer applied to each gene's value."""

    def __init__(self, unknown: set[str] | None = None) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(1, 4), nn.Dropout(0.0))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = torch.device("cpu")
        self.unknown = unknown or set()

    def encode_genes_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        values = torch.as_tensor(np.asarray(expression), dtype=torch.float32)
        return self.model(values[..., None])

    def encode_cls_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        return self.encode_genes_tensor(expression, gene_ids).mean(dim=1)

    def input_gene_embeddings(self, gene_ids: list[str]) -> torch.Tensor:
        return torch.arange(len(gene_ids) * 4, dtype=torch.float32).reshape(-1, 4)

    def trainable_last_layer(self) -> list[nn.Parameter]:
        parameters = list(self.model[0].parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        return np.array([gene not in self.unknown for gene in gene_ids])


def test_protocols_match_the_paper_runs() -> None:
    assert PROTOCOLS["eva"].decode_mode == "cls"
    assert PROTOCOLS["scgpt"].decode_mode == "per_gene"
    assert PROTOCOLS["bulkrnabert"].decode_mode == "per_gene"
    assert PROTOCOLS["scgpt"].encoder_input == "log2_cpm"
    assert PROTOCOLS["scgpt"].encoder_autocast == "float16"
    for name in ("eva", "bulkrnabert"):
        assert PROTOCOLS[name].encoder_input == "model"
        assert PROTOCOLS[name].encoder_autocast is None
    assert PROTOCOLS["eva"].gene_space == "native"
    assert PROTOCOLS["scgpt"].gene_space == "ortholog"
    assert PROTOCOLS["bulkrnabert"].gene_space == "ortholog"
    assert PROTOCOLS["eva"].out_of_vocabulary == "fallback"
    assert PROTOCOLS["scgpt"].out_of_vocabulary == "drop"
    assert PROTOCOLS["bulkrnabert"].out_of_vocabulary == "drop"
    for protocol in PROTOCOLS.values():
        assert protocol.hidden_dims == (2048, 1024)
        assert protocol.learning_rate == 1e-4
        assert protocol.weight_decay == 0.1
        assert protocol.batch_size == 16
        assert protocol.patience == 20
        assert protocol.mixup_alpha == 0.5
        assert protocol.gradient_clip == 1.0


@pytest.mark.parametrize("decode_mode", ["cls", "per_gene"])
@pytest.mark.parametrize("genes", [3, 5])
def test_decoder_reads_panels_of_any_width(decode_mode: str, genes: int) -> None:
    encoder = FakeEncoder()
    gene_ids = [str(index) for index in range(genes)]
    features = _decoder_input(encoder, np.ones((2, genes)), gene_ids, decode_mode)
    width = 8 if decode_mode == "cls" else 4
    decoder = PerturbationDecoder(width, (16, 8))
    assert decoder(features).shape == (2, genes)


def test_decoder_layer_order() -> None:
    layers = list(PerturbationDecoder(8, (16, 4)).mlp)
    kinds = [type(layer) for layer in layers]
    assert kinds == [
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Linear,
    ]


def test_mixup_uses_one_weight_and_permutation_for_both_arrays() -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    baseline, target = _mixup(values, values.copy(), 0.5, np.random.default_rng(0))
    np.testing.assert_allclose(baseline, target)
    assert not np.allclose(baseline, values)


def test_finetuning_trains_the_last_layer_with_the_encoder_in_eval_mode() -> None:
    encoder = FakeEncoder()
    encoder.model[1].p = 0.9
    modes: list[bool] = []
    original = encoder.encode_genes_tensor

    def record(expression: np.ndarray, gene_ids: list[str]) -> torch.Tensor:
        modes.append(encoder.model.training)
        return original(expression, gene_ids)

    encoder.encode_genes_tensor = record
    before = encoder.model[0].weight.detach().clone()
    rng = np.random.default_rng(0)
    baseline = rng.random((8, 3)).astype(np.float32)
    splits = {"train": (baseline, baseline + 1.0), "val": (baseline, baseline + 1.0)}
    protocol = PerturbationProtocol(
        decode_mode="per_gene", hidden_dims=(8,), batch_size=4, max_epochs=3
    )
    decoder = _finetune_decoder(encoder, protocol, splits, ["1", "2", "3"], seed=0)
    assert not torch.equal(before, encoder.model[0].weight)
    assert modes and not any(modes)
    assert not decoder.training


def test_out_of_vocabulary_test_genes_are_filled_and_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "perturbation",
                "task_id": "toy",
                "dataset_id": "toy",
                "file": "toy.csv.gz",
                "target": "expression",
                "seeds": "10",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    genes = ["1", "2", "3"]
    pd.DataFrame(
        {
            "task_id": ["toy"] * 6,
            "seed": [10] * 6,
            "scope": ["train"] * 3 + ["test"] * 3,
            "gene_id": genes * 2,
            "rank": [0, 1, 2] * 2,
        }
    ).to_csv(data_dir / "perturbation_feature_sets.csv.gz", index=False)
    rng = np.random.default_rng(0)
    rows = 10
    frame: dict[str, Any] = {"sample_id": [f"s{index}" for index in range(rows)]}
    for gene in genes:
        baseline = rng.random(rows) * 10
        frame[f"baseline__gene__{gene}"] = baseline
        frame[f"target__gene__{gene}"] = baseline * 2
    frame["split__toy__seed_10"] = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    pd.DataFrame(frame).to_csv(data_dir / "toy.csv.gz", index=False)
    monkeypatch.setattr(
        perturbation, "_load_model", lambda *_, **__: FakeEncoder(unknown={"3"})
    )
    monkeypatch.setitem(
        PROTOCOLS,
        "scgpt",
        PerturbationProtocol(
            decode_mode="per_gene",
            hidden_dims=(8,),
            max_epochs=2,
            out_of_vocabulary="fallback",
        ),
    )

    destination = run_task(
        task_id="toy",
        model_name="scgpt",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=10,
        device="cpu",
        vocab_map=None,
    )

    assert "1 out-of-vocabulary genes predicted" in capsys.readouterr().out
    assert pd.read_csv(destination)["metric"].eq("pearson").any()


class BinningEncoder(FakeEncoder):
    """Fake encoder whose own transform would bin its input."""

    def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        return np.digitize(expression, [1.0, 10.0]).astype(np.float32)


def test_log2_cpm_input_bypasses_the_encoder_transform() -> None:
    log_values = np.array([[0.5, 3.0, 7.0]], dtype=np.float32)
    raw = PerturbationProtocol(decode_mode="per_gene", encoder_input="log2_cpm")
    np.testing.assert_array_equal(
        _encoder_input(BinningEncoder(), log_values, raw), log_values
    )


def test_model_input_applies_the_encoder_transform_to_cpm() -> None:
    log_values = np.log2(1.0 + np.array([[0.5, 3.0, 20.0]], dtype=np.float32))
    protocol = PerturbationProtocol(decode_mode="per_gene")
    np.testing.assert_array_equal(
        _encoder_input(BinningEncoder(), log_values, protocol), [[0, 1, 2]]
    )


class RecordingEncoder(FakeEncoder):
    """Fake encoder that records the gene IDs it is asked to encode."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def encode_cls_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        self.seen.update(gene_ids)
        return super().encode_cls_tensor(expression, gene_ids)


def _write_cross_species_task(data_dir: Path, with_native: bool) -> None:
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": "perturbation",
                "task_id": "xs",
                "dataset_id": "xs",
                "file": "xs.csv.gz",
                "target": "expression",
                "seeds": "10",
                "native_file": "xs_native.csv.gz" if with_native else "",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    pd.DataFrame(
        {
            "task_id": ["xs"] * 4,
            "seed": [10] * 4,
            "scope": ["train", "train", "test", "test"],
            "gene_id": ["1", "2", "1", "2"],
            "rank": [0, 1, 0, 1],
        }
    ).to_csv(data_dir / "perturbation_feature_sets.csv.gz", index=False)
    pd.DataFrame(
        {
            "task_id": ["xs"] * 2,
            "seed": [10] * 2,
            "scope": ["train"] * 2,
            "gene_id": ["101", "102"],
            "rank": [0, 1],
        }
    ).to_csv(data_dir / "perturbation_native_feature_sets.csv.gz", index=False)
    rng = np.random.default_rng(0)
    split = ["train"] * 6 + ["val"] * 2 + ["test"] * 3
    main: dict[str, Any] = {"sample_id": list(range(11))}
    native: dict[str, Any] = {"sample_id": list(range(8))}
    for gene in ["1", "2"]:
        base = rng.random(11) * 10
        main[f"baseline__gene__{gene}"], main[f"target__gene__{gene}"] = base, base * 2
    for gene in ["101", "102"]:
        base = rng.random(8) * 10
        native[f"baseline__gene__{gene}"] = base
        native[f"target__gene__{gene}"] = base * 2
    main["split__xs__seed_10"] = split
    native["split__xs__seed_10"] = split[:8]
    pd.DataFrame(main).to_csv(data_dir / "xs.csv.gz", index=False)
    pd.DataFrame(native).to_csv(data_dir / "xs_native.csv.gz", index=False)


@pytest.mark.parametrize("with_native", [True, False])
def test_native_gene_space_trains_on_the_native_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_native: bool
) -> None:
    data_dir = tmp_path / "datasets"
    _write_cross_species_task(data_dir, with_native)
    encoder = RecordingEncoder()
    monkeypatch.setattr(perturbation, "_load_model", lambda *_, **__: encoder)
    monkeypatch.setitem(
        PROTOCOLS,
        "eva",
        PerturbationProtocol(
            decode_mode="cls", hidden_dims=(8,), max_epochs=2, gene_space="native"
        ),
    )

    run_task(
        task_id="xs",
        model_name="eva",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=10,
        device="cpu",
        vocab_map=None,
    )

    if with_native:
        assert encoder.seen == {"101", "102", "1", "2"}
    else:
        assert encoder.seen == {"1", "2"}


def test_dropped_out_of_vocabulary_genes_leave_the_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    data_dir = tmp_path / "datasets"
    _write_cross_species_task(data_dir, with_native=False)
    monkeypatch.setattr(
        perturbation, "_load_model", lambda *_, **__: FakeEncoder(unknown={"2"})
    )
    seen: dict[str, tuple[int, ...]] = {}
    original = perturbation.perturbation_metrics

    def record(
        observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray
    ) -> dict[str, float]:
        seen["shape"] = observed.shape
        return original(observed, predicted, baseline)

    monkeypatch.setattr(perturbation, "perturbation_metrics", record)
    monkeypatch.setitem(
        PROTOCOLS,
        "scgpt",
        PerturbationProtocol(
            decode_mode="per_gene",
            hidden_dims=(8,),
            max_epochs=2,
            out_of_vocabulary="drop",
        ),
    )

    run_task(
        task_id="xs",
        model_name="scgpt",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=10,
        device="cpu",
        vocab_map=None,
    )

    assert seen["shape"] == (3, 1)
    assert "out-of-vocabulary" not in capsys.readouterr().out


def test_only_encoders_with_their_own_precision_open_an_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EVA and BulkRNABert must inherit the bf16 prediction context of _predict."""
    opened: list[torch.dtype] = []
    real_autocast = torch.autocast

    def record(*args: Any, **kwargs: Any) -> Any:
        opened.append(kwargs["dtype"])
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(perturbation.torch, "autocast", record)
    gene_ids = ["1", "2"]
    for name in ("eva", "bulkrnabert"):
        perturbation._features(
            FakeEncoder(), np.ones((1, 2)), gene_ids, PROTOCOLS[name]
        )
    assert opened == []
    perturbation._features(FakeEncoder(), np.ones((1, 2)), gene_ids, PROTOCOLS["scgpt"])
    assert opened == [torch.float16]
