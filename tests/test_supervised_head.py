"""Tests for the supervised heads, frozen and last-layer fine-tuned."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from benchmark import runner, supervised_head
from benchmark.supervised_head import (
    FROZEN,
    LAST_LAYER,
    TREATMENT_OUTCOME_TASKS,
    SupervisedHead,
    _learning_rate_factor,
    fit_head,
    predict_head,
    protocol_for,
)


class FakeEncoder:
    """Tiny encoder: one linear layer per gene value, averaged over genes."""

    def __init__(self, unknown: set[str] | None = None) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(1, 4), nn.Dropout(0.0))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.unknown = unknown or set()
        self.calls: list[tuple[int, bool, bool]] = []
        self.seen: set[str] = set()

    def encode_samples_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        self.calls.append((len(gene_ids), torch.is_grad_enabled(), self.model.training))
        self.seen.update(gene_ids)
        values = torch.as_tensor(np.asarray(expression), dtype=torch.float32)
        return self.model(values[..., None]).mean(dim=1)

    def encode_samples(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        return self.encode_samples_tensor(expression, gene_ids).detach().numpy()

    def trainable_last_layer(self) -> list[nn.Parameter]:
        parameters = list(self.model[0].parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        return np.array([gene not in self.unknown for gene in gene_ids])


def _data(samples: int = 24, genes: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.array([True, False] * (samples // 2))
    values = rng.random((samples, genes)) + labels[:, None]
    return values.astype(np.float32), labels


def test_frozen_protocol_matches_the_paper_runs() -> None:
    assert FROZEN.hidden_dim == 128
    assert FROZEN.learning_rate == 5e-4
    assert FROZEN.weight_decay == 0.1
    assert FROZEN.max_epochs == 200
    assert FROZEN.patience == 30
    assert FROZEN.mixup_alpha == 0.2
    assert not FROZEN.finetune_last_layer
    assert FROZEN.warmup_cosine


def test_last_layer_protocol_matches_the_paper_run() -> None:
    assert LAST_LAYER.hidden_dim == 16
    assert LAST_LAYER.learning_rate == 1e-5
    assert LAST_LAYER.weight_decay == 0.4
    assert LAST_LAYER.max_epochs == 3_000
    assert LAST_LAYER.patience == 200
    assert LAST_LAYER.mixup_alpha == 0.4
    assert LAST_LAYER.finetune_last_layer
    assert not LAST_LAYER.warmup_cosine


@pytest.mark.parametrize("protocol", [FROZEN, LAST_LAYER])
def test_shared_settings(protocol: supervised_head.SupervisedHeadProtocol) -> None:
    assert protocol.batch_size == 16
    assert protocol.genes_per_step == 600
    assert protocol.gradient_clip == 1.0


def test_only_treatment_outcome_tasks_fine_tune() -> None:
    tasks = pd.read_csv(Path(__file__).parents[1] / "datasets" / "tasks.csv")
    classification = set(tasks.loc[tasks["family"] == "classification", "task_id"])
    assert TREATMENT_OUTCOME_TASKS <= classification
    assert len(TREATMENT_OUTCOME_TASKS) == 6
    for task_id in tasks.loc[tasks["family"].isin(["classification", "regression"])][
        "task_id"
    ]:
        expected = LAST_LAYER if task_id in TREATMENT_OUTCOME_TASKS else FROZEN
        assert protocol_for(task_id) is expected
    assert protocol_for("ra_blood_endotype") is FROZEN


def test_learning_rate_warms_up_from_zero_then_decays_to_a_floor() -> None:
    floor = FROZEN.min_learning_rate / FROZEN.learning_rate
    assert _learning_rate_factor(0, FROZEN) == 0.0
    assert _learning_rate_factor(5, FROZEN) == pytest.approx(1.0)
    assert _learning_rate_factor(80, FROZEN) == pytest.approx(0.5 + 0.5 * floor)
    assert _learning_rate_factor(155, FROZEN) == pytest.approx(floor)
    assert _learning_rate_factor(190, FROZEN) == pytest.approx(floor)


def test_last_layer_learning_rate_is_constant() -> None:
    assert {_learning_rate_factor(epoch, LAST_LAYER) for epoch in (0, 3, 500)} == {1.0}


def test_head_is_linear_gelu_linear() -> None:
    kinds = [type(layer) for layer in SupervisedHead(4, 8, 2).mlp]
    assert kinds == [nn.Linear, nn.GELU, nn.Linear]


@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("base", [FROZEN, LAST_LAYER])
def test_training_samples_genes_and_prediction_uses_all(
    task_type: str, base: supervised_head.SupervisedHeadProtocol
) -> None:
    x, labels = _data()
    y = x[:, 0] if task_type == "regression" else labels
    encoder = FakeEncoder()
    protocol = replace(base, max_epochs=3, genes_per_step=3, batch_size=8)
    genes = [str(index) for index in range(x.shape[1])]

    fitted = fit_head(
        encoder,
        (x[:16], y[:16]),
        (x[16:], y[16:]),
        genes,
        task_type=task_type,
        seed=0,
        protocol=protocol,
    )
    encoder.calls.clear()
    prediction = predict_head(encoder, fitted, x[16:], genes, task_type=task_type)

    assert {width for width, _, _ in encoder.calls} == {len(genes)}
    assert not any(grad for _, grad, _ in encoder.calls)
    assert encoder.batch_size == 2
    if task_type == "classification":
        assert list(fitted.classes) == [False, True]
        assert prediction.shape == (8, 2)
        np.testing.assert_allclose(prediction.sum(axis=1), 1.0, rtol=1e-5)
    else:
        assert fitted.classes is None
        assert prediction.shape == (8,)


@pytest.mark.parametrize("base", [FROZEN, LAST_LAYER])
def test_training_and_validation_read_random_gene_subsets(
    base: supervised_head.SupervisedHeadProtocol,
) -> None:
    x, y = _data()
    encoder = FakeEncoder()
    genes = [str(index) for index in range(x.shape[1])]
    fit_head(
        encoder,
        (x[:16], y[:16]),
        (x[16:], y[16:]),
        genes,
        task_type="classification",
        seed=0,
        protocol=replace(base, max_epochs=2, genes_per_step=3, batch_size=8),
    )
    widths = [width for width, _, _ in encoder.calls]
    assert widths[0] == len(genes)
    assert set(widths[1:]) == {3}


def test_frozen_encoder_never_trains() -> None:
    encoder = FakeEncoder()
    before = encoder.model[0].weight.detach().clone()
    x, y = _data()
    fit_head(
        encoder,
        (x[:16], y[:16]),
        (x[16:], y[16:]),
        [str(gene) for gene in range(x.shape[1])],
        task_type="classification",
        seed=0,
        protocol=replace(FROZEN, max_epochs=3, learning_rate=1e-2, genes_per_step=3),
    )
    assert torch.equal(before, encoder.model[0].weight)
    assert not any(grad for _, grad, _ in encoder.calls)


def test_finetuning_updates_the_last_layer_with_the_encoder_in_eval_mode() -> None:
    encoder = FakeEncoder()
    encoder.model[1].p = 0.9
    before = encoder.model[0].weight.detach().clone()
    x, y = _data()
    fit_head(
        encoder,
        (x[:16], y[:16]),
        (x[16:], y[16:]),
        [str(gene) for gene in range(x.shape[1])],
        task_type="classification",
        seed=0,
        protocol=replace(
            LAST_LAYER, max_epochs=3, learning_rate=1e-2, genes_per_step=3
        ),
    )
    assert not torch.equal(before, encoder.model[0].weight)
    assert not any(training for _, _, training in encoder.calls)
    assert any(grad for _, grad, _ in encoder.calls)
    assert not encoder.model.training


def test_warmup_epoch_leaves_the_head_unchanged() -> None:
    x, _ = _data(samples=20, genes=4)
    protocol = replace(FROZEN, max_epochs=1, genes_per_step=4)
    torch.manual_seed(0)
    initial = SupervisedHead(4, protocol.hidden_dim, 1).state_dict()

    fitted = fit_head(
        FakeEncoder(),
        (x[:16], x[:16, 0]),
        (x[16:], x[16:, 0]),
        ["1", "2", "3", "4"],
        task_type="regression",
        seed=0,
        protocol=protocol,
    )

    for name, value in fitted.head.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_early_stopping_ends_training_after_the_patience(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x, y = _data()
    fit_head(
        FakeEncoder(),
        (x[:16], y[:16]),
        (x[16:], y[16:]),
        [str(gene) for gene in range(x.shape[1])],
        task_type="classification",
        seed=0,
        protocol=replace(LAST_LAYER, max_epochs=50, patience=2, learning_rate=0.0),
    )
    assert "stopped after 3 epochs" in capsys.readouterr().out


def _write_task(data_dir: Path, task_id: str, family: str) -> None:
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "family": family,
                "task_id": task_id,
                "dataset_id": "toy",
                "file": "toy.csv.gz",
                "target": "score",
                "seeds": "10",
            }
        ]
    ).to_csv(data_dir / "tasks.csv", index=False)
    x, labels = _data()
    frame = pd.DataFrame(x[:, :3] * 10, columns=["gene__1", "gene__2", "gene__3"])
    frame.insert(0, "sample_id", [str(index) for index in range(len(x))])
    frame["target__score"] = labels if family == "classification" else x[:, 0]
    frame[f"split__{task_id}__seed_10"] = ["train"] * 16 + ["val"] * 4 + ["test"] * 4
    frame.to_csv(data_dir / "toy.csv.gz", index=False)


@pytest.mark.parametrize(
    ("task_id", "family", "finetuned"),
    [
        ("toy", "regression", False),
        ("ibd_digestive_system_infliximab_endo_remission", "classification", True),
    ],
)
def test_run_task_drops_unknown_genes_and_picks_the_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    family: str,
    finetuned: bool,
) -> None:
    data_dir = tmp_path / "datasets"
    _write_task(data_dir, task_id, family)
    encoder = FakeEncoder(unknown={"3"})
    before = encoder.model[0].weight.detach().clone()
    monkeypatch.setattr(runner, "_load_model", lambda *_, **__: encoder)
    monkeypatch.setattr(
        supervised_head,
        "protocol_for",
        lambda task: replace(
            protocol_for(task), max_epochs=2, learning_rate=1e-2, genes_per_step=2
        ),
    )

    destination = runner.run_task(
        task_id=task_id,
        model_name="eva",
        checkpoint="",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        seed=10,
        device="cpu",
        vocab_map=None,
    )

    assert encoder.seen == {"1", "2"}
    assert not pd.read_csv(destination).empty
    assert torch.equal(before, encoder.model[0].weight) is not finetuned
