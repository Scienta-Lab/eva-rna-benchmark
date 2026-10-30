"""Parity tests for public baseline preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import torch

from models.bulkrnabert import prepare_sparse_expression
from models.linear import LinearBaseline
from models.scgpt import (
    ScGPT,
    _compatible_checkpoint_state,
    bin_scgpt_expression,
    prepare_scgpt_batch,
)


class FakeVocab:
    """Minimal scGPT-like vocabulary."""

    values = {"<cls>": 0, "<pad>": 1, "20": 2, "10": 3}

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __getitem__(self, key: str) -> int:
        return self.values[key]


def test_scgpt_keeps_matrix_order_and_adds_zero_cls() -> None:
    tokens, values, padding = prepare_scgpt_batch(
        np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
        ["10", "unknown", "20"],
        FakeVocab(),
    )
    np.testing.assert_array_equal(tokens.numpy(), [[0, 3, 2]])
    np.testing.assert_array_equal(values.numpy(), [[0.0, 2.0, 8.0]])
    assert not padding.any()


def test_scgpt_bins_nonzero_expression_per_sample() -> None:
    binned = bin_scgpt_expression(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 10.0, 20.0, 30.0],
            ],
            dtype=np.float32,
        ),
        n_bins=4,
    )
    np.testing.assert_array_equal(
        binned,
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
    )


def test_scgpt_rejects_negative_expression_before_binning() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        bin_scgpt_expression(
            np.array([[0.0, -1.0]], dtype=np.float32),
            n_bins=51,
        )


def test_scgpt_maps_ncbi_ids_to_public_vocab_tokens() -> None:
    tokens, values, padding = prepare_scgpt_batch(
        np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
        ["1", "unknown", "2"],
        FakeVocab(),
        {"1": "10", "2": "20"},
        n_bins=51,
    )
    np.testing.assert_array_equal(tokens.numpy(), [[0, 3, 2]])
    np.testing.assert_array_equal(values.numpy(), [[0.0, 1.0, 50.0]])
    assert not padding.any()


def test_scgpt_converts_public_flash_attention_checkpoint() -> None:
    target = {
        "encoder.embedding.weight": torch.zeros(2, 2),
        "value_encoder.linear1.weight": torch.zeros(2, 1),
        "transformer_encoder.layers.0.self_attn.in_proj_weight": torch.zeros(6, 2),
        "cls_decoder.out_layer.weight": torch.zeros(1, 2),
    }
    checkpoint = {
        "encoder.embedding.weight": torch.ones(2, 2),
        "value_encoder.linear1.weight": torch.ones(2, 1),
        "transformer_encoder.layers.0.self_attn.Wqkv.weight": torch.ones(6, 2),
        "mvc_decoder.fc.weight": torch.ones(1, 2),
    }
    compatible = _compatible_checkpoint_state(
        target,
        checkpoint,
        uses_fast_transformer=False,
    )
    assert set(compatible) == {
        "encoder.embedding.weight",
        "value_encoder.linear1.weight",
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
    }


def test_scgpt_rejects_public_checkpoint_without_backbone_weights() -> None:
    target = {
        "encoder.embedding.weight": torch.zeros(2, 2),
        "transformer_encoder.layers.0.linear1.weight": torch.zeros(2, 2),
    }
    with pytest.raises(ValueError, match="transformer_encoder"):
        _compatible_checkpoint_state(
            target,
            {"encoder.embedding.weight": torch.ones(2, 2)},
            uses_fast_transformer=False,
        )


def test_scgpt_encodes_samples_in_small_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScGPT.__new__(ScGPT)
    model.batch_size = 2
    batch_sizes: list[int] = []

    def fake_encode(
        expression: np.ndarray,
        gene_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sizes.append(len(expression))
        encoded = torch.as_tensor(expression[:, :1, None], dtype=torch.float32)
        padding = torch.zeros(encoded.shape[:2], dtype=torch.bool)
        return encoded, padding

    monkeypatch.setattr(model, "_forward", fake_encode)
    output = model.encode_samples(
        np.arange(10, dtype=np.float32).reshape(5, 2),
        ["10", "20"],
    )
    assert batch_sizes == [2, 2, 1]
    np.testing.assert_array_equal(output[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0])


def test_bulkrnabert_keeps_matrix_order_without_transforming() -> None:
    """Selection happens upstream now; this step must only reorder columns."""
    values, positions, genes = prepare_sparse_expression(
        np.array([[9.0, 99.0, 999.0]]),
        ["10", "unknown", "20"],
        {"20": 8, "10": 3},
    )
    np.testing.assert_allclose(values, [[9.0, 999.0]])
    assert positions == [3, 8]
    assert genes == ["10", "20"]


def test_each_encoder_applies_its_published_transform_to_cpm() -> None:
    """Datasets ship CPM, so every adapter must normalize it itself."""
    from models.bulkrnabert import BulkRNABert
    from models.eva_rna import EVARNA
    from models.linear import LinearBaseline
    from models.scgpt import ScGPT

    cpm = np.array([[0.0, 9.0, 999.0]], dtype=np.float32)

    bulk = BulkRNABert.__new__(BulkRNABert)
    np.testing.assert_allclose(bulk.preprocess_expression(cpm), [[0.0, 1.0, 3.0]])

    eva = EVARNA.__new__(EVARNA)
    np.testing.assert_allclose(
        eva.preprocess_expression(cpm), np.log2(1.0 + cpm), rtol=1e-6
    )

    linear = LinearBaseline.__new__(LinearBaseline)
    np.testing.assert_allclose(
        linear.preprocess_expression(cpm), np.log2(1.0 + cpm), rtol=1e-6
    )

    scgpt = ScGPT.__new__(ScGPT)
    scgpt.n_bins = 3
    binned = scgpt.preprocess_expression(cpm)
    assert set(np.unique(binned)) <= {0.0, 1.0, 2.0}


def test_linear_zsp_applies_interaction_and_renormalizes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "interaction_matrix.pt"
    torch.save(
        {
            "gene_ids": [1, 2],
            "left_member": torch.eye(2),
            "low_rank": None,
        },
        checkpoint,
    )
    model = LinearBaseline(str(checkpoint))

    perturbed = model.gradient_perturb(
        np.array([[1.0, 1.0]], dtype=np.float32),
        ["1", "2"],
        ["1"],
        [1],
    )

    np.testing.assert_allclose(perturbed, [[4 / 3, 2 / 3]], rtol=1e-6)
    np.testing.assert_allclose(perturbed.sum(axis=1), [2.0])


class _StubEvaOutput:
    def __init__(self, cls_embedding: torch.Tensor) -> None:
        self.cls_embedding = cls_embedding
        self.gene_embeddings = cls_embedding


class _StubEvaModel:
    """Records the batch sizes it was called with."""

    def __init__(self) -> None:
        self.seen_rows: list[int] = []

    def __call__(self, values: torch.Tensor) -> _StubEvaOutput:
        self.seen_rows.append(len(values))
        return _StubEvaOutput(values.clone())


def test_eva_encode_samples_batches_and_preserves_row_order() -> None:
    """Batched encoding must equal one pass, row for row."""
    from models.eva_rna import EVARNA

    adapter = EVARNA.__new__(EVARNA)
    adapter.batch_size = 2
    adapter.model = _StubEvaModel()
    adapter._tokenize = lambda expression, gene_ids: {
        "values": torch.as_tensor(expression, dtype=torch.float32)
    }

    expression = np.arange(10, dtype=np.float32).reshape(5, 2)
    embeddings = adapter.encode_samples(expression, ["1", "2"])

    np.testing.assert_allclose(embeddings, expression)
    assert adapter.model.seen_rows == [2, 2, 1]


def test_eva_rejects_an_empty_batch() -> None:
    from models.eva_rna import EVARNA

    adapter = EVARNA.__new__(EVARNA)
    adapter.batch_size = 2
    with pytest.raises(ValueError, match="empty EVA sample batch"):
        adapter.encode_samples(np.zeros((0, 3), dtype=np.float32), ["1"])


def test_linear_fit_does_not_depend_on_the_input_dtype() -> None:
    rng = np.random.default_rng(0)
    features = np.log2(1.0 + rng.gamma(0.5, 200.0, size=(40, 3000)))
    targets = rng.random(40) < 0.4
    single = LinearBaseline.fit(features.astype(np.float32), targets, "classification")
    double = LinearBaseline.fit(features, targets, "classification")
    np.testing.assert_allclose(single.coef_, double.coef_, rtol=1e-4, atol=1e-9)


class _EchoScgptModel:
    """Encoder echoing input values into a latent channel the decoder reads."""

    def _encode(
        self, tokens: torch.Tensor, values: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([values, torch.ones_like(values)], dim=-1)

    def decoder(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pred": latent[..., 0] + 0.1 * latent[..., 1]}


def _echo_scgpt() -> ScGPT:
    model = ScGPT.__new__(ScGPT)
    model.vocab = FakeVocab()
    model.gene_token_map = None
    model.n_bins = 3
    model.inputs_prebinned = False
    model.device = torch.device("cpu")
    model.model = _EchoScgptModel()
    return model


def test_scgpt_zsp_perturbation_stays_in_the_binned_space() -> None:
    model = _echo_scgpt()
    log_values = np.array([[0.0, 3.0, 9.0], [5.0, 1.0, 0.0]], dtype=np.float32)
    genes = ["10", "99", "20"]

    perturbed = model.gradient_perturb(log_values, genes, ["20"], [1])
    baseline = model.perturbation_space(log_values, genes)

    assert perturbed.shape == baseline.shape == (2, 2)
    assert baseline.max() <= model.n_bins - 1
    np.testing.assert_allclose(perturbed.sum(1), baseline.sum(1), rtol=1e-6)


def test_scgpt_perturbation_task_reads_unbinned_log_values() -> None:
    model = _echo_scgpt()
    model.inputs_prebinned = True
    log_values = np.array([[0.5, 3.0, 7.0]], dtype=np.float32)

    tokens = model.encode_genes_tensor(log_values, ["10", "20"])

    np.testing.assert_allclose(tokens[..., 0].numpy(), log_values[:, :2])
