"""Tests for EVA expression-decoder detection and ZINB decoding."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from models.eva_rna import (
    EVARNA,
    ZINB_DECODER_FILE,
    _has_expression_decoder,
    _load_zinb_decoder,
)


def test_checkpoint_shipping_decoder_weights_is_usable(tmp_path: Path) -> None:
    save_file(
        {
            "gene_embeddings.weight": torch.zeros(2, 2),
            "expression_decoder.mlp.0.weight": torch.zeros(2, 2),
        },
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )
    assert _has_expression_decoder(tmp_path)


def test_checkpoint_without_decoder_weights_is_detected(tmp_path: Path) -> None:
    save_file(
        {"gene_embeddings.weight": torch.zeros(2, 2)},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )
    assert not _has_expression_decoder(tmp_path)


def _write_zinb(path: Path, hidden: int = 4) -> None:
    save_file(
        {
            "mu_head.weight": torch.ones(1, hidden),
            "mu_head.bias": torch.zeros(1),
            "theta_head.weight": torch.ones(1, hidden),
            "theta_head.bias": torch.zeros(1),
            "pi_head.weight": torch.zeros(1, hidden),
            "pi_head.bias": torch.zeros(1),
        },
        path / ZINB_DECODER_FILE,
        metadata={"format": "pt"},
    )


def test_zinb_decoder_is_absent_when_not_shipped(tmp_path: Path) -> None:
    assert _load_zinb_decoder(tmp_path, torch.device("cpu")) is None


def test_zinb_decoder_loads_every_head(tmp_path: Path) -> None:
    _write_zinb(tmp_path)
    heads = _load_zinb_decoder(tmp_path, torch.device("cpu"))
    assert set(heads) == {
        "mu_head.weight",
        "mu_head.bias",
        "theta_head.weight",
        "theta_head.bias",
        "pi_head.weight",
        "pi_head.bias",
    }


def test_zinb_decode_returns_the_zero_inflated_expected_value(tmp_path: Path) -> None:
    _write_zinb(tmp_path)
    adapter = EVARNA.__new__(EVARNA)
    adapter.zinb_decoder = _load_zinb_decoder(tmp_path, torch.device("cpu"))
    latent = torch.full((2, 3, 4), 0.25)
    decoded = adapter._decode(latent)
    assert decoded.shape == (2, 3)
    expected = (1.0 - torch.sigmoid(torch.tensor(0.0))) * torch.nn.functional.softplus(
        torch.tensor(1.0)
    )
    assert torch.allclose(decoded, expected.expand(2, 3))


def test_a_zinb_checkpoint_counts_as_having_a_decoder(tmp_path: Path) -> None:
    save_file(
        {"gene_embeddings.weight": torch.zeros(2, 2)},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )
    _write_zinb(tmp_path)
    assert not _has_expression_decoder(tmp_path)
    assert _load_zinb_decoder(tmp_path, torch.device("cpu")) is not None
