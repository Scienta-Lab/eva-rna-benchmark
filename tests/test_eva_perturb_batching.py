"""Tests that chunked latent perturbation matches a whole-cohort step."""

from __future__ import annotations

import numpy as np
import torch

from models.eva_rna import EVARNA


class _StubDecoder:
    """Encoder whose gene embeddings depend only on their own sample."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    def __call__(self, **batch: torch.Tensor) -> object:
        values = batch["values"]
        embeddings = torch.stack(
            [values * (index + 1) for index in range(self.dimension)], dim=-1
        )
        return type("Output", (), {"gene_embeddings": embeddings})()


def _adapter(batch_size: int) -> EVARNA:
    adapter = EVARNA.__new__(EVARNA)
    adapter.batch_size = batch_size
    adapter.device = torch.device("cpu")
    adapter.zinb_decoder = None
    adapter.has_expression_decoder = True
    adapter.model = _StubDecoder()
    adapter._tokenize = lambda expression, gene_ids: {
        "values": torch.as_tensor(expression, dtype=torch.float32)
    }
    adapter._decode = lambda latent: latent.sum(-1)
    return adapter


def test_chunked_perturbation_matches_whole_cohort() -> None:
    genes = [str(index) for index in range(6)]
    expression = np.abs(
        np.random.RandomState(0).randn(7, len(genes)).astype(np.float32)
    )
    whole = _adapter(7).gradient_perturb(expression, genes, ["0", "2"], [1, -1])
    for batch_size in (1, 2, 3):
        chunked = _adapter(batch_size).gradient_perturb(
            expression, genes, ["0", "2"], [1, -1]
        )
        assert chunked.shape == whole.shape
        np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def test_perturbation_preserves_library_size() -> None:
    genes = [str(index) for index in range(5)]
    expression = np.abs(
        np.random.RandomState(1).randn(4, len(genes)).astype(np.float32)
    )
    perturbed = _adapter(2).gradient_perturb(expression, genes, ["1"], [1])
    np.testing.assert_allclose(
        perturbed.sum(1), expression.sum(1), rtol=1e-5, atol=1e-5
    )
