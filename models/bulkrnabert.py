"""Sparse BulkRNABert adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from models.checkpoints import BUNDLED_DIR

EMBEDDING_LAYER = 4
DEFAULT_BATCH_SIZE = 4
DEFAULT_VOCAB_MAP = BUNDLED_DIR / "bulkrnabert_ncbi_to_position.json"
"""NCBI GeneID to BulkRNABert gene position, used unless --vocab-map overrides it."""


def prepare_sparse_expression(
    expression: np.ndarray,
    gene_ids: list[str],
    ncbi_to_position: dict[str, int],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Select mapped genes in input order; values must already be preprocessed."""
    known = [
        (index, gene, ncbi_to_position[str(gene)])
        for index, gene in enumerate(gene_ids)
        if str(gene) in ncbi_to_position
    ]
    if not known:
        raise ValueError("No input gene maps to the BulkRNABert vocabulary")
    columns = [index for index, _, _ in known]
    positions = [position for _, _, position in known]
    mapped_genes = [str(gene) for _, gene, _ in known]
    return expression[:, columns], positions, mapped_genes


class BulkRNABert:
    """Official BulkRNABert model using sparse selected-gene attention."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        vocab_map: str | Path = DEFAULT_VOCAB_MAP,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        with Path(vocab_map).open() as handle:
            self.ncbi_to_position = {
                str(key): int(value) for key, value in json.load(handle).items()
            }
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            checkpoint, config=config, trust_remote_code=True
        )
        self.device = torch.device(device)
        if batch_size < 1:
            raise ValueError("BulkRNABert batch_size must be positive")
        self.batch_size = batch_size
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _encode(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        with torch.no_grad():
            return self._forward(expression, gene_ids)

    def _forward(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """Run the encoder up to the benchmark layer, keeping the autograd graph."""
        values, positions, mapped_genes = prepare_sparse_expression(
            expression, gene_ids, self.ncbi_to_position
        )
        token_ids = self.tokenizer.batch_encode_plus(values, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        gene_index = torch.as_tensor(positions, device=self.device)
        hidden = self.model.expression_embedding_layer(token_ids)
        if self.model.config.use_gene_embedding:
            gene_embedding = self.model.gene_embedding_layer(gene_index)
            if self.model.config.project_gene_embedding:
                gene_embedding = self.model.fc_gene_embedding(gene_embedding)
            hidden = hidden + gene_embedding
        mask = torch.ones(
            (len(expression), 1, len(positions), len(positions)),
            dtype=torch.bool,
            device=self.device,
        )
        for layer_number, layer in enumerate(self.model.transformer_layers, start=1):
            hidden = layer(hidden, attention_mask=mask)["embeddings"]
            if layer_number == EMBEDDING_LAYER:
                break
        return hidden, mapped_genes

    def encode_genes_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable per-gene tokens for one batch of samples."""
        hidden, _ = self._forward(expression, gene_ids)
        return hidden

    def trainable_last_layer(self) -> list[torch.nn.Parameter]:
        """Unfreeze the transformer layer the benchmark reads its embedding from.

        The perturbation protocol does not freeze the encoder fully: the layer
        producing the embedding trains jointly with the decoder.
        """
        layer = self.model.transformer_layers[EMBEDDING_LAYER - 1]
        parameters = list(layer.parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    def encode_samples(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Average layer-four sparse token embeddings; no CLS token is used."""
        if not len(expression):
            raise ValueError("Cannot encode an empty BulkRNABert sample batch")
        pooled_batches: list[np.ndarray] = []
        for start in range(0, len(expression), self.batch_size):
            with torch.no_grad():
                pooled = self.encode_samples_tensor(
                    expression[start : start + self.batch_size],
                    gene_ids,
                )
            pooled_batches.append(pooled.detach().cpu().numpy())
        return np.concatenate(pooled_batches)

    def encode_samples_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable mean layer-four token embeddings for one batch."""
        hidden, _ = self._forward(expression, gene_ids)
        return hidden.mean(dim=1)

    def encode_genes(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Return layer-four token embeddings, one per requested gene."""
        if not len(expression):
            raise ValueError("Cannot encode an empty BulkRNABert sample batch")
        token_batches: list[np.ndarray] = []
        for start in range(0, len(expression), self.batch_size):
            hidden, mapped_genes = self._encode(
                expression[start : start + self.batch_size],
                gene_ids,
            )
            if hidden.shape[1] != len(mapped_genes):
                raise ValueError(
                    "BulkRNABert returned "
                    f"{hidden.shape[1]} tokens for {len(mapped_genes)} genes"
                )
            token_batches.append(hidden.detach().cpu().numpy())
        return np.concatenate(token_batches)

    def gene_embeddings(self, gene_ids: list[str]) -> np.ndarray:
        """Return unprojected official gene embedding rows."""
        positions = [self.ncbi_to_position[str(gene)] for gene in gene_ids]
        index = torch.as_tensor(positions, device=self.device)
        return self.model.gene_embedding_layer(index).detach().cpu().numpy()

    def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        """Apply the official BulkRNABert transform to released CPM values."""
        return np.log10(1.0 + expression)

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        """Identify genes mapped to official BulkRNABert positions."""
        return np.array(
            [str(gene) in self.ncbi_to_position for gene in gene_ids],
            dtype=bool,
        )


def create_model(
    checkpoint: str | Path,
    device: str = "cpu",
    **kwargs: Any,
) -> BulkRNABert:
    """Create the benchmark adapter."""
    return BulkRNABert(
        checkpoint,
        vocab_map=kwargs.get("vocab_map") or DEFAULT_VOCAB_MAP,
        device=device,
        batch_size=int(kwargs.get("batch_size", DEFAULT_BATCH_SIZE)),
    )
