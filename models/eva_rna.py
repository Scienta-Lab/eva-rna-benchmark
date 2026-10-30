"""EVA encoder and optional gradient decoder adapter."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

DEFAULT_BATCH_SIZE = 2
ZINB_DECODER_FILE = "expression_decoder_zinb.safetensors"
PREPROJECTION_FILE = "gene_embeddings_preprojection.safetensors"


def _has_expression_decoder(checkpoint: str | Path) -> bool:
    """Report whether the checkpoint actually ships expression-decoder weights.

    Transformers silently random-initializes absent weights, so a checkpoint
    trained with a decoder the release format does not define would still load
    and would still decode -- from noise.
    """
    weights = Path(checkpoint) / "model.safetensors"
    if not weights.is_file():
        return True
    with safe_open(weights, "pt") as handle:
        return any(key.startswith("expression_decoder.") for key in handle.keys())


def _load_zinb_decoder(
    checkpoint: str | Path, device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Load a zero-inflated negative-binomial decoder shipped beside a checkpoint.

    The released Transformers interface only implements the deterministic MLP
    decoder, so a checkpoint trained with a ZINB head carries it in a separate
    file rather than in `model.safetensors`.
    """
    weights = Path(checkpoint) / ZINB_DECODER_FILE
    if not weights.is_file():
        return None
    with safe_open(weights, "pt") as handle:
        return {key: handle.get_tensor(key).to(device) for key in handle.keys()}


def _load_preprojection(checkpoint: str | Path) -> torch.Tensor | None:
    """Load the pre-projection gene embedding table, when the checkpoint has one."""
    weights = Path(checkpoint) / PREPROJECTION_FILE
    if not weights.is_file():
        return None
    with safe_open(weights, "pt") as handle:
        return handle.get_tensor("gene_embeddings")


class EVARNA:
    """Load an EVA checkpoint through its Transformers release interface."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("EVA batch_size must be positive")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(
            self.device
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.checkpoint = Path(checkpoint)
        self.zinb_decoder = _load_zinb_decoder(checkpoint, self.device)
        self.preprojection_embeddings = _load_preprojection(checkpoint)
        self.has_expression_decoder = (
            self.zinb_decoder is not None or _has_expression_decoder(checkpoint)
        )

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode gene embeddings to expected expression.

        A ZINB head predicts a distribution rather than a value, so its expected
        expression is `(1 - pi) * mu`, which is the quantity correlated against
        observed expression.
        """
        if self.zinb_decoder is None:
            return self.model.decode(latent)
        heads = self.zinb_decoder
        mu = torch.nn.functional.softplus(
            torch.nn.functional.linear(
                latent, heads["mu_head.weight"], heads["mu_head.bias"]
            )
        )
        pi = torch.sigmoid(
            torch.nn.functional.linear(
                latent, heads["pi_head.weight"], heads["pi_head.bias"]
            )
        )
        return ((1.0 - pi) * mu).squeeze(-1)

    def _tokenize(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> dict[str, torch.Tensor]:
        genes = [gene_ids] * len(expression)
        values = expression.tolist()
        batch = self.tokenizer(
            genes,
            values,
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

    def _forward_in_batches(
        self,
        expression: np.ndarray,
        gene_ids: list[str],
        field: str,
    ) -> np.ndarray:
        """Run the encoder in batches so attention memory stays bounded."""
        if not len(expression):
            raise ValueError("Cannot encode an empty EVA sample batch")
        outputs: list[np.ndarray] = []
        for start in range(0, len(expression), self.batch_size):
            chunk = expression[start : start + self.batch_size]
            with torch.no_grad():
                output = self.model(**self._tokenize(chunk, gene_ids))
            outputs.append(getattr(output, field).detach().cpu().numpy())
        return np.concatenate(outputs)

    def encode_samples(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Return frozen CLS sample embeddings."""
        return self._forward_in_batches(expression, gene_ids, "cls_embedding")

    def encode_genes(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Return contextualized gene embeddings."""
        return self._forward_in_batches(expression, gene_ids, "gene_embeddings")

    def encode_genes_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable contextualized gene tokens for one batch."""
        return self.model(**self._tokenize(expression, gene_ids)).gene_embeddings

    def encode_cls_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable CLS embeddings for one batch."""
        return self.model(**self._tokenize(expression, gene_ids)).cls_embedding

    def encode_samples_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable sample embeddings, pooled as encode_samples does."""
        return self.encode_cls_tensor(expression, gene_ids)

    def input_gene_embeddings(self, gene_ids: list[str]) -> torch.Tensor:
        """Return the input embedding of each gene, before any transformer layer."""
        token_ids = self.tokenizer.convert_tokens_to_ids(gene_ids)
        return self.model.gene_embeddings(
            torch.as_tensor(token_ids, device=self.device)
        )

    def trainable_last_layer(self) -> list[torch.nn.Parameter]:
        """Unfreeze the final transformer layer for last-layer fine-tuning.

        The published perturbation protocol fine-tunes the last layer of each
        encoder so the decoder has more freedom in expressing the perturbed
        profile. EVA pools its embedding after the whole stack, so the layer
        producing that embedding is the final one.
        """
        parameters = list(self.model.layers[-1].parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    def gene_embeddings(self, gene_ids: list[str]) -> np.ndarray:
        """Return static gene embeddings in requested NCBI GeneID order.

        These are the pre-projection embeddings when the checkpoint ships them,
        and the projected table inside `model.safetensors` otherwise.
        """
        token_ids = self.tokenizer.convert_tokens_to_ids(gene_ids)
        if self.preprojection_embeddings is not None:
            index = torch.as_tensor(token_ids)
            return self.preprojection_embeddings[index].numpy()
        warnings.warn(
            f"{self.checkpoint} ships no {PREPROJECTION_FILE}; falling back to the "
            "projected gene embeddings in model.safetensors, which score below the "
            "published gene-function results",
            stacklevel=2,
        )
        weights = self.model.gene_embeddings.weight
        index = torch.as_tensor(token_ids, device=weights.device)
        return weights[index].detach().cpu().numpy()

    def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        """Apply EVA's log1p base-2 transform to released CPM values."""
        return np.log2(1.0 + expression)

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        """Identify genes represented by a dedicated vocabulary token."""
        return np.array(
            [self.tokenizer.gene_in_vocab(str(gene)) for gene in gene_ids],
            dtype=bool,
        )

    def gradient_perturb(
        self,
        expression: np.ndarray,
        gene_ids: list[str],
        target_gene_ids: list[str],
        directions: list[int],
    ) -> np.ndarray:
        """Apply the unit-normalized latent gradient step and decode."""
        if not self.has_expression_decoder:
            raise ValueError(
                f"{self.checkpoint} ships no expression-decoder weights, so its "
                "decoder would be randomly initialized. ZSP is unavailable for "
                "this checkpoint; encoder-only tasks are unaffected."
            )
        indices = [gene_ids.index(gene) for gene in target_gene_ids]
        outputs = [
            self._perturb_chunk(
                expression[start : start + self.batch_size],
                gene_ids,
                indices,
                directions,
            )
            for start in range(0, len(expression), self.batch_size)
        ]
        return np.concatenate(outputs)

    def _perturb_chunk(
        self,
        expression: np.ndarray,
        gene_ids: list[str],
        indices: list[int],
        directions: list[int],
    ) -> np.ndarray:
        """Perturb one batch of samples, keeping the backward graph bounded.

        Samples do not interact: the objective averages over the batch, but the
        step is renormalized per sample, so the averaging factor cancels and a
        chunked run matches a whole-cohort one. Running a cohort in one graph
        instead exhausts GPU memory -- a 1,174-sample cohort over a 600-gene
        panel asks a 24-layer encoder for ~13 GB of attention activations.
        """
        with torch.no_grad():
            encoded = self.model(**self._tokenize(expression, gene_ids)).gene_embeddings
        latent = encoded.detach().clone().requires_grad_(True)
        decoded = self._decode(latent)
        objective = sum(
            direction * decoded[:, index].mean()
            for index, direction in zip(indices, directions, strict=True)
        )
        gradient = torch.autograd.grad(objective, latent)[0]
        norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
        perturbed = latent + gradient / norm[:, None, None]
        output = self._decode(perturbed).clamp_min(0)
        source_sum = torch.as_tensor(expression, device=output.device).sum(1)
        output_sum = output.sum(1).clamp_min(1e-12)
        return (output * (source_sum / output_sum)[:, None]).detach().cpu().numpy()


def create_model(checkpoint: str | Path, device: str = "cpu", **kwargs: Any) -> EVARNA:
    """Create the benchmark adapter."""
    return EVARNA(
        checkpoint,
        device=device,
        batch_size=int(kwargs.get("batch_size") or DEFAULT_BATCH_SIZE),
    )
