"""scGPT adapter matching the public checkpoint preprocessing path."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.checkpoints import BUNDLED_DIR

DEFAULT_BATCH_SIZE = 4
DEFAULT_BINNING_SEED = 42
DEFAULT_VOCAB_MAP = BUNDLED_DIR / "scgpt_ncbi_to_symbol.json"
"""NCBI GeneID to scGPT-human gene symbol, used unless --vocab-map overrides it."""
CORE_MODEL_PREFIXES = ("encoder.", "value_encoder.", "transformer_encoder.")


def _gene_token(
    gene_id: str,
    gene_token_map: dict[str, str] | None,
) -> str:
    """Resolve a benchmark GeneID to the token used by the checkpoint."""
    if gene_token_map is None:
        return str(gene_id)
    return gene_token_map.get(str(gene_id), str(gene_id))


def _known_gene_columns(
    gene_ids: list[str],
    vocab: Any,
    gene_token_map: dict[str, str] | None,
) -> list[tuple[int, str]]:
    """Return input columns and checkpoint tokens in matrix order."""
    return [
        (index, token)
        for index, gene_id in enumerate(gene_ids)
        if (token := _gene_token(gene_id, gene_token_map)) in vocab
    ]


def _digitize(
    values: np.ndarray,
    bins: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Match scGPT's randomized assignment when quantile edges are tied."""
    left = np.digitize(values, bins)
    right = np.digitize(values, bins, right=True)
    digits = rng.random(len(values)) * (right - left) + left
    return np.ceil(digits).astype(np.int64)


def bin_scgpt_expression(
    expression: np.ndarray,
    n_bins: int,
    *,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Bin each sample using the official scGPT nonzero-quantile algorithm."""
    if n_bins < 2:
        raise ValueError("scGPT n_bins must be at least two")
    if np.any(expression < 0):
        raise ValueError("scGPT binning requires non-negative expression values")
    generator = rng
    if generator is None:
        generator = np.random.RandomState(DEFAULT_BINNING_SEED)
    binned_rows: list[np.ndarray] = []
    quantiles = np.linspace(0, 1, n_bins - 1)
    for row in expression:
        nonzero = np.flatnonzero(row)
        binned = np.zeros_like(row, dtype=np.int64)
        if len(nonzero):
            nonzero_values = row[nonzero]
            bins = np.quantile(nonzero_values, quantiles)
            binned[nonzero] = _digitize(nonzero_values, bins, generator)
        binned_rows.append(binned)
    return np.stack(binned_rows).astype(np.float32, copy=False)


def prepare_scgpt_batch(
    expression: np.ndarray,
    gene_ids: list[str],
    vocab: Any,
    gene_token_map: dict[str, str] | None = None,
    *,
    n_bins: int | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter known genes, optionally bin values, and prepend zero-valued CLS."""
    known = _known_gene_columns(gene_ids, vocab, gene_token_map)
    if not known:
        raise ValueError("No input gene occurs in the scGPT vocabulary")
    columns = [index for index, _ in known]
    tokens = [vocab[token] for _, token in known]
    known_expression = expression[:, columns]
    if n_bins is not None:
        known_expression = bin_scgpt_expression(
            known_expression,
            n_bins,
            rng=rng,
        )
    cls_id = vocab["<cls>"]
    pad_id = vocab["<pad>"]
    token_ids = torch.full((len(expression), len(tokens) + 1), pad_id, dtype=torch.long)
    values = torch.zeros((len(expression), len(tokens) + 1), dtype=torch.float32)
    token_ids[:, 0] = cls_id
    token_ids[:, 1:] = torch.as_tensor(tokens, dtype=torch.long)
    values[:, 1:] = torch.as_tensor(known_expression, dtype=torch.float32)
    padding_mask = token_ids.eq(pad_id)
    return token_ids, values, padding_mask


def _compatible_checkpoint_state(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
    *,
    uses_fast_transformer: bool,
) -> dict[str, torch.Tensor]:
    """Select compatible public checkpoint weights and require the backbone."""
    if not uses_fast_transformer:
        checkpoint_state = {
            key.replace("Wqkv.", "in_proj_"): value
            for key, value in checkpoint_state.items()
        }
    compatible = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    missing = [
        key
        for key in model_state
        if key.startswith(CORE_MODEL_PREFIXES) and key not in compatible
    ]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"Checkpoint is missing required scGPT weights: {preview}")
    return compatible


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    """Load the public backbone while ignoring unused checkpoint heads."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    compatible = _compatible_checkpoint_state(
        model_state,
        state,
        uses_fast_transformer=bool(getattr(model, "use_fast_transformer", False)),
    )
    model_state.update(compatible)
    model.load_state_dict(model_state)


class ScGPT:
    """Official scGPT model supporting public and benchmark vocabularies."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        vocab_map: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        try:
            from scgpt.model import TransformerModel
            from scgpt.tokenizer.gene_tokenizer import GeneVocab
        except ImportError as error:
            raise ImportError("Install the 'scgpt' optional dependency") from error

        root = Path(checkpoint)
        with (root / "args.json").open() as handle:
            config = json.load(handle)
        self.vocab = GeneVocab.from_file(root / "vocab.json")
        self.gene_token_map: dict[str, str] | None = None
        if vocab_map is not None:
            with Path(vocab_map).open() as handle:
                mapping = json.load(handle)
            if not isinstance(mapping, dict):
                raise ValueError("scGPT --vocab-map must contain a JSON object")
            self.gene_token_map = {
                str(gene_id): str(token) for gene_id, token in mapping.items()
            }
        if batch_size < 1:
            raise ValueError("scGPT batch_size must be positive")
        self.batch_size = batch_size
        self.n_bins: int | None = None
        if config.get("input_style") == "binned":
            self.n_bins = int(config["n_bins"])
            if self.n_bins < 2:
                raise ValueError("scGPT n_bins must be at least two")
        self.inputs_prebinned = False
        kwargs = {
            "ntoken": len(self.vocab),
            "d_model": config["embsize"],
            "nhead": config["nheads"],
            "d_hid": config["d_hid"],
            "nlayers": config["nlayers"],
            "nlayers_cls": config.get("n_layers_cls", 3),
            "n_cls": 1,
            "vocab": self.vocab,
            "dropout": config.get("dropout", 0.2),
            "pad_token": "<pad>",
            "pad_value": 0.0,
            "input_emb_style": "continuous",
            "cell_emb_style": "avg-pool",
            "use_fast_transformer": config.get("fast_transformer", True),
        }
        signature = inspect.signature(TransformerModel)
        accepted = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        self.device = torch.device(device)
        self.model = TransformerModel(**accepted)
        _load_checkpoint(self.model, root / "best_model.pt")
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        """Bin released CPM values; quantile binning needs no library scaling."""
        return self.bin_expression(expression)

    def bin_expression(self, expression: np.ndarray) -> np.ndarray:
        """Bin values with the checkpoint's configuration, or pass them through."""
        if self.n_bins is None:
            return expression
        return bin_scgpt_expression(
            expression,
            self.n_bins,
            rng=np.random.RandomState(DEFAULT_BINNING_SEED),
        )

    def _encode(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self._forward(expression, gene_ids)

    def _forward(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer stack, keeping the autograd graph."""
        tokens, values, padding = prepare_scgpt_batch(
            expression,
            gene_ids,
            self.vocab,
            self.gene_token_map,
            n_bins=None if self.inputs_prebinned else self.n_bins,
            rng=np.random.RandomState(DEFAULT_BINNING_SEED),
        )
        tokens, values, padding = (
            tokens.to(self.device),
            values.to(self.device),
            padding.to(self.device),
        )
        encoded = self.model._encode(tokens, values, padding)
        return encoded, padding

    def encode_genes_tensor(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> torch.Tensor:
        """Return differentiable per-gene tokens, excluding the CLS position."""
        encoded, _ = self._forward(expression, gene_ids)
        genes = encoded[:, 1 : len(gene_ids) + 1]
        if genes.shape[1] != len(gene_ids):
            raise ValueError(
                f"scGPT returned {genes.shape[1]} gene tokens for {len(gene_ids)} genes"
            )
        return genes

    def trainable_last_layer(self) -> list[torch.nn.Parameter]:
        """Unfreeze the final transformer layer for last-layer fine-tuning.

        scGPT runs its whole stack before pooling, so the layer producing the
        embedding is the last one, unlike BulkRNABert which stops at layer four.

        The nested-tensor fast path has to go: the frozen early layers take it
        and emit a NestedTensor, which MultiheadAttention then refuses to accept
        once the final layer needs gradients.
        """
        self.model.transformer_encoder.enable_nested_tensor = False
        self.model.transformer_encoder.use_nested_tensor = False
        layer = self.model.transformer_encoder.layers[-1]
        parameters = list(layer.parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    def encode_samples(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Average token embeddings, including CLS and excluding padding."""
        if not len(expression):
            raise ValueError("Cannot encode an empty scGPT sample batch")
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
        """Return differentiable mean token embeddings for one batch."""
        encoded, padding = self._forward(expression, gene_ids)
        keep = (~padding).unsqueeze(-1)
        return (encoded * keep).sum(1) / keep.sum(1).clamp_min(1)

    def encode_genes(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Return token embeddings for the requested genes, excluding CLS."""
        if not len(expression):
            raise ValueError("Cannot encode an empty scGPT sample batch")
        token_batches: list[np.ndarray] = []
        for start in range(0, len(expression), self.batch_size):
            encoded, _ = self._encode(
                expression[start : start + self.batch_size],
                gene_ids,
            )
            genes = encoded[:, 1 : len(gene_ids) + 1]
            if genes.shape[1] != len(gene_ids):
                raise ValueError(
                    f"scGPT returned {genes.shape[1]} gene tokens for "
                    f"{len(gene_ids)} genes"
                )
            token_batches.append(genes.detach().cpu().numpy())
        return np.concatenate(token_batches)

    def gene_embeddings(self, gene_ids: list[str]) -> np.ndarray:
        """Return static embedding rows in requested order."""
        tokens = [_gene_token(gene_id, self.gene_token_map) for gene_id in gene_ids]
        indices = torch.as_tensor(
            [self.vocab[token] for token in tokens],
            device=self.device,
        )
        return self.model.encoder.embedding(indices).detach().cpu().numpy()

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        """Identify genes represented by the selected checkpoint vocabulary."""
        return np.array(
            [
                _gene_token(gene_id, self.gene_token_map) in self.vocab
                for gene_id in gene_ids
            ],
            dtype=bool,
        )

    def perturbation_space(
        self, expression: np.ndarray, gene_ids: list[str]
    ) -> np.ndarray:
        """Return expression as `gradient_perturb` outputs it.

        That is the in-vocabulary genes as the encoder reads them, binned with
        the same seed, so a baseline and its perturbation share one space.
        """
        _, values, _ = prepare_scgpt_batch(
            expression,
            gene_ids,
            self.vocab,
            self.gene_token_map,
            n_bins=None if self.inputs_prebinned else self.n_bins,
            rng=np.random.RandomState(DEFAULT_BINNING_SEED),
        )
        return values[:, 1:].numpy()

    def gradient_perturb(
        self,
        expression: np.ndarray,
        gene_ids: list[str],
        target_gene_ids: list[str],
        directions: list[int],
    ) -> np.ndarray:
        """Apply one unit-normalized latent step through the scGPT decoder.

        Expression is binned when the checkpoint declares `input_style`
        `binned`, which the public whole-human checkpoint does at 51 bins, and
        the decoder predicts in that space. The result therefore stays in it:
        one column per in-vocabulary gene, rescaled to the binned baseline
        total. Score it against `perturbation_space` of the baseline and healthy
        cohorts, not against the log2(CPM+1) values passed in.
        """
        tokens, values, padding = prepare_scgpt_batch(
            expression,
            gene_ids,
            self.vocab,
            self.gene_token_map,
            n_bins=None if self.inputs_prebinned else self.n_bins,
            rng=np.random.RandomState(DEFAULT_BINNING_SEED),
        )
        source = values[:, 1:].numpy()
        tokens, values, padding = (
            tokens.to(self.device),
            values.to(self.device),
            padding.to(self.device),
        )
        with torch.no_grad():
            encoded = self.model._encode(tokens, values, padding)
        latent = encoded.detach().clone().requires_grad_(True)
        decoded_raw = self.model.decoder(latent)
        decoded = decoded_raw["pred"] if isinstance(decoded_raw, dict) else decoded_raw
        known_genes = [
            str(gene_id)
            for gene_id in gene_ids
            if _gene_token(gene_id, self.gene_token_map) in self.vocab
        ]
        indices = [known_genes.index(str(gene)) + 1 for gene in target_gene_ids]
        objective = sum(
            direction * decoded[:, index].mean()
            for index, direction in zip(indices, directions, strict=True)
        )
        gradient = torch.autograd.grad(objective, latent)[0]
        norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
        perturbed_latent = latent + gradient / norm[:, None, None]
        output_raw = self.model.decoder(perturbed_latent)
        output = output_raw["pred"] if isinstance(output_raw, dict) else output_raw
        output = output[:, 1 : len(known_genes) + 1].clamp_min(0)
        perturbed = output.detach().float().cpu().numpy()
        source_total = perturbed.sum(1)
        scale = np.divide(
            source.sum(1),
            source_total,
            out=np.ones_like(source_total),
            where=source_total > 0,
        )
        return perturbed * scale[:, None]


def create_model(
    checkpoint: str | Path,
    device: str = "cpu",
    **kwargs: Any,
) -> ScGPT:
    """Create the benchmark adapter."""
    return ScGPT(
        checkpoint,
        device=device,
        vocab_map=kwargs.get("vocab_map") or DEFAULT_VOCAB_MAP,
        batch_size=int(kwargs.get("batch_size", DEFAULT_BATCH_SIZE)),
    )
