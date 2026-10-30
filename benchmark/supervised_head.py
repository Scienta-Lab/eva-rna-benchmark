"""Supervised heads on encoder sample embeddings, frozen or last-layer fine-tuned."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from benchmark.runner import apply_model_preprocessing


@dataclass(frozen=True)
class SupervisedHeadProtocol:
    """Head and training settings shared by every encoder for one task category."""

    hidden_dim: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    mixup_alpha: float
    finetune_last_layer: bool
    warmup_cosine: bool
    batch_size: int = 16
    genes_per_step: int = 600
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    decay_epochs: int = 150
    min_learning_rate: float = 1e-6
    max_seconds: float = 12 * 3600


FROZEN = SupervisedHeadProtocol(
    hidden_dim=128,
    learning_rate=5e-4,
    weight_decay=0.1,
    max_epochs=200,
    patience=30,
    mixup_alpha=0.2,
    finetune_last_layer=False,
    warmup_cosine=True,
)
"""Settings of the paper's EVA runs for clinical activity and endotypes."""

LAST_LAYER = SupervisedHeadProtocol(
    hidden_dim=16,
    learning_rate=1e-5,
    weight_decay=0.4,
    max_epochs=3_000,
    patience=200,
    mixup_alpha=0.4,
    finetune_last_layer=True,
    warmup_cosine=False,
)
"""Settings of the paper's EVA run for treatment outcome."""

TREATMENT_OUTCOME_TASKS = frozenset(
    {
        "ibd_digestive_system_adalimumab_endo_remission",
        "ibd_digestive_system_infliximab_endo_remission",
        "ibd_adalimumab_to_infliximab_endo_remission",
        "ibd_infliximab_to_adalimumab_endo_remission",
        "ibd_vedolizumab_microarray_to_rnaseq_endo_remission",
        "ibd_vedolizumab_rnaseq_to_microarray_endo_remission",
    }
)
"""Tasks whose encoders fine-tune their last layer instead of staying frozen."""


def protocol_for(task_id: str) -> SupervisedHeadProtocol:
    """Return the protocol of a supervised task's category."""
    return LAST_LAYER if task_id in TREATMENT_OUTCOME_TASKS else FROZEN


class SupervisedHead(nn.Module):
    """Two-layer head on a sample embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return one prediction, or one logit per class, per sample."""
        return self.mlp(embedding)


@dataclass
class FittedHead:
    """A trained head and, for classification, the class order of its logits."""

    head: SupervisedHead
    classes: np.ndarray | None


def _learning_rate_factor(epoch: int, protocol: SupervisedHeadProtocol) -> float:
    """Linear warmup from zero, cosine decay, then a floor, stepped per epoch."""
    if not protocol.warmup_cosine:
        return 1.0
    if epoch < protocol.warmup_epochs:
        return epoch / max(1, protocol.warmup_epochs)
    floor = protocol.min_learning_rate / protocol.learning_rate
    progress = epoch - protocol.warmup_epochs
    if progress >= protocol.decay_epochs:
        return floor
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress / protocol.decay_epochs))
    return floor + (1.0 - floor) * cosine


def _embed(
    model: Any,
    log_expression: np.ndarray,
    gene_ids: list[str],
    *,
    trainable: bool = False,
) -> torch.Tensor:
    """Encode log2(CPM+1) values into sample embeddings.

    Mixup interpolates in log space, so the encoder's own transform is applied
    to the CPM recovered after mixing. `trainable` keeps the graph through the
    encoder's last layer; otherwise the embedding is computed without gradients
    and returned on the CPU.
    """
    cpm = np.exp2(log_expression.astype(np.float64)) - 1.0
    expression = apply_model_preprocessing(model, cpm.astype(np.float32))
    if trainable:
        return model.encode_samples_tensor(expression, gene_ids).float()
    with torch.no_grad():
        embedding = model.encode_samples(expression, gene_ids)
    return torch.as_tensor(np.asarray(embedding), dtype=torch.float32)


def _loss(output: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return nn.functional.cross_entropy(output, target)
    return nn.functional.mse_loss(output.squeeze(-1), target)


def fit_head(
    model: Any,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    gene_ids: list[str],
    *,
    task_type: str,
    seed: int,
    protocol: SupervisedHeadProtocol,
    log_dir: Path | None = None,
) -> FittedHead:
    """Train a head on embeddings recomputed by the encoder at every step.

    Expression is log2(CPM+1). Mixup mixes inputs and targets with one weight
    per batch; each batch then draws `genes_per_step` random genes, for training
    and validation alike. With `finetune_last_layer`, one AdamW optimizer also
    updates the encoder's last layer at the head's rate. The encoder stays in
    eval mode, so its dropout never acts. Early stopping only ends training:
    the last epoch's weights are kept.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_x, train_y = train
    val_x, val_y = val
    classes = None
    if task_type == "classification":
        classes = np.unique(train_y)
        index = {value: position for position, value in enumerate(classes)}
        train_y = np.array([index[value] for value in train_y])
        val_y = np.array([index[value] for value in val_y])
        output_dim = len(classes)
    else:
        train_y = np.asarray(train_y, dtype=np.float32)
        val_y = np.asarray(val_y, dtype=np.float32)
        output_dim = 1
    encoder_parameters = (
        model.trainable_last_layer() if protocol.finetune_last_layer else []
    )
    model.model.eval()
    device = model.device if protocol.finetune_last_layer else torch.device("cpu")
    encode_batch = getattr(model, "batch_size", None)
    if encode_batch is not None:
        model.batch_size = max(encode_batch, protocol.batch_size)
    width = _embed(model, train_x[:1], gene_ids).shape[-1]
    head = SupervisedHead(width, protocol.hidden_dim, output_dim).to(device)
    trainable = [*head.parameters(), *encoder_parameters]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    genes = np.arange(len(gene_ids))

    def batch_genes() -> np.ndarray:
        if len(genes) <= protocol.genes_per_step:
            return genes
        return np.sort(rng.permutation(genes)[: protocol.genes_per_step])

    def targets(values: np.ndarray) -> torch.Tensor:
        if task_type == "classification":
            return torch.as_tensor(values, dtype=torch.long)
        return torch.as_tensor(values, dtype=torch.float32)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
    best, stale, epoch = float("inf"), 0, 0
    started = time.monotonic()
    for epoch in range(protocol.max_epochs):
        rate = protocol.learning_rate * _learning_rate_factor(epoch, protocol)
        for group in optimizer.param_groups:
            group["lr"] = rate
        head.train()
        train_losses, train_sizes = [], []
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), protocol.batch_size):
            rows = order[start : start + protocol.batch_size]
            x, y = train_x[rows], targets(train_y[rows])
            if protocol.mixup_alpha > 0:
                weight = float(rng.beta(protocol.mixup_alpha, protocol.mixup_alpha))
                shuffled = rng.permutation(len(rows))
                x = weight * x + (1.0 - weight) * x[shuffled]
                if task_type == "classification":
                    one_hot = nn.functional.one_hot(y, output_dim).float()
                    y = weight * one_hot + (1.0 - weight) * one_hot[shuffled]
                else:
                    y = weight * y + (1.0 - weight) * y[shuffled]
            columns = batch_genes()
            embedding = _embed(
                model,
                x[:, columns],
                [gene_ids[c] for c in columns],
                trainable=protocol.finetune_last_layer,
            )
            optimizer.zero_grad()
            loss = _loss(head(embedding.to(device)), y.to(device), task_type)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, protocol.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            train_sizes.append(len(rows))
        head.eval()
        val_loss, val_size = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_x), protocol.batch_size):
                rows = np.arange(start, min(start + protocol.batch_size, len(val_x)))
                columns = batch_genes()
                embedding = _embed(
                    model, val_x[rows][:, columns], [gene_ids[c] for c in columns]
                )
                loss = _loss(
                    head(embedding.to(device)),
                    targets(val_y[rows]).to(device),
                    task_type,
                )
                val_loss += float(loss) * len(rows)
                val_size += len(rows)
        validation = val_loss / val_size
        if writer is not None:
            training = float(np.average(train_losses, weights=train_sizes))
            writer.add_scalar("loss/train", training, epoch)
            writer.add_scalar("loss/val", validation, epoch)
            writer.add_scalar("lr", rate, epoch)
        if validation < best:
            best, stale = validation, 0
        else:
            stale += 1
        if epoch % 100 == 0:
            print(
                f"  epoch {epoch}: train loss {np.mean(train_losses):.4f} "
                f"val loss {validation:.4f}",
                flush=True,
            )
        if stale >= protocol.patience:
            break
        if time.monotonic() - started > protocol.max_seconds:
            print("  stopped at the time limit", flush=True)
            break
    if writer is not None:
        writer.close()
    if encode_batch is not None:
        model.batch_size = encode_batch
    print(f"  stopped after {epoch + 1} epochs, best val loss {best:.4f}", flush=True)
    model.model.eval()
    head.eval()
    return FittedHead(head=head, classes=classes)


def predict_head(
    model: Any,
    fitted: FittedHead,
    log_expression: np.ndarray,
    gene_ids: list[str],
    *,
    task_type: str,
) -> np.ndarray:
    """Predict on every gene: class probabilities, or one value per sample."""
    embedding = _embed(model, log_expression, gene_ids)
    device = next(fitted.head.parameters()).device
    with torch.no_grad():
        output = fitted.head(embedding.to(device)).cpu()
    if task_type == "classification":
        return torch.softmax(output, dim=-1).numpy()
    return output.squeeze(-1).numpy()
