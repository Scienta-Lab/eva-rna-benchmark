"""Perturbation benchmark: predict post-treatment expression from baseline."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from benchmark.metrics import perturbation_metrics
from benchmark.runner import (
    _load_model,
    _task_row,
    _tensorboard_dir,
    _write_results,
    apply_model_preprocessing,
    result_file_name,
)


@dataclass(frozen=True)
class PerturbationProtocol:
    """Decoder and fine-tuning settings of one encoder family."""

    decode_mode: str
    hidden_dims: tuple[int, ...] = (2048, 1024)
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 16
    patience: int = 20
    max_epochs: int = 100_000
    max_seconds: float = 12 * 3600
    mixup_alpha: float = 0.5
    gradient_clip: float = 1.0
    encoder_input: str = "model"
    encoder_autocast: str | None = None
    gene_space: str = "ortholog"
    out_of_vocabulary: str = "fallback"


PROTOCOLS = {
    "eva": PerturbationProtocol(decode_mode="cls", gene_space="native"),
    "scgpt": PerturbationProtocol(
        decode_mode="per_gene",
        encoder_input="log2_cpm",
        encoder_autocast="float16",
        out_of_vocabulary="drop",
    ),
    "bulkrnabert": PerturbationProtocol(
        decode_mode="per_gene", out_of_vocabulary="drop"
    ),
}
"""Settings of the runs reported in the paper, per encoder family."""

NATIVE_FEATURE_SETS = "perturbation_native_feature_sets.csv.gz"
"""Panels on each species' own genes, for tasks with a `native_file`."""


def _log2_cpm(values: np.ndarray) -> np.ndarray:
    """Return the benchmark's canonical log2(CPM+1) expression space.

    Perturbation predicts expression, so the released CPM matrix is both the
    encoder input and the regression target. Encoders apply their own
    normalization through preprocess_expression, but the target and the metric
    baseline must stay in the single space the published deltas were measured
    in; scoring raw CPM deltas lets a few high-expression genes dominate the
    row-wise correlation and inflates every task.
    """
    return np.log2(1.0 + values).astype(np.float32)


def _vocabulary_genes(model: Any, gene_ids: list[str], scope: str) -> list[str]:
    """Restrict a panel to the genes the encoder can tokenize.

    Training uses only these genes. At test time the other genes are predicted
    as their baseline plus the mean training change, and stay in the score.
    """
    known_gene_mask = getattr(model, "known_gene_mask", None)
    if known_gene_mask is None:
        return gene_ids
    known = known_gene_mask(gene_ids)
    covered = [
        gene_id for gene_id, usable in zip(gene_ids, known, strict=True) if usable
    ]
    if not covered:
        raise ValueError(f"No {scope}-panel gene is present in the encoder vocabulary")
    if len(covered) < len(gene_ids):
        print(
            f"{scope} panel: {len(covered)} of {len(gene_ids)} genes are in the "
            "encoder vocabulary",
            flush=True,
        )
    return covered


class PerturbationDecoder(nn.Module):
    """MLP mapping one decoder input row to one gene's post-treatment value."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for index, (width_in, width_out) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(nn.Linear(width_in, width_out))
            if index < len(dims) - 2:
                layers.extend([nn.LayerNorm(width_out), nn.GELU()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one value per sample and gene."""
        return self.mlp(features).squeeze(-1)


def _decoder_input(
    model: Any, expression: np.ndarray, gene_ids: list[str], decode_mode: str
) -> torch.Tensor:
    """Return decoder input rows of shape (samples, genes, features).

    `cls` pairs the sample's CLS embedding with each gene's input embedding;
    `per_gene` uses each gene's contextual token.
    """
    if decode_mode == "per_gene":
        return model.encode_genes_tensor(expression, gene_ids)
    if decode_mode != "cls":
        raise ValueError(f"Unknown decode mode {decode_mode!r}")
    cls_embedding = model.encode_cls_tensor(expression, gene_ids)
    gene_embedding = model.input_gene_embeddings(gene_ids).to(cls_embedding.dtype)
    samples, genes = len(cls_embedding), len(gene_ids)
    return torch.cat(
        [
            cls_embedding[:, None, :].expand(samples, genes, -1),
            gene_embedding[None, :, :].expand(samples, genes, -1),
        ],
        dim=-1,
    )


def _encoder_input(
    model: Any, log_expression: np.ndarray, protocol: PerturbationProtocol
) -> np.ndarray:
    """Return what the encoder reads from log2(CPM+1) values.

    `model` converts back to CPM and applies the encoder's own transform, after
    mixup, which interpolates in log space. `log2_cpm` passes the log values
    through unchanged.
    """
    if protocol.encoder_input == "log2_cpm":
        return log_expression.astype(np.float32)
    if protocol.encoder_input != "model":
        raise ValueError(f"Unknown encoder input {protocol.encoder_input!r}")
    cpm = np.exp2(log_expression.astype(np.float64)) - 1.0
    return apply_model_preprocessing(model, cpm.astype(np.float32))


def _features(
    model: Any,
    log_expression: np.ndarray,
    gene_ids: list[str],
    protocol: PerturbationProtocol,
) -> torch.Tensor:
    """Encode log2(CPM+1) values into float32 decoder input rows.

    An encoder with its own autocast dtype (scGPT) runs under it. The others
    keep the caller's precision: float32 in training, bfloat16 at prediction,
    as in the paper's runs.
    """
    precision = (
        torch.autocast(
            "cuda",
            dtype=getattr(torch, protocol.encoder_autocast),
            enabled=model.device.type == "cuda",
        )
        if protocol.encoder_autocast is not None
        else contextlib.nullcontext()
    )
    with precision:
        features = _decoder_input(
            model,
            _encoder_input(model, log_expression, protocol),
            gene_ids,
            protocol.decode_mode,
        )
    return features.float()


def _mixup(
    baseline: np.ndarray,
    target: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a batch with a permutation of itself, using one weight for both arrays."""
    weight = rng.beta(alpha, alpha)
    order = rng.permutation(len(baseline))
    return (
        weight * baseline + (1.0 - weight) * baseline[order],
        weight * target + (1.0 - weight) * target[order],
    )


def _predict(
    model: Any,
    decoder: PerturbationDecoder,
    baseline: np.ndarray,
    gene_ids: list[str],
    protocol: PerturbationProtocol,
    autocast: bool,
) -> torch.Tensor:
    """Decode log2(CPM+1) baselines in batches, without gradients."""
    use_bf16 = autocast and model.device.type == "cuda"
    outputs = []
    with torch.no_grad():
        for start in range(0, len(baseline), protocol.batch_size):
            rows = baseline[start : start + protocol.batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                features = _features(model, rows, gene_ids, protocol)
                outputs.append(decoder(features).float())
    return torch.cat(outputs)


def _finetune_decoder(
    model: Any,
    protocol: PerturbationProtocol,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    gene_ids: list[str],
    seed: int,
    log_dir: Path | None = None,
) -> PerturbationDecoder:
    """Train the decoder jointly with the encoder's last layer.

    The encoder stays in eval mode throughout, so its dropout never acts, as in
    the paper's runs. Early stopping only ends training: the weights of the
    last epoch are kept.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder_parameters = model.trainable_last_layer()
    model.model.eval()
    train_baseline, train_target = splits["train"]
    val_baseline, val_target = splits["val"]
    with torch.no_grad():
        input_dim = _features(model, train_baseline[:1], gene_ids, protocol).shape[-1]
    decoder = PerturbationDecoder(input_dim, protocol.hidden_dims).to(model.device)
    trainable = [*decoder.parameters(), *encoder_parameters]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    val_target_tensor = torch.as_tensor(val_target, device=model.device)
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
    best_loss, stale, epoch = float("inf"), 0, 0
    started = time.monotonic()
    for epoch in range(protocol.max_epochs):
        decoder.train()
        batch_losses = []
        order = rng.permutation(len(train_baseline))
        for start in range(0, len(order), protocol.batch_size):
            rows = order[start : start + protocol.batch_size]
            baseline, target = train_baseline[rows], train_target[rows]
            if protocol.mixup_alpha > 0 and len(rows) > 1:
                baseline, target = _mixup(baseline, target, protocol.mixup_alpha, rng)
            optimizer.zero_grad()
            features = _features(model, baseline, gene_ids, protocol)
            loss = nn.functional.mse_loss(
                decoder(features),
                torch.as_tensor(target, dtype=torch.float32, device=model.device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, protocol.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        decoder.eval()
        prediction = _predict(
            model, decoder, val_baseline, gene_ids, protocol, autocast=False
        )
        validation = float(nn.functional.mse_loss(prediction, val_target_tensor))
        if writer is not None:
            writer.add_scalar("loss/train", float(np.mean(batch_losses)), epoch)
            writer.add_scalar("loss/val", validation, epoch)
        if validation < best_loss:
            best_loss, stale = validation, 0
        else:
            stale += 1
        if epoch % 25 == 0:
            print(
                f"  epoch {epoch}: train mse {np.mean(batch_losses):.4f} "
                f"val mse {validation:.4f}",
                flush=True,
            )
        if stale >= protocol.patience:
            break
        if time.monotonic() - started > protocol.max_seconds:
            print("  stopped at the time limit", flush=True)
            break
    if writer is not None:
        writer.close()
    print(
        f"  stopped after {epoch + 1} epochs, best val mse {best_loss:.4f}",
        flush=True,
    )
    model.model.eval()
    decoder.eval()
    return decoder


def _read_feature_sets(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"task_id": str, "gene_id": str})


def _has_panel(feature_sets: pd.DataFrame, task_id: str, seed: int, scope: str) -> bool:
    return bool(
        (
            feature_sets["task_id"].eq(task_id)
            & feature_sets["seed"].eq(seed)
            & feature_sets["scope"].eq(scope)
        ).any()
    )


def _gene_panel(
    feature_sets: pd.DataFrame,
    task_id: str,
    seed: int,
    scope: str,
) -> list[str]:
    selected = feature_sets[
        feature_sets["task_id"].eq(task_id)
        & feature_sets["seed"].eq(seed)
        & feature_sets["scope"].eq(scope)
    ].sort_values("rank")
    if selected.empty:
        raise ValueError(f"No {scope} perturbation features for {task_id} seed {seed}")
    return selected["gene_id"].astype(str).tolist()


def run_perturbation(
    *,
    task_id: str,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    seed: int,
    device: str = "cpu",
    vocab_map: str | None = None,
    batch_size: int | None = None,
    tensorboard: bool = False,
    result_model_name: str | None = None,
    **_: Any,
) -> Path:
    """Predict post-treatment expression and score it against the observed change."""
    task = _task_row(data_dir, task_id)
    if task["family"] != "perturbation":
        raise ValueError(f"{task_id} is not a perturbation task")
    frame = pd.read_csv(data_dir / task["file"])
    feature_sets = _read_feature_sets(data_dir / "perturbation_feature_sets.csv.gz")
    train_frame, train_sets = test_frame, test_sets = frame, feature_sets
    protocol = PROTOCOLS.get(model_name)
    if (
        protocol is not None
        and protocol.gene_space == "native"
        and task.get("native_file", "")
    ):
        train_frame = pd.read_csv(data_dir / task["native_file"])
        train_sets = _read_feature_sets(data_dir / NATIVE_FEATURE_SETS)
        if _has_panel(train_sets, task_id, seed, "test"):
            test_frame, test_sets = train_frame, train_sets
    train_gene_ids = _gene_panel(train_sets, task_id, seed, "train")
    test_gene_ids = _gene_panel(test_sets, task_id, seed, "test")
    train_baseline_columns = [
        f"baseline__gene__{gene_id}" for gene_id in train_gene_ids
    ]
    train_target_columns = [f"target__gene__{gene_id}" for gene_id in train_gene_ids]
    test_baseline_columns = [f"baseline__gene__{gene_id}" for gene_id in test_gene_ids]
    test_target_columns = [f"target__gene__{gene_id}" for gene_id in test_gene_ids]
    for source, columns in (
        (train_frame, train_baseline_columns + train_target_columns),
        (test_frame, test_baseline_columns + test_target_columns),
    ):
        missing = set(columns) - set(source)
        if missing:
            raise ValueError(f"{task_id} data is missing {len(missing)} panel columns")
    split_column = f"split__{task_id}__seed_{seed}"
    masks = {
        name: train_frame[split_column].eq(name).to_numpy() for name in ("train", "val")
    }
    masks["test"] = test_frame[split_column].eq("test").to_numpy()
    model = _load_model(model_name, checkpoint, device, vocab_map, batch_size)
    if model_name == "linear":
        train_baseline = _log2_cpm(
            train_frame.loc[masks["train"], train_baseline_columns].to_numpy(
                dtype=np.float32
            )
        )
        train_target = _log2_cpm(
            train_frame.loc[masks["train"], train_target_columns].to_numpy(
                dtype=np.float32
            )
        )
        average_delta = model.fit_perturbation(
            train_baseline,
            train_target,
            train_gene_ids,
        )
        test_baseline = _log2_cpm(
            test_frame.loc[masks["test"], test_baseline_columns].to_numpy(
                dtype=np.float32
            )
        )
        test_target = _log2_cpm(
            test_frame.loc[masks["test"], test_target_columns].to_numpy(
                dtype=np.float32
            )
        )
        prediction = model.predict_perturbation(
            test_baseline,
            test_gene_ids,
            average_delta,
        )
    else:
        protocol = PROTOCOLS[model_name]
        if getattr(model, "n_bins", None) is not None:
            model.inputs_prebinned = True
        fit_gene_ids = _vocabulary_genes(model, train_gene_ids, "train")
        predicted_gene_ids = _vocabulary_genes(model, test_gene_ids, "test")
        splits = {
            name: (
                _log2_cpm(
                    train_frame.loc[
                        masks[name], [f"baseline__gene__{g}" for g in fit_gene_ids]
                    ].to_numpy(dtype=np.float32)
                ),
                _log2_cpm(
                    train_frame.loc[
                        masks[name], [f"target__gene__{g}" for g in fit_gene_ids]
                    ].to_numpy(dtype=np.float32)
                ),
            )
            for name in ("train", "val")
        }
        train_val = masks["train"] | masks["val"]
        fallback_delta = float(
            np.mean(
                _log2_cpm(
                    train_frame.loc[train_val, train_target_columns].to_numpy(
                        np.float32
                    )
                )
                - _log2_cpm(
                    train_frame.loc[train_val, train_baseline_columns].to_numpy(
                        np.float32
                    )
                )
            )
        )
        decoder = _finetune_decoder(
            model,
            protocol,
            splits,
            fit_gene_ids,
            seed,
            log_dir=_tensorboard_dir(
                results_dir,
                model_name=result_file_name(model_name, checkpoint, result_model_name),
                task_id=task_id,
                seed=seed,
                enabled=tensorboard,
            ),
        )
        test_baseline = _log2_cpm(
            test_frame.loc[masks["test"], test_baseline_columns].to_numpy(
                dtype=np.float32
            )
        )
        test_target = _log2_cpm(
            test_frame.loc[masks["test"], test_target_columns].to_numpy(
                dtype=np.float32
            )
        )
        in_vocabulary = [test_gene_ids.index(g) for g in predicted_gene_ids]
        if protocol.out_of_vocabulary == "drop":
            test_baseline = test_baseline[:, in_vocabulary]
            test_target = test_target[:, in_vocabulary]
            in_vocabulary = list(range(len(predicted_gene_ids)))
        elif protocol.out_of_vocabulary != "fallback":
            raise ValueError(
                f"Unknown out_of_vocabulary {protocol.out_of_vocabulary!r}"
            )
        elif len(predicted_gene_ids) < len(test_gene_ids):
            print(
                f"test panel: {len(test_gene_ids) - len(predicted_gene_ids)} "
                "out-of-vocabulary genes predicted as baseline plus the mean "
                "training change",
                flush=True,
            )
        prediction = test_baseline + fallback_delta
        prediction[:, in_vocabulary] = (
            _predict(
                model,
                decoder,
                test_baseline[:, in_vocabulary],
                predicted_gene_ids,
                protocol,
                autocast=True,
            )
            .cpu()
            .numpy()
        )
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    metrics = perturbation_metrics(
        test_target,
        prediction,
        test_baseline,
    )
    return _write_results(
        results_dir,
        family=task["family"],
        task_id=task_id,
        target=task["target"],
        model_name=result_file_name(model_name, checkpoint, result_model_name),
        seed=seed,
        metrics=metrics,
    )
