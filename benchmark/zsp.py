"""Gradient-based zero-shot perturbation evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betainc
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from benchmark.runner import _load_model, result_file_name

_PANEL_SIZE = 600
_MAX_COMPONENTS = 50
_CLUSTER_MIN_SIZE = 10
_CLUSTER_MIN_SAMPLES = 5

_TARGET_EXPLAINED_VARIANCE = 0.8
"""Variance the adaptive PCA must reach, for every model alike.

The retained dimension enters the null distribution through
`betainc((d - 1) / 2, ...)`, so every model is scored under one target.
"""


def _limit_panel(ranked_genes: list[str], target_genes: list[str]) -> list[str]:
    """Keep the top-ranked panel and ensure every intervention target is present."""
    return list(dict.fromkeys([*ranked_genes[:_PANEL_SIZE], *target_genes]))


def _check_targets_in_vocabulary(model: Any, target_ids: list[str]) -> None:
    """Fail when the perturbation objective cannot reference its own targets.

    Out-of-vocabulary panel genes are not an error: they hold their baseline
    value so the panel width never changes. An intervention target is
    different, since the objective is built from it.
    """
    known_gene_mask = getattr(model, "known_gene_mask", None)
    if known_gene_mask is None:
        return
    known = known_gene_mask(target_ids)
    missing = [
        target for target, usable in zip(target_ids, known, strict=True) if not usable
    ]
    if missing:
        raise ValueError(
            f"Intervention target genes {missing} are outside the encoder "
            "vocabulary, so the perturbation objective cannot be formed"
        )


def _in_perturbation_space(
    model: Any, expression: np.ndarray, gene_ids: list[str]
) -> np.ndarray:
    """Express a cohort in the space the model's perturbation is returned in.

    scGPT decodes binned expression, so its baseline and healthy cohorts are
    binned to match its perturbed profiles. EVA and the linear baseline return
    log2(CPM+1) values, which are used as stored.
    """
    to_space = getattr(model, "perturbation_space", None)
    return expression if to_space is None else to_space(expression, gene_ids)


def _adaptive_pca(
    values: np.ndarray,
    *,
    target_explained_variance: float,
) -> tuple[np.ndarray, int]:
    maximum = min(_MAX_COMPONENTS, values.shape[0], values.shape[1])
    ratios = np.cumsum(
        PCA(n_components=maximum, random_state=0).fit(values).explained_variance_ratio_
    )
    dimensions = max(
        2,
        min(
            int(np.searchsorted(ratios, target_explained_variance) + 1),
            maximum,
        ),
    )
    transformed = PCA(n_components=dimensions, random_state=0).fit_transform(values)
    return transformed, dimensions


def _cluster_alignment(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    healthy: np.ndarray,
    *,
    target_explained_variance: float,
) -> np.ndarray | None:
    """Score one pair, or None when the healthy cohort will not cluster.

    Such a pair records no cluster metric and is left out of the global AUROC
    rather than failing the suite.
    """
    combined, dimension = _adaptive_pca(
        np.vstack([baseline, perturbed, healthy]),
        target_explained_variance=target_explained_variance,
    )
    n_baseline = len(baseline)
    n_perturbed = len(perturbed)
    baseline_pca = combined[:n_baseline]
    perturbed_pca = combined[n_baseline : n_baseline + n_perturbed]
    healthy_pca = combined[n_baseline + n_perturbed :]
    labels = HDBSCAN(
        min_cluster_size=_CLUSTER_MIN_SIZE,
        min_samples=_CLUSTER_MIN_SAMPLES,
        copy=False,
    ).fit_predict(healthy_pca)
    valid = sorted(set(labels) - {-1})
    if len(valid) < 2:
        return None
    centroids = np.stack([healthy_pca[labels == label].mean(0) for label in valid])
    distances = ((baseline_pca[:, None] - centroids[None, :]) ** 2).sum(2)
    target = centroids[distances.argmin(1)]
    displacement = perturbed_pca - baseline_pca
    direction = target - baseline_pca
    displacement /= np.linalg.norm(displacement, axis=1, keepdims=True).clip(1e-15)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-15)
    cosine = (displacement * direction).sum(1)
    cosine = np.clip(cosine, -1.0, 1.0)
    return betainc(
        (dimension - 1) / 2,
        (dimension - 1) / 2,
        (1 + cosine) / 2,
    )


def run_zsp(
    *,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    device: str = "cpu",
    vocab_map: str | None = None,
    batch_size: int | None = None,
    result_model_name: str | None = None,
) -> Path:
    """Evaluate all 83 figure-defined disease/intervention pairs.

    `batch_size` bounds how many samples share one backward graph. It changes
    only memory and throughput -- the perturbation step is renormalized per
    sample -- but the adapter defaults are sized for encoding, not for holding
    gradients, so a large cohort on a small default barely occupies the GPU.
    """
    if model_name not in {"eva", "scgpt", "linear"}:
        raise ValueError("ZSP is available only for EVA, scGPT, and linear")
    pairs = pd.read_csv(data_dir / "zsp_pairs.csv", dtype=str)
    feature_sets_path = data_dir / "zsp_feature_sets.csv.gz"
    feature_sets = (
        pd.read_csv(
            feature_sets_path,
            dtype={"pair_id": str, "gene_id": str},
        )
        if feature_sets_path.is_file()
        else None
    )
    model = _load_model(model_name, checkpoint, device, vocab_map, batch_size)
    rows: list[dict[str, object]] = []
    print(f"scoring {len(pairs)} pairs for {model_name}", flush=True)
    for dataset_id, group in pairs.groupby("dataset_id", sort=False):
        frame = pd.read_csv(data_dir / f"{dataset_id}.csv.gz")
        for pair in group.itertuples(index=False):
            print(
                f"  [{len(rows) + 1}/{len(pairs)}] {dataset_id} {pair.pair_id}",
                flush=True,
            )
            target_ids = pair.target_gene_ids.split("|")
            if feature_sets is None:
                ranked_genes = [
                    column.removeprefix("gene__")
                    for column in frame
                    if column.startswith("gene__")
                ]
            else:
                selected = feature_sets[
                    feature_sets["pair_id"].eq(pair.pair_id)
                    & feature_sets["model"].eq(model_name)
                ].sort_values("rank")
                if selected.empty:
                    raise ValueError(
                        f"No ZSP features for {pair.pair_id} and {model_name}"
                    )
                ranked_genes = selected["gene_id"].tolist()
            _check_targets_in_vocabulary(model, target_ids)
            gene_ids = _limit_panel(ranked_genes, target_ids)
            gene_columns = [f"gene__{gene_id}" for gene_id in gene_ids]
            expression = frame[gene_columns].to_numpy(dtype=np.float32)
            diseased = frame["cohort"].eq("diseased").to_numpy()
            healthy = expression[frame["cohort"].eq("healthy").to_numpy()]
            baseline = expression[diseased]
            directions = [int(value) for value in pair.directions.split("|")]
            perturbed = model.gradient_perturb(
                baseline, gene_ids, target_ids, directions
            )
            baseline = _in_perturbation_space(model, baseline, gene_ids)
            healthy = _in_perturbation_space(model, healthy, gene_ids)
            percentile = _cluster_alignment(
                baseline,
                perturbed,
                healthy,
                target_explained_variance=_TARGET_EXPLAINED_VARIANCE,
            )
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "dataset_id": dataset_id,
                    "disease": pair.disease,
                    "intervention": pair.intervention,
                    "expected_efficacy": int(pair.expected_efficacy == "true"),
                    "alignment_percentile_median": (
                        None if percentile is None else float(np.median(percentile))
                    ),
                    "scored": percentile is not None,
                }
            )
            if percentile is None:
                print(
                    f"{pair.pair_id}: healthy cohort did not cluster, excluded",
                    flush=True,
                )
    output = pd.DataFrame(rows)
    scored = output[output["scored"]]
    if len(scored) < len(output):
        print(
            f"{len(scored)} of {len(output)} pairs scored; the AUROC below covers "
            "only those, and the rest are retained with scored=False",
            flush=True,
        )
    auroc = roc_auc_score(
        scored["expected_efficacy"], scored["alignment_percentile_median"]
    )
    category_dir = results_dir / "zsp"
    category_dir.mkdir(parents=True, exist_ok=True)
    output_name = result_file_name(model_name, checkpoint, result_model_name)

    output["scored_pairs"] = len(scored)
    output["global_auroc"] = auroc
    output.to_csv(category_dir / f"{output_name}_pairs.csv", index=False)

    destination = category_dir / f"{output_name}.csv"
    pd.DataFrame(
        [
            {
                "family": "zsp",
                "task_id": "target_efficacy",
                "target": "expected_efficacy",
                "model": output_name,
                "split": "test",
                "seed": "",
                "aggregation": "fixed",
                "metric": "auroc",
                "value": float(auroc),
            }
        ]
    ).to_csv(destination, index=False)
    return destination


def main() -> None:
    """Run ZSP from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["eva", "scgpt", "linear"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-map")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--result-model-name")
    arguments = parser.parse_args()
    destination = run_zsp(
        model_name=arguments.model,
        checkpoint=arguments.checkpoint,
        data_dir=arguments.data_dir,
        results_dir=arguments.results_dir,
        device=arguments.device,
        vocab_map=arguments.vocab_map,
        batch_size=arguments.batch_size,
        result_model_name=arguments.result_model_name,
    )
    print(destination)


if __name__ == "__main__":
    main()
