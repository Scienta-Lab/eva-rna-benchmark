"""Linear and logistic frozen-expression baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge


class LinearBaseline:
    """Identity encoder paired with linear or logistic regression."""

    def __init__(self, checkpoint: str = "") -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._interaction_data: dict[str, Any] | None = None
        self._gene_embeddings: dict[str, Any] | None = None

    def preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        """Apply the published log1p base-2 transform to released CPM values."""
        return np.log2(1.0 + expression)

    def _load_gene_embeddings(self) -> dict[str, Any]:
        """Load the released gene embedding table used by gene-function probes.

        Gene-function tasks classify a gene from a vector describing that gene,
        which expression alone does not provide. The baseline supplies one from
        a low-dimensional projection of the gene-gene coexpression structure,
        the same source the ZSP interaction matrix is built from.
        """
        if self._gene_embeddings is None:
            if self.checkpoint is None:
                raise ValueError("Linear gene embeddings require a checkpoint")
            path = (
                self.checkpoint / "gene_embeddings.pt"
                if self.checkpoint.is_dir()
                else self.checkpoint
            )
            data = torch.load(path, map_location="cpu", weights_only=True)
            self._gene_embeddings = {
                "index": {
                    str(gene_id): position
                    for position, gene_id in enumerate(data["gene_ids"])
                },
                "embeddings": data["embeddings"].numpy(),
            }
        return self._gene_embeddings

    def known_gene_mask(self, gene_ids: list[str]) -> np.ndarray:
        """Identify genes present in the released gene embedding table."""
        index = self._load_gene_embeddings()["index"]
        return np.array([str(gene) in index for gene in gene_ids], dtype=bool)

    def gene_embeddings(self, gene_ids: list[str]) -> np.ndarray:
        """Return released gene embedding rows in requested NCBI GeneID order."""
        data = self._load_gene_embeddings()
        positions = [data["index"][str(gene)] for gene in gene_ids]
        return data["embeddings"][positions]

    def encode_samples(self, expression: np.ndarray, gene_ids: list[str]) -> np.ndarray:
        """Return expression values unchanged."""
        if expression.shape[1] != len(gene_ids):
            raise ValueError("expression and gene_ids have inconsistent dimensions")
        return np.asarray(expression, dtype=np.float32)

    @staticmethod
    def fit(
        features: np.ndarray,
        targets: np.ndarray,
        task_type: str,
        seed: int = 0,
    ) -> Ridge | LogisticRegression:
        """Fit the appropriate non-neural baseline.

        Features are fitted in float64, as in the paper. From scikit-learn 1.8
        the solvers keep float32 input in float32, and LBFGS then stops at the
        published tolerance after a fraction of the iterations.
        """
        np.random.seed(seed)
        features = np.asarray(features, dtype=np.float64)
        if task_type == "classification":
            return LogisticRegression(
                C=2e-5,
                tol=1e-7,
                solver="lbfgs",
                max_iter=20_000,
            ).fit(features, targets)
        return Ridge(
            alpha=8e4,
            solver="saga",
            tol=1e-7,
            max_iter=20_000,
        ).fit(features, targets)

    @staticmethod
    def fit_perturbation(
        baseline: np.ndarray,
        target: np.ndarray,
        gene_ids: list[str],
    ) -> pd.Series:
        """Fit the publication's per-gene average-perturbation baseline."""
        if baseline.shape != target.shape or baseline.shape[1] != len(gene_ids):
            raise ValueError("Perturbation arrays and gene IDs are inconsistent")
        return pd.Series((target - baseline).mean(axis=0), index=gene_ids)

    @staticmethod
    def predict_perturbation(
        baseline: np.ndarray,
        gene_ids: list[str],
        average_delta: pd.Series,
    ) -> np.ndarray:
        """Add learned mean changes, using zero for unseen cross-species genes."""
        if baseline.shape[1] != len(gene_ids):
            raise ValueError("Perturbation array and gene IDs are inconsistent")
        aligned = average_delta.reindex(gene_ids, fill_value=0.0).to_numpy(
            dtype=np.float32
        )
        return baseline + aligned

    def gradient_perturb(
        self,
        expression: np.ndarray,
        gene_ids: list[str],
        target_ids: list[str],
        directions: list[int],
    ) -> np.ndarray:
        """Apply the publication's normalized linear interaction perturbation."""
        if self.checkpoint is None:
            raise ValueError("Linear ZSP requires an interaction-matrix checkpoint")
        if len(target_ids) != len(directions):
            raise ValueError("Target genes and directions are inconsistent")
        if self._interaction_data is None:
            path = (
                self.checkpoint / "interaction_matrix.pt"
                if self.checkpoint.is_dir()
                else self.checkpoint
            )
            self._interaction_data = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        data = self._interaction_data
        interaction_genes = [str(gene_id) for gene_id in data["gene_ids"]]
        positions = {gene_id: index for index, gene_id in enumerate(interaction_genes)}
        perturbation = torch.zeros(len(interaction_genes), dtype=torch.float32)
        for target_id, direction in zip(target_ids, directions, strict=True):
            if target_id in positions:
                perturbation[positions[target_id]] = direction
        left = data["left_member"].float()
        right = data.get("right_member")
        effect = (
            left @ (right.float() @ perturbation)
            if right is not None
            else left @ perturbation
        )
        selected = torch.tensor(
            [positions.get(gene_id, -1) for gene_id in gene_ids],
            dtype=torch.long,
        )
        selected_effect = torch.zeros(len(gene_ids), dtype=torch.float32)
        known = selected.ge(0)
        selected_effect[known] = effect[selected[known]]
        selected_effect /= selected_effect.norm().clamp(min=1e-8)
        values = torch.as_tensor(expression, dtype=torch.float32)
        original_sum = values.sum(dim=1, keepdim=True).clamp(min=1e-8)
        perturbed = (values + selected_effect).clamp(min=0.0)
        perturbed *= original_sum / perturbed.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return perturbed.numpy()


def create_model(checkpoint: str = "", **_: Any) -> LinearBaseline:
    """Create the benchmark baseline."""
    return LinearBaseline(checkpoint=checkpoint)
