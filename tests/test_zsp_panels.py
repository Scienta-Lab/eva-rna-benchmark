"""Tests pinning the ZSP gene panels to one shared definition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PANELS = Path(__file__).parents[1] / "datasets" / "zsp_feature_sets.csv.gz"


@pytest.mark.skipif(not PANELS.is_file(), reason="ZSP panels ship with the data")
def test_every_model_scores_the_same_panel() -> None:
    """The panel is stored per model but must not vary by model.

    `zsp_feature_sets.csv.gz` carries one copy per model, which reads as though
    each encoder were scored on genes chosen for it -- that would undercut any
    head-to-head comparison. The copies are identical, and this keeps them so.
    """
    panels = pd.read_csv(PANELS, dtype=str)
    assert set(panels["model"]) == {"eva", "linear", "scgpt"}
    differing = [
        pair_id
        for pair_id, group in panels.groupby("pair_id")
        if len({frozenset(rows["gene_id"]) for _, rows in group.groupby("model")}) > 1
    ]
    assert not differing, f"panels differ by model for {differing[:5]}"


@pytest.mark.skipif(not PANELS.is_file(), reason="ZSP panels ship with the data")
def test_panel_ranking_is_shared_too() -> None:
    """Ranks must match as well, since the panel is truncated by rank."""
    panels = pd.read_csv(PANELS, dtype=str)
    ordered = panels.sort_values(["pair_id", "model", "rank"])
    by_model = {
        model: group.groupby("pair_id")["gene_id"].apply(list)
        for model, group in ordered.groupby("model")
    }
    reference = by_model["eva"]
    for model, ranking in by_model.items():
        pd.testing.assert_series_equal(
            ranking, reference, check_names=False, obj=f"{model} ranking"
        )
