"""Tests for named checkpoints, bundled or pinned on the Hub."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models import bulkrnabert, checkpoints, scgpt
from models.eva_rna import EVARNA

BUNDLED_FILES = {
    "linear": ("interaction_matrix.pt", "gene_embeddings.pt"),
}


def test_bundled_checkpoints_ship_every_file_they_need() -> None:
    assert set(BUNDLED_FILES) == set(checkpoints.BUNDLED)
    for name, files in BUNDLED_FILES.items():
        directory = Path(checkpoints.resolve_checkpoint(name))
        missing = [item for item in files if not (directory / item).is_file()]
        assert not missing, f"{name} is missing {missing}"


def test_hub_checkpoints_download_their_pinned_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(repo_id: str, *, revision: str) -> str:
        calls.append((repo_id, revision))
        return f"/cache/{repo_id}"

    monkeypatch.setattr(checkpoints, "snapshot_download", fake_download)
    assert checkpoints.resolve_checkpoint("eva-60m") == "/cache/ScientaLab/eva-rna"
    assert calls == [checkpoints.HUB["eva-60m"]]
    for repo_id, revision in checkpoints.HUB.values():
        assert "/" in repo_id
        assert len(revision) == 40, f"{repo_id} must be pinned to a full commit"


def test_unknown_checkpoints_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "snapshot_download", pytest.fail)
    assert checkpoints.resolve_checkpoint("some/local/dir") == "some/local/dir"
    assert checkpoints.resolve_checkpoint("") == ""


def test_default_vocabulary_maps_ship_with_their_adapters() -> None:
    for path in (scgpt.DEFAULT_VOCAB_MAP, bulkrnabert.DEFAULT_VOCAB_MAP):
        with path.open() as handle:
            mapping = json.load(handle)
        assert isinstance(mapping, dict) and mapping, path


def test_eva_warns_when_falling_back_to_projected_gene_embeddings() -> None:
    adapter = EVARNA.__new__(EVARNA)
    adapter.checkpoint = Path("eva-checkpoint")
    adapter.preprojection_embeddings = None
    adapter.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda genes: [1, 0])
    adapter.model = SimpleNamespace(gene_embeddings=torch.nn.Embedding(2, 3))
    with pytest.warns(UserWarning, match="gene_embeddings_preprojection"):
        embeddings = adapter.gene_embeddings(["a", "b"])
    assert embeddings.shape == (2, 3)
