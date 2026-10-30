"""Tests for the release inventory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.data import sha256_file
from benchmark.inventory import collect, render


def _release(root: Path) -> Path:
    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    pd.DataFrame(
        {
            "family": ["regression"],
            "task_id": ["task"],
            "dataset_id": ["matrix"],
            "file": ["matrix.csv.gz"],
            "seeds": ["10"],
        }
    ).to_csv(datasets / "tasks.csv", index=False)
    pd.DataFrame({"dataset_id": ["cohort"], "pair_id": ["pair"]}).to_csv(
        datasets / "zsp_pairs.csv", index=False
    )
    frame = pd.DataFrame({"gene__1": [1.0]})
    for name in ("matrix.csv.gz", "cohort.csv.gz"):
        frame.to_csv(datasets / name, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["matrix"],
            "file": ["matrix.csv.gz"],
            "sha256": [sha256_file(datasets / "matrix.csv.gz")],
        }
    ).to_csv(datasets / "manifest.csv", index=False)
    return root


def test_inventory_reports_present_and_missing_assets(tmp_path: Path) -> None:
    assets = {asset.name: asset for asset in collect(_release(tmp_path))}
    assert assets["matrix.csv.gz"].present
    assert "checksum ok" in assets["matrix.csv.gz"].detail
    assert assets["tasks.csv"].present
    assert not assets["feature_sets.csv.gz"].present


def test_inventory_flags_a_checksum_mismatch(tmp_path: Path) -> None:
    root = _release(tmp_path)
    (root / "datasets" / "matrix.csv.gz").write_bytes(b"tampered")
    assets = {asset.name: asset for asset in collect(root)}
    assert not assets["matrix.csv.gz"].present
    assert "checksum mismatch" in assets["matrix.csv.gz"].detail


def test_render_reports_missing_datasets(tmp_path: Path) -> None:
    report = render(collect(_release(tmp_path)))
    assert "datasets (" in report
    assert "[MISS]" in report
