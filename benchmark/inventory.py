"""Check that the datasets needed to run the benchmark are on disk and intact.

Checkpoints are not checked here: the linear baseline ships in
`checkpoints/`, and the others are downloaded from the Hub on first use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark.data import sha256_file

MANIFEST_FILES = (
    "manifest.csv",
    "tasks.csv",
    "zsp_pairs.csv",
    "gene_function_classes.csv",
    "feature_sets.csv.gz",
    "perturbation_feature_sets.csv.gz",
    "zsp_feature_sets.csv.gz",
    "perturbation_native_feature_sets.csv.gz",
)


@dataclass(frozen=True)
class Asset:
    """One required dataset file and what was found for it."""

    name: str
    present: bool
    detail: str


def _dataset_assets(root: Path) -> list[Asset]:
    """Describe every dataset file the manifests declare."""
    datasets = root / "datasets"
    tasks = pd.read_csv(datasets / "tasks.csv", dtype=str)
    zsp = pd.read_csv(datasets / "zsp_pairs.csv", dtype=str)
    expected = pd.read_csv(datasets / "manifest.csv", dtype=str).set_index("file")
    native = tasks["native_file"].dropna() if "native_file" in tasks else []
    required = sorted(
        {*tasks["file"], *native, *(f"{name}.csv.gz" for name in zsp["dataset_id"])}
    )
    return [
        _describe_matrix(datasets / name, expected)
        for name in (*MANIFEST_FILES, *required)
    ]


def _describe_matrix(path: Path, expected: pd.DataFrame) -> Asset:
    """Report presence and, when the manifest covers it, checksum agreement."""
    if not path.is_file():
        return Asset(path.name, False, "missing")
    size = f"{path.stat().st_size / 1e6:.1f} MB"
    if path.name not in expected.index:
        return Asset(path.name, True, size)
    recorded = expected.loc[path.name, "sha256"]
    if sha256_file(path) != recorded:
        return Asset(path.name, False, f"{size}, checksum mismatch")
    return Asset(path.name, True, f"{size}, checksum ok")


def collect(root: Path) -> list[Asset]:
    """Return the full dataset inventory."""
    return _dataset_assets(root)


def render(assets: list[Asset]) -> str:
    """Format the inventory as a readable report."""
    width = max(len(asset.name) for asset in assets)
    found = sum(asset.present for asset in assets)
    lines = [f"\ndatasets ({found}/{len(assets)} present)"]
    for asset in assets:
        mark = "ok  " if asset.present else "MISS"
        lines.append(f"  [{mark}] {asset.name:<{width}}  {asset.detail}")
    return "\n".join(lines)


def main() -> None:
    """Print the inventory and exit non-zero when an asset is unusable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    arguments = parser.parse_args()
    assets = collect(arguments.root)
    print(render(assets))
    unusable = [asset for asset in assets if not asset.present]
    if unusable:
        names = ", ".join(asset.name for asset in unusable)
        raise SystemExit(f"\n{len(unusable)} dataset file(s) unusable: {names}")
    print("\nAll dataset files are present.")


if __name__ == "__main__":
    main()
