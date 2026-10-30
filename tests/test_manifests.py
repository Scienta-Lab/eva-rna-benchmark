"""Tests pinning the paper task inventories."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]

EXPECTED_TASK_COUNTS = {
    "classification": 8,
    "gene_function": 5,
    "perturbation": 6,
    "regression": 11,
}
"""Task count per family, as reported in the paper."""

EXPECTED_ZSP_COUNTS = {
    "Arthritis, Psoriatic": 11,
    "Colitis, Ulcerative": 15,
    "Crohn Disease": 11,
    "Dermatitis, Atopic": 22,
    "Hidradenitis Suppurativa": 11,
    "Psoriasis": 13,
}
"""ZSP pair count per disease, as reported in the paper."""

NORMALIZATION_BY_FAMILY = {
    "classification": "CPM",
    "feature_sets": "not_applicable",
    "gene_function": "not_applicable",
    "perturbation": "CPM",
    "regression": "CPM",
    "zsp": "log2(CPM+1)",
}
"""Expression convention each released family stores."""


def test_task_family_counts() -> None:
    tasks = pd.read_csv(ROOT / "datasets/tasks.csv")
    assert tasks.groupby("family").size().to_dict() == EXPECTED_TASK_COUNTS


def test_zsp_figure_pair_counts() -> None:
    pairs = pd.read_csv(ROOT / "datasets/zsp_pairs.csv")
    assert len(pairs) == 83
    assert pairs.groupby("disease").size().to_dict() == EXPECTED_ZSP_COUNTS
    assert not pairs["pair_id"].duplicated().any()


def test_zsp_multitarget_columns_are_aligned() -> None:
    pairs = pd.read_csv(ROOT / "datasets/zsp_pairs.csv", dtype=str)
    for row in pairs.itertuples(index=False):
        assert len(row.target_gene_symbols.split("|")) == len(
            row.target_gene_ids.split("|")
        )
        assert len(row.target_gene_ids.split("|")) == len(row.directions.split("|"))


def test_manifest_normalization_matches_its_family() -> None:
    manifest = pd.read_csv(ROOT / "datasets/manifest.csv", dtype=str)
    unknown = set(manifest["family"]) - set(NORMALIZATION_BY_FAMILY)
    assert not unknown, f"Unknown families in manifest: {sorted(unknown)}"
    for row in manifest.itertuples(index=False):
        assert row.normalization == NORMALIZATION_BY_FAMILY[row.family], (
            f"{row.dataset_id}: expected "
            f"{NORMALIZATION_BY_FAMILY[row.family]!r}, found {row.normalization!r}"
        )


def test_manifest_sources_match_third_party_licenses() -> None:
    manifest = pd.read_csv(ROOT / "datasets/manifest.csv", dtype=str)
    pattern = re.compile(r"GSE\d+|E-MTAB-\d+")
    declared = {a for source in manifest["source"] for a in pattern.findall(source)}
    licenses = (ROOT / "THIRD_PARTY_LICENSES.md").read_text()
    listed = {
        pattern.search(line).group()
        for line in licenses.splitlines()
        if line.startswith("| [") and "| Data |" in line
    }
    assert declared == listed, (
        f"in manifest only: {sorted(declared - listed)}; "
        f"in THIRD_PARTY_LICENSES.md only: {sorted(listed - declared)}"
    )


def test_every_encoder_can_finetune_its_last_layer() -> None:
    import inspect

    from models import bulkrnabert, eva_rna, scgpt

    for module in (eva_rna, scgpt, bulkrnabert):
        adapter = next(
            value
            for _, value in inspect.getmembers(module, inspect.isclass)
            if value.__module__ == module.__name__
        )
        assert hasattr(adapter, "trainable_last_layer"), module.__name__
        assert hasattr(adapter, "encode_genes_tensor"), module.__name__
    assert hasattr(eva_rna.EVARNA, "encode_cls_tensor")
    assert hasattr(eva_rna.EVARNA, "input_gene_embeddings")
