"""Named model checkpoints, bundled in this repository or pinned on the Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

BUNDLED_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
"""Model files committed with this repository"""

BUNDLED = ("linear",)
"""Checkpoints shipped in `checkpoints/`, because no Hub repository hosts them."""

HUB = {
    "eva-60m": ("ScientaLab/eva-rna", "4232edf3c7efb01e4bb828be016f3fc86600f3e6"),
    "scgpt-human": (
        "wanglab/scGPT-human",
        "a24c237737a40f3720f75abb555489e9fe753be6",
    ),
    "bulkrnabert": (
        "InstaDeepAI/BulkRNABert",
        "7bcae6ac4f68fe1c781e8c1a626d8ee023747228",
    ),
}
"""Hub repository and commit for each remote checkpoint.

The commit pins the weights and, since EVA and BulkRNABert load with
`trust_remote_code`, the model code that runs.
"""


def resolve_checkpoint(checkpoint: str) -> str:
    """Return a local directory for a named checkpoint.

    Hub checkpoints are downloaded on first use and cached by `huggingface_hub`.
    Anything that is not a known name, such as a local path, is returned as is.
    """
    if checkpoint in BUNDLED:
        return str(BUNDLED_DIR / checkpoint)
    if checkpoint in HUB:
        repo_id, revision = HUB[checkpoint]
        return snapshot_download(repo_id, revision=revision)
    return checkpoint
