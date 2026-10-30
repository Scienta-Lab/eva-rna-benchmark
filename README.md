# EVA RNA benchmark

This repository contains the EVA RNA benchmark published in our NeurIPS 2026 paper: 
"When Transcriptomic Foundation Models Scale: Domain-Focused Pretraining for Drug 
Development in Immunology and Inflammation".

It covers regression, classification, perturbation, gene-function
prediction, and gradient-based zero-shot perturbation (ZSP).

This benchmark has since evolved into an open-source, community-driven
leaderboard: [PRIMO leaderboard](https://primomics-primo-eval.hf.space/?tab=leaderboard).

## Layout

- `datasets/`: one compressed CSV per source matrix, plus task and ZSP manifests.
- `checkpoints/`: the linear-baseline checkpoint and the gene maps
  for scGPT and BulkRNABert; the other checkpoints come from the Hugging Face Hub.
- `models/`: one adapter per evaluated model family, and the checkpoint registry.
- `benchmark/`: local Metaflow harness, preprocessing, heads, metrics, and checks.

## Environment

`uv sync --locked --extra test` covers EVA, BulkRNABert, and the linear baseline. It does
not cover scGPT: scGPT's `GeneVocab` subclasses `torchtext.vocab.Vocab`, and
torchtext is archived at 0.18, whose compiled extension only loads against
torch 2.3. Installing scGPT beside a newer torch resolves, then fails at import
with `Could not load this library: libtorchtext.so`.

Build the full environment against that pin instead. Order matters: torch comes
from the CUDA index, torchtext only exists on PyPI, and scGPT leaves two
dependencies undeclared.

```bash
uv venv --python 3.12 .venv-full
uv pip install --python .venv-full/bin/python \
  torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-full/bin/python torchtext==0.18.0
uv pip install --python .venv-full/bin/python -e ".[scgpt,test]"
uv pip install --python .venv-full/bin/python -U typing_extensions ipython
```

The last line is not optional. Current `anndata` needs `typing_extensions`
newer than what torch 2.3.1 pins, and `scgpt.utils.util` imports `IPython`
without declaring it. `requirements-full.txt` pins the resolved set.

Verify all four adapters load before a long run:

```bash
.venv-full/bin/python -m pytest tests/ -q
```

## Getting the data

This repository holds code and manifests. The expression
matrices are ~170 MB and are hosted separately:

**https://osf.io/7qd8y/?view_only=5e9472a8914e454ea70d1acbc2c8f6a1**

Download its `datasets` folder and place it at the root of this repository, so
that `datasets/` sits beside `benchmark/`. Then confirm the result:

```bash
uv run python -m benchmark.inventory
```

It lists every expected file and exits non-zero if one is missing or does not
match its checksum, so it doubles as a check that the download landed correctly.

## Checkpoints

Pass a checkpoint by name with `--checkpoint`:

| `--checkpoint` | `--model` | Source |
| --- | --- | --- |
| `eva-60m` | `eva` | Hub [`ScientaLab/eva-rna`](https://huggingface.co/ScientaLab/eva-rna) |
| `scgpt-human` | `scgpt` | Hub [`wanglab/scGPT-human`](https://huggingface.co/wanglab/scGPT-human) |
| `bulkrnabert` | `bulkrnabert` | Hub [`InstaDeepAI/BulkRNABert`](https://huggingface.co/InstaDeepAI/BulkRNABert) |
| `linear` | `linear` | `checkpoints/linear/` in this repository |

Hub checkpoints are pinned to a commit in `models/checkpoints.py`, downloaded on
first use, and cached by `huggingface_hub`; set `HF_HUB_OFFLINE=1` to run from
the cache afterwards. EVA 60M is gated: accept its license on the model page,
then log in once with `uv run hf auth login`. A local directory also works as
`--checkpoint`.

scGPT and BulkRNABert map the benchmark's NCBI GeneIDs to their own vocabularies
with `checkpoints/scgpt_ncbi_to_symbol.json` and
`checkpoints/bulkrnabert_ncbi_to_position.json`; `--vocab-map` overrides either.


## Running

```bash
uv sync --extra test
uv run python -m benchmark.flow run \
  --benchmark-task ad_skin_easi \
  --model eva \
  --checkpoint eva-60m \
  --data-dir datasets \
  --results-dir runs
```

See `python -m benchmark.flow --help` and the manifests in `datasets/`. Other
models run the same way:

```bash
uv run python -m benchmark.flow run \
  --benchmark-task ad_skin_easi \
  --model scgpt \
  --checkpoint scgpt-human \
  --device cuda
```

EVA, scGPT, and BulkRNABert each encode in batches sized by their own
`DEFAULT_BATCH_SIZE`, chosen for their typical gene-panel size. Pass
`--batch-size` to override it per run:

```bash
uv run python -m benchmark.flow run \
  --benchmark-task cd_digestive_system_ghas \
  --model bulkrnabert \
  --checkpoint bulkrnabert \
  --batch-size 1 \
  --device cuda
```

ZSP writes two files: `<model>.csv` holds the single AUROC in the same schema as
every other family, and `<model>_pairs.csv` holds the per-pair detail it is
computed from, including pairs that could not be scored.

Run the 83-pair decoder benchmark separately:

```bash
uv run python -m benchmark.zsp \
  --model eva \
  --checkpoint eva-60m
```

Run its linear interaction baseline with:

```bash
uv run python -m benchmark.zsp \
  --model linear \
  --checkpoint linear
```

Run every task and seed a model supports -- 130 runs over the four families
the manifest drives. ZSP is not part of the sweep; it is run separately below:

```bash
uv run python -m benchmark.sweep --model eva --checkpoint eva-60m \
  --results-dir runs
```

`--dry-run` lists the runs without executing them, and `--family` or `--task`
narrow the sweep. Runs are sequential, because each merges its rows into a
shared result file and concurrent writers would interleave; shard with `--task`
across processes, giving each shard its own `--results-dir`.

`benchmark.suite` remains as a faster path for the linear regression suite
alone, and supports `--workers`:

```bash
uv run python -m benchmark.suite --model linear --results-dir runs --workers 4
```


## Licenses

The `LICENSE` file at the root of this repository is Apache-2.0 and covers the code 
and the linear checkpoint. It does not cover the third-party models, data, and resources
which are listed in `THIRD_PARTY_LICENSES`.

- Checkpoints are downloaded from the Hugging Face Hub rather than redistributed here, 
they keep their own terms.

- All datasets and resources used in this work are referenced in `manifest.csv`. Each 
come with their own license, users remain responsible for following the terms attached 
to each source study.
