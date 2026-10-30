# Dataset guide and provenance

The public matrices used by the EVA benchmark. This document covers what the
files contain and where the data came from; task definitions, target
semantics, source publications and the evaluation protocol are in the paper.

The released files are derived benchmark tables, not unmodified copies of the
source studies.

## Licensing notes

GEO and BioStudies accessions describe provenance; they do not by themselves
grant a new license for derived files. Users remain responsible for following
the terms attached to each source study. The source accessions, resources and
releases behind every file are in the `source` column of `manifest.csv`; the
license and citation of each are in `THIRD_PARTY_LICENSES.md`.


## Manifests

Machine-readable provenance lives beside the matrices, and the harness reads it
rather than this document:

- `manifest.csv` -- one row per released matrix: family, organism,
  normalization, dimensions, sources (`|`-separated accessions, or the resource
  and release for gene-function files), and SHA-256. Checksums identify the
  exact derived matrices used by this benchmark.
- `tasks.csv` -- one row per task: target column, seeds, gene budget, primary
  metric, and which models support it. Transfer directions are encoded in the
  `task_id` (for example `ibd_adalimumab_to_infliximab_endo_remission`).
- `zsp_pairs.csv` -- the 83 disease/intervention pairs, with target genes,
  expected direction, and expected efficacy.
- `gene_function_classes.csv` -- the authoritative class list per gene-function
  task, so a run cannot silently score a different label set.
- `feature_sets.csv.gz`, `perturbation_feature_sets.csv.gz`,
  `zsp_feature_sets.csv.gz` -- the exact per-task, per-seed gene panels. All
  three are checksummed in `manifest.csv`, since they determine which genes a
  run sees.
- `*_mouse_native.csv.gz` and `perturbation_native_feature_sets.csv.gz` -- for
  the three perturbation tasks with mouse samples, those samples on mouse NCBI
  Gene IDs and their panels, read by EVA instead of the ortholog-mapped columns.

## Column contract

All expression features use NCBI Gene IDs, named `gene__<gene_id>`. Each
`*.csv.gz` is a unique source matrix; tasks that share samples reference the
same file from `tasks.csv`.

| family | columns |
| --- | --- |
| regression, classification | `sample_id`, optional `subject_id`, `gene__<gene_id>`, `target__<target>`, `split__<task_id>__seed_<seed>` |
| perturbation | matched `baseline__gene__<gene_id>` and `target__gene__<gene_id>` |
| gene function | `gene_id`, binary `target__<class>`, `split__<class>__seed_<seed>` |
| ZSP | `sample_id`, `cohort` (`diseased` or `healthy`), expression columns |

Split values are `train`, `val`, `test`, or empty when the sample is outside
that task. A matrix retains only the samples carrying its target, so files
derived from the same cohort can have different sample sets. Sample and subject
IDs were pseudonymized where retaining the accession-derived identifier was not
necessary for the task.

Export and evaluation fail if a gene-function matrix has missing or additional
target or split columns.

## Normalization

Expression is released as **CPM**, without a log transform, so every encoder
applies its own published normalization through `preprocess_expression`:
`log2(CPM+1)` for EVA and the linear baseline, `log10(1+CPM)` for BulkRNABert,
and 51-bin nonzero-quantile binning for scGPT. Normalization runs before gene
selection, so a panel is always fitted on the values the encoder consumes.

ZSP cohorts are the exception and store `log2(CPM + 1)`, because the ZSP runner
feeds them to the encoder directly. Which convention applies is a property of
the family, not of the file, and is recorded per matrix in the `normalization`
column of `manifest.csv`.

## Gene panels

Matrices contain only the genes the benchmark uses:

- supervised tasks: 8,000 genes selected on each training split;
- perturbation: 1,200 genes by absolute train-plus-validation log-fold change;
- ZSP: 600 genes by absolute disease-versus-healthy log-fold change, plus
  every intervention target gene.

Supervised selection is fit on training samples only. A source matrix stores
the union of its tasks' panels, and the loader selects the matching panel at
run time from the feature-set tables; perturbation panels are recorded
separately for train and test, after ortholog mapping, except the native panels
of `perturbation_native_feature_sets.csv.gz`, selected on mouse NCBI Gene IDs.
Perturbation panel genes outside an encoder's vocabulary are left out of
training; at test time EVA predicts them as their baseline plus the mean
training change and keeps them in the score, while scGPT and BulkRNABert leave
them out of the score.

## Gene-function datasets

Gene-level annotation matrices rather than patient cohorts. A positive label
means the source database associated a gene with the cell type, disease or
pathway **at the snapshot below**, which is the only record of these versions:

| file | source | snapshot |
| --- | --- | --- |
| `benchmark_gene_celltype_markers_matrix.csv.gz` | CNCB Cell Taxonomy | 15 evaluated human immune cell types |
| `benchmark_gene_disease_matrix.csv.gz` | Open Targets Platform `known_drug` | release 25.12, clinical phase 3+, Ensembl mapped to NCBI |
| `benchmark_gene_go_matrix.csv.gz` | GO Consortium `goa_human.gaf` | release 2025-10-10, ten immune/inflammatory BP terms |
| `benchmark_gene_reactome_matrix.csv.gz` | Reactome NCBI gene-to-pathway | release 95, twenty immune/inflammatory pathways |
| `benchmark_gene_wikipathways_matrix.csv.gz` | WikiPathways | eighteen selected immunology pathways; WP2637 carried no NCBI Gene IDs, so the matrix holds seventeen |

Labels are sparse and unevenly distributed -- a class may carry as few as two
positives out of 409 genes -- which is why the probe weights classes inversely
to their frequency and why a class whose test split ends up single-labelled is
excluded from the reported mean.

Because a released split is fixed per gene, it cannot depend on which encoder
is evaluated. The probe keeps the full gene set and represents genes outside an
encoder's vocabulary by a zero vector; dropping them would score each encoder
on a different test set. The foundation models use their own gene embeddings;
the linear baseline instead uses the released 512-dimensional projection of
gene-gene coexpression in `checkpoints/linear/gene_embeddings.pt`, which covers
97.5-100% of the genes in each matrix.

## ZSP cohorts

The five ZSP files are disease-versus-healthy bulk RNA-seq cohorts assembled
from public GEO studies. **They are not single studies.** Diseased samples are
pooled only after filtering to the named tissue and diagnosis, and healthy
samples come from a tissue-matched shared control pool -- which is why all three
skin cohorts contain the same 512 controls. The matrices hold only the gene
union the pairs in `zsp_pairs.csv` require, and are not complete reprocessed
versions of the source studies.

Each file's contributing GEO series are listed in the `source` column of
`manifest.csv`, without separating diseased from healthy samples. Publication
metadata for each study is on its GEO record and in `THIRD_PARTY_LICENSES.md`.

| file | diseased | healthy | pairs |
| --- | --- | --- | --- |
| `rnadataset-skin-dermatitis_atopic.csv.gz` | 257 | 512 | 22 |
| `rnadataset-skin-hidradenitis_suppurativa.csv.gz` | 65 | 512 | 11 |
| `rnadataset-skin-psoriasis.csv.gz` | 359 | 512 | 24 |
| `rnadataset-ileum-crohn_disease.csv.gz` | 1,174 | 450 | 11 |
| `rnadataset-colon-colitis_ulcerative.csv.gz` | 248 | 200 | 15 |
