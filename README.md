# Synthetic Handwriting for HTR Fine-Tuning: Controlled Evaluation Across Languages, Domains and Data Sizes (reproduction package)

Reproduction package for the article *Synthetic Handwriting for HTR Fine-Tuning: Controlled Evaluation Across
Languages, Domains and Data Sizes* (J. Oliva, D. Grao, R. Tomàs, X. Carrera and V. L. da Silva). It regenerates every
table and every data figure of the article and of its supplementary material from the captured results, and checks
them, cell by cell, against the LaTeX sources of the article.

<!-- After the first release: [![DOI](https://zenodo.org/badge/DOI/<concept DOI>.svg)](https://doi.org/<concept DOI>) -->

**Status:** 1.0.0 in preparation (first submission). The concept DOI is assigned by Zenodo when the first version is
released.

## Quick start

```bash
docker build --target analysis -t repro-htr-framework:analysis .
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/repro -v /path/to/paper-checkout:/paper:ro \
  repro-htr-framework:analysis make all check PAPER=/paper
```

`make all` writes `out/` (statistics, `out/tables.md`, `out/figures/*.csv`); `make check` compares them with the article
sources and writes `out/check_manuscript.md` (exit status 1 if a table or figure cell differs). Without Docker:
`pip install -r docker/requirements-analysis.txt && make all`.

## What is included

| Folder | Content |
|---|---|
| `results/kraken/` | Every Kraken run of the article: `all_results.csv` (campaign c2, synthetic background matched to the real paper), `all_results_mixedbg.csv` (campaign c1, heterogeneous background), `results_round2.csv` and `results_round3.csv` (S1/S2 controls); `ketos_test_summaries.csv` (character, error, insertion, deletion and substitution counts of each `ketos test` report) and `phase1_checkpoints.csv` (checkpoint that starts each second curriculum phase) |
| `results/trocr/` | `test_results.csv`: aggregate test metrics of every TrOCR run |
| `results/aggregated/` | `lines.json`: captured output of the per-line analysis on CatK12HW-actions (cell means, hierarchical-bootstrap intervals, writer bound, pages worse, strata and interactions). No per-line value |
| `results/corpus/` | Corpus statistics of CatK12HW-actions v0.4 and the text coverage figures of the article (CC BY-NC 4.0, see `results/corpus/LICENSE`) |
| `analysis/` | `holm_family.py` (Welch intervals and the single Holm family of 82 contrasts), `lines_analysis.py` (hierarchical bootstrap over test pages and seeds, writer bound, strata), `tables.py` (every table and figure), `check_manuscript.py` (comparison with the article) |
| `code/` | Training and evaluation scripts as run on the experiment server: `training/` (Kraken, `ketos` 7.0.2), `scripts/` (pool compilation, subsampling, evaluation and registration), `trocr/`, and the S1/S2 control rounds (`audit_round2/`, `audit_round3/`). Paths are those of the server |
| `configs/pools/` | For every synthetic pool: `GENERATION.md` (exact generator command, counts and checksums of the compiled files), `FONTS.txt` and `dataset_info.json` |
| `checksums/` | SHA-256 of the compiled pool files and of every font file |
| `docker/`, `Dockerfile` | Environments with the package versions of the experiment server |
| `tools/export_derived.py` | How the per-line scores (on request), `results/corpus/` and the summaries were exported from the private data (runs only on the server) |
| `docs/curation_log.txt` | Edits made to the copied scripts |

## Map: table or figure → script

All entries are produced by `make all`; the regenerated values are in `out/tables.md` and `out/figures/`.

| Article | Script (label in `out/expected.json`) | Inputs |
|---|---|---|
| Table I (corpus) | `analysis/tables.py` (`tab:catk12hw`) | `results/corpus/catk12hw.json` |
| Table II (settings) | `analysis/tables.py` (`tab:settings`) | `all_results.csv`, `ketos_test_summaries.csv` |
| Table III, Fig. 4 (full real set) | `analysis/tables.py` (`tab:full`, `fig:deltafull`) | `all_results.csv`, `all_results_mixedbg.csv` |
| Table IV, Fig. 5, Fig. S1 (real-set size) | `analysis/tables.py` (`tab:size`, `fig:curves`, `fig:s-size`) | idem |
| Tables V, VI, VII (domain proximity, dose, variants) | `analysis/holm_family.py`, `analysis/tables.py` (`tab:content`, `tab:dose`, `tab:variants`) | `all_results.csv`, `results_round2.csv`, `results/corpus/` |
| Table VIII (hierarchical bootstrap) | `analysis/lines_analysis.py`, `analysis/tables.py` (`tab:boot`) | `results/aggregated/lines.json` (recomputed from the per-line scores when available) |
| Table IX (S1 controls) | idem (`tab:s1s2`) | `results_round2/3.csv`, `phase1_checkpoints.csv`, `results/aggregated/`, `results/corpus/` |
| Table X (TrOCR) | idem (`tab:trocr`) | `results/trocr/`, `results/aggregated/` |
| Table XI, Table S4 (strata) | idem (`tab:strata`, `tab:s-strata`) | `results/aggregated/` |
| Table XII (starting points) | idem (`tab:cross`) | `results/aggregated/`, `all_results.csv` |
| Table S3 (base model), Table S5 (from scratch) | `analysis/tables.py` (`tab:s-base`, `tab:s-scratch`) | `all_results_mixedbg.csv`, `all_results.csv` |
| Fig. S2 (error composition) | `analysis/tables.py` (`fig:s-errors`) | `ketos_test_summaries.csv` |
| Intervals and adjusted *p* values quoted in the text, factorial analysis, Spearman correlation | `analysis/tables.py` (`out/prose_values.json`), checked by `check_manuscript.py` | idem |

Figs. 1 and 3 are diagrams and Fig. 2 shows generator samples; Tables S1 and S2 review the literature. In Fig. S2 the
article calls *deletions* the reference characters the model does not emit, which `ketos test` reports in its
*Insertions* row (and vice versa).

## Hardware and running time

The analysis runs on any CPU in about 10 seconds (Docker image of about 500 MB). The experiments ran on one server
with two NVIDIA RTX A4000 GPUs (16 GB each); per-run training times were not logged. Retraining needs the data, the
base models and the generator:

- Kraken 7.0.2 with the FoNDUE-GD base models; the `kraken` target of the Dockerfile installs the environment.
- TrOCR (`microsoft/trocr-base-handwritten` and `microsoft/trocr-base-stage1`); `trocr` target.
- Synthetic pools: the generator is not copied here. It is pinned by version: *multilingual-synthetic-htr-dataset-generator*
  1.0.0 (code identical to commit `12f0918`, which adds the `--text-lines` mode used for the domain-text pools; repository
  github.com/joaquimoliva/multilingual-synthetic-htr-dataset-generator, to move to the GCDA-UDL organization). Generation
  is not deterministic, so `configs/pools/*/GENERATION.md` and `checksums/POOLS_SHA256.txt` describe the exact pools.

## What is not included, and why

- **CatK12HW-actions** images, transcriptions and per-line predictions: the corpus contains handwriting of minors and is
  deposited separately, with restricted access (CC BY-NC 4.0).
- **Per-line scores on the CatK12HW-actions test set** (`results/perline/`: characters, CER-cs, CER-ci and word errors
  of every line and run, pseudonymous page, strata; no image, transcription or prediction): available on request, with
  the same conditions and licence as the corpus (CC BY-NC 4.0), from the restricted Zenodo record of CatK12HW-actions.
  Copied into `results/perline/`, they let `make all` recompute the hierarchical bootstrap, and `make check-bootstrap`
  checks that the result equals `results/aggregated/lines.json`.
- **RIMES-2011-line and IAM**: available from their distributors.
- **Checkpoints** (about 60 GB) and **synthetic pool images**: the pools can be regenerated with the pinned generator;
  their checksums are in `checksums/`.
- **Fonts**: DaFont fonts have heterogeneous licences; the lists (`configs/pools/*/FONTS.txt`) and checksums
  (`checksums/FONTS_SHA256.txt`) are published instead.

## How to cite

Cite the concept DOI of this record (it resolves to the latest version):

> Oliva, J., Grao, D., Tomàs, R., Carrera, X., & da Silva, V. L. (2026). *Synthetic Handwriting for HTR Fine-Tuning:
> Controlled Evaluation Across Languages, Domains and Data Sizes (reproduction package)* (Version 1.0.0) [Computer
> software]. Zenodo. https://doi.org/&lt;concept DOI&gt;

`CITATION.cff` has the same information.

## Licence

Code: MIT (`LICENSE`). Captured results and documentation: CC BY 4.0 (`LICENSE-data`), except the data derived from
CatK12HW-actions (`results/corpus/` and the per-line scores available on request): CC BY-NC 4.0, as the corpus.

## Funding

This work has been granted by the Ministerio de Ciencia, Innovación y Universidades (MICIU) AEI/10.13039/501100011033 under contract PID2023-146193OB-I00.
