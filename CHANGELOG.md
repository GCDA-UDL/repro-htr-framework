# Changelog

All notable changes to this reproduction package are recorded here. Versions follow the git tags of the repository
(without the leading "v").

## [1.0.0] - unreleased (first submission of the article)

### Added
- Captured results of every Kraken and TrOCR run reported in the article (three settings: RIMES, IAM, CatK12HW-actions).
- Captured output of the per-line analysis on CatK12HW-actions (`results/aggregated/lines.json`) and the corpus
  statistics used by the article (CC BY-NC 4.0). The per-line scores themselves are available on request.
- Statistical analysis: Welch intervals, Holm correction over the single family of 82 contrasts, hierarchical bootstrap
  over test pages and seeds, writer-clustering bound and strata by line text.
- `analysis/tables.py` and `analysis/check_manuscript.py`: regeneration of every table and data figure and a cell-by-cell
  comparison with the article sources.
- Training and evaluation scripts, synthetic-pool configurations, checksums of pools and fonts.
- Docker environment (targets `analysis`, `kraken`, `trocr`) with the package versions of the experiment server.
