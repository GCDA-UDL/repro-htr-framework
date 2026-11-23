# Re-run of the English (IAM) experiments on the standard Aachen split

The English results reported in the article come from this re-run (November 2026), not from the original campaign,
which used a non-standard 8,917 / 2,732 / 1,704 split of the 13,353 IAM lines. The re-run keeps the recipe of
`code/training/train_blockA.sh` and `train_blockC.sh` (`ketos train -f binary --resize add -r 1e-4 --lag 5
--min-epochs 3 -u NFD`; curriculum-LR phase 2 at 1e-5; block C = phase 2 only, from the block A phase-1 checkpoint of
the same seed) and the same evaluation (`ketos test -f binary -u NFD` on the best checkpoint), and changes three things:

- real data = the Aachen split as released by Teklia on Hugging Face (`Teklia/IAM-line`, revision
  `fbdad97500ce54635c0d1ba306bf535cb40656cf`): 6,482 / 976 / 2,915 lines, writers disjoint across splits;
- the 1:1 synthetic subsample has 6,482 lines (`code/scripts/subsample_arrow.py`, seed 42) instead of 8,917;
- every run has an explicit seed (1, 2, 3); the original seed-1 runs had no fixed seed.

| File | Role |
|---|---|
| `convert_teklia_iam.py` | Converts the Hugging Face parquet files to Kraken `path` format (8-bit grayscale PNG as released, transcription unchanged); writes `iam_aachen_conversion_stats.json` |
| `iam_queue.py` | Resumable scheduler of the 62 jobs (one job per GPU, dependencies respected); the job definitions are the exact training and test commands |
| `iam_aachen_conversion_stats.json` | Line, character and image statistics of the three splits after conversion |

`iam_queue.py` calls `code/scripts/ketos_safe.py` and the arrow files produced by `ketos compile` from the converted
splits; their checksums are in `checksums/IAM_AACHEN_SHA256.txt`. The per-run reports (`eval_full.log`, `train.log`,
`run_info.txt`) are kept in the working repository of the project; their metrics are the English rows of
`results/kraken/all_results.csv` (campaign c2), `all_results_mixedbg.csv` (campaign c1) and `ketos_test_summaries.csv`,
written by `tools/ingest_iam_aachen.py`.
