# ca_kidstextpert500_white_6k — audit round 2, control B1 (2026-10-22)

Same recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts, matched text with --text-lines, white plain
paper, --perturbations clean/degraded/severe 40/40/20, --unique-texts, 6000 images) with ONE change: the text corpus is
restricted to the 500 transcriptions of the Block C real subset (data/real/catk12hw/catk12hw_500_train.arrow, seed 42),
repeated x12 in a seeded shuffle (audit_round2/scripts/make_corpus_500.py). No other training, validation or test
transcription is used. Purpose: separate the visual effect of the matched synthetic data from the extra text
supervision that the full-train corpus provides in the "500 real lines" scenario.

Generated from code/framework/ (uncommitted --text-lines patch, same as the other matched-text pools):
  python build_dataset.py --data-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round2/data/text/catk12hw_500_x12 --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round2/data/synthetic/output --output-name ca_kidstextpert500_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations
dataset_info.json: num_fonts 23, total_samples 5979, splits {'train': {'name': 'train', 'num_samples': 4783}, 'validation': {'name': 'validation', 'num_samples': 597}, 'test': {'name': 'test', 'num_samples': 599}}.
Framework output (PNG + metadata.jsonl kept): /home/joc13/workrepo-htr-synthetic-framework/audit_round2/data/synthetic/output_ca_kidstextpert500_white_6k
Arrows: /home/joc13/workrepo-htr-synthetic-framework/audit_round2/data/synthetic/ca_kidstextpert500_white_6k/*.arrow; 1:1 subsample ca_kidstextpert500_white_4152_train_bl.arrow (code/scripts/subsample_arrow.py, seed 42).
Checksums: SHA256.txt. Fonts: FONTS.txt (copied from the kidstextpert pool; same KidsCurated folder).
Normalization: train and test with -u NFD, as every pool.
