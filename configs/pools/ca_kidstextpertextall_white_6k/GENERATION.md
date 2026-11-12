# ca_kidstextpertextall_white_6k — audit round 3 (2026-10-23)

Control B9b (S2-extall): every training line whose text (case-insensitive) is NOT in the 500-line subset, natural frequencies (1919 lines x5 -> 9595 lines; make_corpus_415ext.py, seed 42). High diversity, no text of the 500 labelled lines.

Same recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts, --text-lines, white plain paper,
--perturbations clean/degraded/severe 40/40/20, --unique-texts, 6000 images); only the text corpus changes.
Generated from code/framework/ (uncommitted --text-lines patch, same as the other matched-text pools):
  python build_dataset.py --data-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/text/catk12hw_extall_x5 --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/synthetic/output --output-name ca_kidstextpertextall_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations
dataset_info.json: num_fonts 23, total_samples 5961, splits {'train': {'name': 'train', 'num_samples': 4769}, 'validation': {'name': 'validation', 'num_samples': 596}, 'test': {'name': 'test', 'num_samples': 596}}.
Framework output (PNG + metadata.jsonl kept): /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/synthetic/output_ca_kidstextpertextall_white_6k
Arrows:
  ca_kidstextpertextall_white_4152_train_bl.arrow: 4152 lines, 1552 distinct texts, 61085 characters
  ca_kidstextpertextall_white_6k_test_bl.arrow: 596 lines, 490 distinct texts, 8645 characters
  ca_kidstextpertextall_white_6k_train_bl.arrow: 4769 lines, 1585 distinct texts, 70470 characters
  ca_kidstextpertextall_white_6k_validation_bl.arrow: 596 lines, 507 distinct texts, 8780 characters
1:1 subsample ca_kidstextpertextall_white_4152_train_bl.arrow (code/scripts/subsample_arrow.py, seed 42). Checksums: SHA256.txt. Fonts: FONTS.txt.
Normalization: train and test with -u NFD, as every pool.
