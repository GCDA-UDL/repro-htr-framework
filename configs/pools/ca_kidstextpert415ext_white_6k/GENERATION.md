# ca_kidstextpert415ext_white_6k — audit round 3 (2026-10-23)

Control B9a (S2-415ext): 415 training transcriptions NOT in the 500-line subset (case-insensitive), with the frequency profile of the S1 corpus (x12 -> 6000 lines; make_corpus_415ext.py, seed 42). Same textual diversity as S1, new in-domain text.

Same recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts, --text-lines, white plain paper,
--perturbations clean/degraded/severe 40/40/20, --unique-texts, 6000 images); only the text corpus changes.
Generated from code/framework/ (uncommitted --text-lines patch, same as the other matched-text pools):
  python build_dataset.py --data-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/text/catk12hw_415ext_x12 --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-dir /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/synthetic/output --output-name ca_kidstextpert415ext_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations
dataset_info.json: num_fonts 23, total_samples 5895, splits {'train': {'name': 'train', 'num_samples': 4721}, 'validation': {'name': 'validation', 'num_samples': 584}, 'test': {'name': 'test', 'num_samples': 590}}.
Framework output (PNG + metadata.jsonl kept): /home/joc13/workrepo-htr-synthetic-framework/audit_round3/data/synthetic/output_ca_kidstextpert415ext_white_6k
Arrows:
  ca_kidstextpert415ext_white_4152_train_bl.arrow: 4152 lines, 414 distinct texts, 62778 characters
  ca_kidstextpert415ext_white_6k_test_bl.arrow: 590 lines, 289 distinct texts, 8973 characters
  ca_kidstextpert415ext_white_6k_train_bl.arrow: 4721 lines, 414 distinct texts, 71419 characters
  ca_kidstextpert415ext_white_6k_validation_bl.arrow: 584 lines, 303 distinct texts, 8839 characters
1:1 subsample ca_kidstextpert415ext_white_4152_train_bl.arrow (code/scripts/subsample_arrow.py, seed 42). Checksums: SHA256.txt. Fonts: FONTS.txt.
Normalization: train and test with -u NFD, as every pool.
