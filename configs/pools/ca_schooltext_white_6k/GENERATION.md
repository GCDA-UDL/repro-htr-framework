# ca_schooltext_white_6k — catalan synthetic pool, fonts of the category School_ca

Article 2 combined cell 'schooltext': School fonts with the matched-text corpus (--text-lines), white plain paper.

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/catk12hw_train --fonts-dir fonts_sel --language catalan --category-filter School_ca --background-color white --background-type plain --total-images 6000 --output-name ca_schooltext_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines
dataset_info.json: num_fonts 28, total_samples 5969 (31 of the 6000 tasks skipped because the
font lacked a required character), splits train 4777 / validation 596 / test 596.
Perturbations OFF (Clean), style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (07/10/2026, median border colour): 150 sampled training images -> {'white': 150}.
Fonts: FONTS.txt in this folder (31 fonts chosen by hand: names, files and licence files).

Converted + compiled 07/10/2026 with `POOL=ca_schooltext_white_6k TAG=ca_schooltext_white_6k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_schooltext_white_4152_train_bl.arrow:   4152 lines, 45.1% unique texts, 55763 characters (kraken_recognition_baseline, RGB)
  ca_schooltext_white_6k_test_bl.arrow:    596 lines, 77.5% unique texts, 8070 characters (kraken_recognition_baseline, RGB)
  ca_schooltext_white_6k_train_bl.arrow:   4777 lines, 41.7% unique texts, 64322 characters (kraken_recognition_baseline, RGB)
  ca_schooltext_white_6k_validation_bl.arrow:    596 lines, 78.2% unique texts, 8099 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_schooltext_white_6k/ca_schooltext_white_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_schooltext_white_6k/ca_schooltext_white_6k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
