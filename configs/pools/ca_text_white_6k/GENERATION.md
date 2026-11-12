# ca_text_white_6k — catalan synthetic pool, fonts of the category Handwritten

Article 2 text cell 'text': Handwritten fonts (as the base pool ca_white_30k) with the matched-text corpus (CatK12HW training transcriptions x2, --text-lines), white plain paper.

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/catk12hw_train --fonts-dir fonts --language catalan --category-filter Handwritten --background-color white --background-type plain --total-images 6000 --output-name ca_text_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines
dataset_info.json: num_fonts 145, total_samples 5883 (117 of the 6000 tasks skipped because the
font lacked a required character), splits train 4710 / validation 590 / test 583.
Perturbations OFF (Clean), style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (07/10/2026, median border colour): 150 sampled training images -> {'white': 150}.
Fonts: FONTS.txt in this folder (196 fonts chosen by hand: names, files and licence files).

Converted + compiled 07/10/2026 with `POOL=ca_text_white_6k TAG=ca_text_white_6k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_text_white_4152_train_bl.arrow :   4152 lines, 45.2% unique texts, 56087 characters (kraken_recognition_baseline, RGB)
  ca_text_white_6k_test_bl.arrow    :    583 lines, 77.2% unique texts, 7798 characters (kraken_recognition_baseline, RGB)
  ca_text_white_6k_train_bl.arrow   :   4710 lines, 42.0% unique texts, 63347 characters (kraken_recognition_baseline, RGB)
  ca_text_white_6k_validation_bl.arrow:    590 lines, 74.9% unique texts, 8108 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_text_white_6k/ca_text_white_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_text_white_6k/ca_text_white_6k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
