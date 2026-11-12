# ca_school_white_6k — catalan synthetic pool, fonts of the category School_ca

Article 2 typography cell 'school': every DaFont School font that passed the Catalan glyph check (fonts_sel/School_ca), generic Wikipedia text, white plain paper.

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/wikipedia_ca --fonts-dir fonts_sel --language catalan --category-filter School_ca --background-color white --background-type plain --total-images 6000 --output-name ca_school_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts
dataset_info.json: num_fonts 28, total_samples 6000 (0 of the 6000 tasks skipped because the
font lacked a required character), splits train 4800 / validation 600 / test 600.
Perturbations OFF (Clean), style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (07/10/2026, median border colour): 150 sampled training images -> {'white': 150}.
Fonts: FONTS.txt in this folder (31 fonts chosen by hand: names, files and licence files).

Converted + compiled 07/10/2026 with `POOL=ca_school_white_6k TAG=ca_school_white_6k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_school_white_4152_train_bl.arrow:   4152 lines, 98.6% unique texts, 200361 characters (kraken_recognition_baseline, RGB)
  ca_school_white_6k_test_bl.arrow  :    600 lines, 99.7% unique texts, 30662 characters (kraken_recognition_baseline, RGB)
  ca_school_white_6k_train_bl.arrow :   4800 lines, 98.3% unique texts, 231104 characters (kraken_recognition_baseline, RGB)
  ca_school_white_6k_validation_bl.arrow:    600 lines, 99.7% unique texts, 29457 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_school_white_6k/ca_school_white_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_school_white_6k/ca_school_white_6k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
