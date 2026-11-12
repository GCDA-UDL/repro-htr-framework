# ca_kidstextpert_white_6k — catalan synthetic pool, fonts of the category KidsCurated

Article 2 quality cell 'kidstext-pert': KidsCurated fonts with the matched-text corpus (CatK12HW training transcriptions x2, --text-lines), white plain paper, --perturbations with the default quality distribution (clean 40, degraded 40, severe 20).

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/catk12hw_train --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-name ca_kidstextpert_white_6k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations
dataset_info.json: num_fonts 23, total_samples 5968 (32 of the 6000 tasks skipped because the
font lacked a required character), splits train 4777 / validation 595 / test 596.
Perturbations ON, style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (16/10/2026, median border colour): 150 sampled training images -> {'white': 126, 'grey': 24}.
Fonts: FONTS.txt in this folder (25 fonts chosen by hand: names, files and licence files).

Converted + compiled 16/10/2026 with `POOL=ca_kidstextpert_white_6k TAG=ca_kidstextpert_white_6k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_kidstextpert_white_4152_train_bl.arrow:   4152 lines, 45.1% unique texts, 56300 characters (kraken_recognition_baseline, RGB)
  ca_kidstextpert_white_6k_test_bl.arrow:    596 lines, 77.2% unique texts, 8008 characters (kraken_recognition_baseline, RGB)
  ca_kidstextpert_white_6k_train_bl.arrow:   4777 lines, 42.1% unique texts, 64843 characters (kraken_recognition_baseline, RGB)
  ca_kidstextpert_white_6k_validation_bl.arrow:    595 lines, 78.7% unique texts, 8200 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_kidstextpert_white_6k/ca_kidstextpert_white_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_kidstextpert_white_6k/ca_kidstextpert_white_6k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
