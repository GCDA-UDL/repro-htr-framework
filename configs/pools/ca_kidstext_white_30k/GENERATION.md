# ca_kidstext_white_30k — catalan synthetic pool, fonts of the category KidsCurated

Article 2 volume cell 'kidstext-30k': KidsCurated fonts with the matched-text corpus (CatK12HW training transcriptions, replicated x8 for --text-lines), white plain paper; 5x ca_kidstext_white_6k, to test the effect of synthetic volume (TrOCR phase 1, Kraken curriculum).

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/catk12hw_train_x8 --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 30000 --output-name ca_kidstext_white_30k --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines
dataset_info.json: num_fonts 23, total_samples 29853 (147 of the 30000 tasks skipped because the
font lacked a required character), splits train 23888 / validation 2981 / test 2984.
Perturbations OFF (Clean), style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (15/10/2026, median border colour): 150 sampled training images -> {'white': 150}.
Fonts: FONTS.txt in this folder (25 fonts chosen by hand: names, files and licence files).

Converted + compiled 15/10/2026 with `POOL=ca_kidstext_white_30k TAG=ca_kidstext_white_30k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_kidstext_white_30k_test_bl.arrow:   2984 lines, 47.8% unique texts, 40263 characters (kraken_recognition_baseline, RGB)
  ca_kidstext_white_30k_train_bl.arrow:  23888 lines, 9.8% unique texts, 322826 characters (kraken_recognition_baseline, RGB)
  ca_kidstext_white_30k_validation_bl.arrow:   2981 lines, 48.6% unique texts, 40667 characters (kraken_recognition_baseline, RGB)
  ca_kidstext_white_4152_train_bl.arrow:   4152 lines, 40.1% unique texts, 55286 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_kidstext_white_30k/ca_kidstext_white_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_kidstext_white_30k/ca_kidstext_white_30k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
