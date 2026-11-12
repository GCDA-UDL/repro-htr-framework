# en_white_30k — english synthetic pool, fonts of the category Handwritten

Campaign-2 pool of the english environment (paper matched to the real data: white plain). Same fonts (DaFont Handwritten category as downloaded and verified by the framework), same Wikipedia texts and options as the campaign-1 pool en_pool_30k; only the paper background changes.

Generated (log: generation.log in this folder): run_pipeline.py (v2.1.0) from code/framework/
  python run_pipeline.py --language english --unique-texts --category-filter Handwritten --background-color white --background-type plain \
    --total-images 30000 --output-name en_white_30k --skip-fonts --skip-text --workers 10 --verbose
dataset_info.json: num_fonts 187, total_samples 29390 (610 of the 30000 tasks skipped because the
font lacked a required character), splits train 23512 / validation 2948 / test 2930.
Perturbations OFF (Clean), style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (05/10/2026, median border colour): 150 sampled training images -> {'white': 150}.
Fonts: the whole category as downloaded and verified by the framework.

Converted + compiled 05/10/2026 with `POOL=en_white_30k TAG=en_white_30k bash code/scripts/build_pool.sh en 8917` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  en_white_30k_test_bl.arrow        :   2930 lines, 99.9% unique texts, 153124 characters (kraken_recognition_baseline, RGB)
  en_white_30k_train_bl.arrow       :  23512 lines, 99.1% unique texts, 1205338 characters (kraken_recognition_baseline, RGB)
  en_white_30k_validation_bl.arrow  :   2948 lines, 99.6% unique texts, 153823 characters (kraken_recognition_baseline, RGB)
  en_white_8917_train_bl.arrow      :   8917 lines, 99.5% unique texts, 457933 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/en_white_30k/en_white_8917_train_bl.arrow
SYN_VAL_OVR=data/synthetic/en_white_30k/en_white_30k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
