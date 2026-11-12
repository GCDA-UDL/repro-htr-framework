# ca_whitepert_30k — catalan synthetic pool, fonts of the category Handwritten

From-scratch arm quality cell 'whitepert30k': as the generic base pool ca_white_30k (Handwritten fonts, Wikipedia-ca unique texts, white plain paper) with --perturbations (default quality distribution clean 40, degraded 40, severe 20).

Generated (log: generation.log in this folder): framework v2.1.0, run from code/framework/
  python build_dataset.py --data-dir data/wikipedia_ca --fonts-dir fonts --language catalan --category-filter Handwritten --background-color white --background-type plain --total-images 30000 --output-name ca_whitepert_30k --mode lines --style normal --workers 10 --verbose --unique-texts --perturbations
dataset_info.json: num_fonts 145, total_samples 29205 (795 of the 30000 tasks skipped because the
font lacked a required character), splits train 23374 / validation 2922 / test 2909.
Perturbations ON, style normal, mode lines.
Backgrounds: only the white plain paper textures of code/framework/backgrounds/ (campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).
Paper check (18/10/2026, median border colour): 150 sampled training images -> {'white': 114, 'grey': 36}.
Fonts: FONTS.txt in this folder (196 fonts chosen by hand: names, files and licence files).

Converted + compiled 18/10/2026 with `POOL=ca_whitepert_30k TAG=ca_whitepert_30k bash code/scripts/build_pool.sh ca 4152` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
  ca_whitepert_30k_test_bl.arrow    :   2909 lines, 98.8% unique texts, 144452 characters (kraken_recognition_baseline, RGB)
  ca_whitepert_30k_train_bl.arrow   :  23374 lines, 96.7% unique texts, 1145309 characters (kraken_recognition_baseline, RGB)
  ca_whitepert_30k_validation_bl.arrow:   2922 lines, 98.7% unique texts, 142305 characters (kraken_recognition_baseline, RGB)
  ca_whitepert_4152_train_bl.arrow  :   4152 lines, 98.6% unique texts, 203609 characters (kraken_recognition_baseline, RGB)
Use with the training scripts: SYN_TR_OVR=data/synthetic/ca_whitepert_30k/ca_whitepert_4152_train_bl.arrow
SYN_VAL_OVR=data/synthetic/ca_whitepert_30k/ca_whitepert_30k_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
