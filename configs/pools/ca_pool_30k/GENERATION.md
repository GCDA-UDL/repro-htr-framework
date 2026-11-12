# ca_pool_30k — single CA synthetic pool (Article 1, third environment, coherent design)

THE Catalan synthetic pool. All subsamples for all CA blocks are drawn from here. Same recipe as
fr_pool_30k and en_pool_30k: framework v2.1.0, the same 196 downloaded fonts, the same options.

Generated 25/09/2026 (log: generation.log in this folder): run_pipeline.py (v2.1.0) from code/framework/
  python run_pipeline.py --language catalan --unique-texts \
    --category-filter Handwritten --total-images 30000 \
    --output-name ca_pool_30k --skip-fonts --verbose
Text: 1982 Catalan Wikipedia articles fetched through the API (2005 planned), unique texts.
Fonts: 145 Handwritten fonts used (num_fonts in output_ca_pool_30k/dataset_info.json; of the 194 fonts found in
the category, the pipeline keeps those with real glyphs for every required character of languages/catalan.json:
à è é í ï ò ó ú ü ç and the middle dot). For comparison FR used 159 and EN 187. Perturbations OFF (Clean),
style normal, mode lines. Total generated: 29090 images (dataset_info total_samples).

Converted + compiled 25/09/2026 with code/scripts/build_ca_pool.sh (log: build.log in this folder):
  convert_hf_to_kraken.py --input code/framework/output_ca_pool_30k --output data/synthetic/ca_pool_30k
  ketos compile -f path --force-type baseline -o ca_synth_30k_<split>_bl.arrow <split>/*.png
  subsample_arrow.py ca_synth_30k_train_bl.arrow ca_synth_4152_train_bl.arrow 4152   (seed 42)

Contents (.arrow, type kraken_recognition_baseline, im_mode RGB like the FR/EN pools):
  ca_synth_30k_train_bl.arrow      : 23268 lines, 96.6% unique texts, 1139059 characters
  ca_synth_30k_validation_bl.arrow :  2910 lines, 98.8% unique, 141573 characters
  ca_synth_30k_test_bl.arrow       :  2912 lines, 98.8% unique, 142974 characters
  ca_synth_4152_train_bl.arrow     :  4152 lines (subsample, seed 42; 1:1 with CatK12HW-actions train), 98.6% unique
The PNG/.gt.txt folders train/, validation/, test/ are the compile sources (not backed up; the arrows contain them).

## Normalization

Text is stored in the arrow files as generated. All training and evaluation in this framework use `-u NFD`;
always pass the same `-u` option to `ketos train` and `ketos test` (see the main README).
