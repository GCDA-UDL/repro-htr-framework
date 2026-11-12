# fr_pool_30k — single FR synthetic pool (Article 1, coherent design)

This is THE French synthetic pool. All subsamples for all FR blocks
(protocols, proportion, real-size) are drawn from here. Do not mix with others.

Generated: run_pipeline.py (v2.1.0) from code/framework/
  python run_pipeline.py --language french --unique-texts \
    --category-filter Handwritten --total-images 30000 \
    --output-name fr_page_30k --skip-fonts --verbose

Converted + compiled (CRITICAL: --force-type baseline):
  convert_hf_to_kraken.py --input output_fr_page_30k --output <dir>
  ketos compile -f path --force-type baseline -o <split>_bl.arrow <split>/*.png

Contents (.arrow baseline):
  fr_synth_30k_train_bl.arrow      : 23418 lines, 98.2% unique
  fr_synth_30k_validation_bl.arrow : 2909
  fr_synth_30k_test_bl.arrow       : 2927
  fr_synth_20k4_train_bl.arrow     : 20376 (subsample seed 42, for ratio 2:1)

159 Handwritten fonts used (num_fonts in output_fr_page_30k/dataset_info.json; of the 196 downloaded, the pipeline keeps those with real glyphs for every required character of languages/french.json), perturbations OFF (Clean).
Additional subsamples (5k, 10k...) are produced with subsample_arrow.py (seed 42)
from fr_synth_30k_train_bl.arrow as needed.

## Normalization

Text is stored in the arrow files as generated. All training and evaluation in this framework use `-u NFD`; always pass the same `-u` option to `ketos train` and `ketos test` (see the main README).
