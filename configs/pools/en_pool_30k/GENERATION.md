# en_pool_30k — single EN synthetic pool (Article 1, coherent design)

THE English synthetic pool. All subsamples for all EN blocks are drawn from here.

Generated: run_pipeline.py (v2.1.0) from code/framework/
  python run_pipeline.py --language english --unique-texts \
    --category-filter Handwritten --total-images 30000 \
    --output-name en_pool_30k --skip-fonts --verbose

Converted + compiled (CRITICAL: --force-type baseline):
  convert_hf_to_kraken.py --input output_en_pool_30k --output <dir>
  ketos compile -f path --force-type baseline -o <split>_bl.arrow <split>/*.png

Verified before compile: train 23484 lines, 99.0% unique, 75 chars (basic latin, no accents).
1965 articles, 37507 raw lines. 187 Handwritten fonts used (num_fonts in output_en_pool_30k/dataset_info.json; of the 196 downloaded, the pipeline keeps those with real glyphs for every required character of languages/english.json), perturbations OFF (Clean). [Corrected 25/09/2026: this file previously said 159, which is the French count.]
Subsamples produced with subsample_arrow.py (seed 42) from en_synth_30k_train_bl.arrow.

## Normalization

Text is stored in the arrow files as generated. All training and evaluation in this framework use `-u NFD`; always pass the same `-u` option to `ketos train` and `ketos test` (see the main README).
