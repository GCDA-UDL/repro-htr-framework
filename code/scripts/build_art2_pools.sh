#!/bin/bash
# Article 2 content cells: generate the synthetic pools, convert + compile them and document them, one after another.
# Generation calls code/framework/build_dataset.py directly with the options run_pipeline.py passes (lines mode, normal
# style, unique texts, white plain / beige lined paper) plus --fonts-dir (curated folders in fonts_sel/) and --text-lines
# (matched-text corpus) where needed. Then build_pool.sh (same convert + ketos compile + 1:1 subsample steps as every
# pool), FONTS.txt, GENERATION.md (exact command) and POOLS_SHA256.txt. A pool whose GENERATION.md exists is skipped;
# an output folder without GENERATION.md stops the pool (remove or rename it by hand). CPU only; never touches the queue.
# Usage: bash code/scripts/build_art2_pools.sh [pool ...]     (default: all 7 pools; logs in data/synthetic/<pool>/)
set -o pipefail
cd ~/workrepo-htr-synthetic-framework || exit 1
FW=code/framework
GEN="--mode lines --style normal --workers 10 --verbose --unique-texts"
ONLY="$*"

pool() {  # pool <name> <language> <code> <category> <fonts-dir> <data-dir> <text-lines|-> <bg-color> <bg-type> <total> <subsample n> <desc>
  local P=$1 L=$2 C=$3 CAT=$4 FD=$5 DD=$6 TL=$7 BGC=$8 BGT=$9 TOT=${10} N=${11} DESC=${12} EXTRA=${13} D=data/synthetic/$1
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $P "; then return 0; fi
  if [ -f "$D/GENERATION.md" ]; then echo "== $P: already built (GENERATION.md exists)"; return 0; fi
  if [ -d "$FW/output_$P" ]; then echo "== $P: $FW/output_$P exists but the pool is not documented - remove or rename it first"; return 1; fi
  [ -d "$FW/$FD/$CAT" ] || { echo "== $P: font folder $FW/$FD/$CAT missing"; return 1; }
  [ -d "$FW/$DD" ] || { echo "== $P: text folder $FW/$DD missing"; return 1; }
  mkdir -p "$D"
  local CMD="python build_dataset.py --data-dir $DD --fonts-dir $FD --language $L --category-filter $CAT --background-color $BGC --background-type $BGT --total-images $TOT --output-name $P $GEN"
  [ "$TL" = "text-lines" ] && CMD="$CMD --text-lines"
  [ -n "$EXTRA" ] && CMD="$CMD $EXTRA"
  echo "== $P: generating ($(date -Iseconds)): $CMD"
  (cd $FW && $CMD) > "$D/generation.log" 2>&1
  grep -E "\[OK\]|\[INFO\]|\[ERROR\]|Error|Traceback|skipped|Total" "$D/generation.log" | tail -n 20
  [ -f "$FW/output_$P/dataset_info.json" ] || { echo "== $P: GENERATION FAILED (no dataset_info.json, see $D/generation.log)"; return 1; }
  echo "== $P: converting + compiling"
  POOL=$P TAG=$P bash code/scripts/build_pool.sh $C $N > "$D/build.log" 2>&1
  grep -E "^==|lines:|DONE" "$D/build.log"
  grep -q "=== BUILD DONE ===" "$D/build.log" || { echo "== $P: BUILD FAILED (see $D/build.log)"; return 1; }
  if [ -f "$FW/$FD/$CAT/FONTS.txt" ]; then
    cp "$FW/$FD/$CAT/FONTS.txt" "$D/FONTS.txt"
  else
    { echo "# $CAT: $(ls -d $FW/$FD/$CAT/*/ | wc -l) font folders of $FW/$FD/$CAT (one file per folder is used by the generator)"; echo "# name	files"
      for f in $FW/$FD/$CAT/*/; do echo "$(basename $f)	$(ls "$f" | tr '\n' ' ')"; done; } > "$D/FONTS.txt"
  fi
  python code/scripts/pool_generation_md.py $P $L $CAT $TOT $N --bg-color $BGC --bg-type $BGT --command "$CMD" --desc "$DESC" || { echo "== $P: GENERATION.md FAILED"; return 1; }
  if head -n 1 data/synthetic/POOLS_SHA256.txt | grep -q "data/synthetic/"; then sha256sum data/synthetic/$P/*.arrow >> data/synthetic/POOLS_SHA256.txt; else (cd data/synthetic && sha256sum $P/*.arrow >> POOLS_SHA256.txt); fi
  echo "== $P: DONE ($(date -Iseconds)); $(grep -c "" "$D/FONTS.txt") lines in FONTS.txt"
}

pool ca_kids_white_6k       catalan ca KidsCurated      fonts     data/wikipedia_ca   -          white plain 6000  4152  "Article 2 typography cell 'kids': the 25 child-like fonts chosen by hand (fonts/KidsCurated), generic Wikipedia text, white plain paper (campaign-2 standard)."
pool ca_text_white_6k       catalan ca Handwritten      fonts     data/catk12hw_train text-lines white plain 6000  4152  "Article 2 text cell 'text': Handwritten fonts (as the base pool ca_white_30k) with the matched-text corpus (CatK12HW training transcriptions x2, --text-lines), white plain paper."
pool ca_kidstext_white_6k   catalan ca KidsCurated      fonts     data/catk12hw_train text-lines white plain 6000  4152  "Article 2 combined cell 'kidstext': KidsCurated fonts with the matched-text corpus (CatK12HW training transcriptions x2, --text-lines), white plain paper."
pool ca_school_white_6k     catalan ca School_ca        fonts_sel data/wikipedia_ca   -          white plain 6000  4152  "Article 2 typography cell 'school': every DaFont School font that passed the Catalan glyph check (fonts_sel/School_ca), generic Wikipedia text, white plain paper."
pool ca_schooltext_white_6k catalan ca School_ca        fonts_sel data/catk12hw_train text-lines white plain 6000  4152  "Article 2 combined cell 'schooltext': School fonts with the matched-text corpus (--text-lines), white plain paper."
pool ca_hw25_white_6k       catalan ca Handwritten25_ca fonts_sel data/wikipedia_ca   -          white plain 6000  4152  "Article 2 font-count control 'hw25': 25 Handwritten fonts sampled at random (seed 42, fonts_sel/Handwritten25_ca/FONTS.txt), generic Wikipedia text, white plain paper."
# Volume cell (2026-10-15): 5x the kidstext pool, same fonts and text source; corpus replicated x8 so --text-lines covers 30k images.
pool ca_kidstext_white_30k  catalan ca KidsCurated      fonts     data/catk12hw_train_x8 text-lines white plain 30000 4152 "Article 2 volume cell 'kidstext-30k': KidsCurated fonts with the matched-text corpus (CatK12HW training transcriptions, replicated x8 for --text-lines), white plain paper; 5x ca_kidstext_white_6k, to test the effect of synthetic volume (TrOCR phase 1, Kraken curriculum)."
# Quality cell (2026-10-16): as ca_kidstext_white_6k plus --perturbations (default distribution clean,degraded,severe = 40,40,20).
pool ca_kidstextpert_white_6k catalan ca KidsCurated      fonts     data/catk12hw_train text-lines white plain 6000  4152  "Article 2 quality cell 'kidstext-pert': KidsCurated fonts with the matched-text corpus (CatK12HW training transcriptions x2, --text-lines), white plain paper, --perturbations with the default quality distribution (clean 40, degraded 40, severe 20)." "--perturbations"
# From-scratch arm, quality cell (2026-10-18): the generic base pool ca_white_30k (Handwritten fonts x Wikipedia-ca, --unique-texts, white plain) plus --perturbations (default clean,degraded,severe = 40,40,20).
pool ca_whitepert_30k      catalan ca Handwritten      fonts     data/wikipedia_ca   -          white plain 30000 4152  "From-scratch arm quality cell 'whitepert30k': as the generic base pool ca_white_30k (Handwritten fonts, Wikipedia-ca unique texts, white plain paper) with --perturbations (default quality distribution clean 40, degraded 40, severe 20)." "--perturbations"
echo "=== ART2 POOLS DONE ($(date -Iseconds)) ==="
