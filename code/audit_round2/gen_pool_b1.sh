#!/bin/bash
# Audit round 2, control B1: synthetic pool with the SAME recipe as ca_kidstextpert_white_6k (KidsCurated fonts,
# --text-lines, white plain paper, --perturbations 40/40/20, 6000 images) but whose text corpus is restricted to the
# transcriptions of the 500-line real subset (audit_round2/data/text/catk12hw_500_x12). Everything is written under
# audit_round2/ (framework output, Kraken arrows, 1:1 subsample of 4152 lines with seed 42, FONTS.txt, GENERATION.md,
# SHA256). Joaquim's code is used read-only (build_dataset.py, convert_hf_to_kraken.py, subsample_arrow.py).
# Usage: bash audit_round2/scripts/gen_pool_b1.sh      (CPU only, ~10 workers)
set -eo pipefail
ROOT=/home/joc13/workrepo-htr-synthetic-framework
ENV=/home/joc13/miniconda3/envs/kraken7/bin
cd "$ROOT"
POOL=ca_kidstextpert500_white_6k
TXT=$ROOT/audit_round2/data/text/catk12hw_500_x12
OUTBASE=$ROOT/audit_round2/data/synthetic/output        # build_dataset.py appends _<name>
SRC=${OUTBASE}_${POOL}
DST=$ROOT/audit_round2/data/synthetic/$POOL
[ -f "$TXT/catk12hw_500_x12.txt" ] || { echo "corpus missing: run make_corpus_500.py first"; exit 1; }
if [ -f "$DST/GENERATION.md" ]; then echo "== $POOL already built"; exit 0; fi
mkdir -p "$DST"
CMD="python build_dataset.py --data-dir $TXT --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-dir $OUTBASE --output-name $POOL --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations"
if [ ! -f "$SRC/dataset_info.json" ]; then
  echo "== generating ($(date -Iseconds)): $CMD"
  (cd code/framework && $ENV/$CMD) > "$DST/generation.log" 2>&1 || { echo "GENERATION FAILED (see $DST/generation.log)"; tail -n 20 "$DST/generation.log"; exit 1; }
  grep -E "\[OK\]|\[INFO\]|\[ERROR\]|Error|Traceback|skipped|Total" "$DST/generation.log" | tail -n 15
fi
[ -f "$SRC/dataset_info.json" ] || { echo "no dataset_info.json"; exit 1; }
echo "== dataset_info"; $ENV/python -c "import json; d=json.load(open('$SRC/dataset_info.json')); print({k: d[k] for k in ('version','num_fonts','total_samples','mode','style','perturbations_enabled','splits') if k in d})"
if [ ! -f "$DST/${POOL}_train_bl.arrow" ]; then
  echo "== convert"; $ENV/python code/scripts/dataset_conversion/convert_hf_to_kraken.py --input "$SRC" --output "$DST"
  for s in train validation test; do
    echo "== compile $s: $(ls "$DST/$s"/*.png | wc -l) png"
    $ENV/ketos compile -f path --force-type baseline -o "$DST/${POOL}_${s}_bl.arrow" "$DST/$s"/*.png 2>&1 | tail -n 1
  done
fi
[ -f "$DST/ca_kidstextpert500_white_4152_train_bl.arrow" ] || $ENV/python code/scripts/subsample_arrow.py "$DST/${POOL}_train_bl.arrow" "$DST/ca_kidstextpert500_white_4152_train_bl.arrow" 4152
cp data/synthetic/ca_kidstextpert_white_6k/FONTS.txt "$DST/FONTS.txt"
echo "== verification"
$ENV/python - "$DST" <<'PYEOF'
import glob, json, sys, pyarrow as pa
for f in sorted(glob.glob(sys.argv[1] + "/*.arrow")):
    r = pa.ipc.open_file(f); t = r.read_all(); m = json.loads(r.schema.metadata[b"lines"])
    texts = []
    for ch in t.column("lines").chunks:
        texts += ch.field("text").to_pylist()
    print(f.split('/')[-1], "| lines:", t.num_rows, "| unique texts:", len(set(texts)), f"({100*len(set(texts))/max(1,len(texts)):.1f}%)", "| chars:", sum(map(len, texts)), "| type:", m.get("type"))
PYEOF
(cd "$ROOT" && sha256sum audit_round2/data/synthetic/$POOL/*.arrow) > "$DST/SHA256.txt"
$ENV/python - "$SRC" "$DST" "$CMD" <<'PYEOF'
import json, sys, datetime
src, dst, cmd = sys.argv[1:4]
d = json.load(open(f"{src}/dataset_info.json"))
open(f"{dst}/GENERATION.md", "w", encoding="utf-8").write(f"""# ca_kidstextpert500_white_6k — audit round 2, control B1 ({datetime.date.today()})

Same recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts, matched text with --text-lines, white plain
paper, --perturbations clean/degraded/severe 40/40/20, --unique-texts, 6000 images) with ONE change: the text corpus is
restricted to the 500 transcriptions of the Block C real subset (data/real/catk12hw/catk12hw_500_train.arrow, seed 42),
repeated x12 in a seeded shuffle (audit_round2/scripts/make_corpus_500.py). No other training, validation or test
transcription is used. Purpose: separate the visual effect of the matched synthetic data from the extra text
supervision that the full-train corpus provides in the "500 real lines" scenario.

Generated from code/framework/ (uncommitted --text-lines patch, same as the other matched-text pools):
  {cmd}
dataset_info.json: num_fonts {d.get('num_fonts')}, total_samples {d.get('total_samples')}, splits {d.get('splits')}.
Framework output (PNG + metadata.jsonl kept): {src}
Arrows: {dst}/*.arrow; 1:1 subsample ca_kidstextpert500_white_4152_train_bl.arrow (code/scripts/subsample_arrow.py, seed 42).
Checksums: SHA256.txt. Fonts: FONTS.txt (copied from the kidstextpert pool; same KidsCurated folder).
Normalization: train and test with -u NFD, as every pool.
""")
PYEOF
echo "=== POOL B1 DONE ($(date -Iseconds)) ==="
