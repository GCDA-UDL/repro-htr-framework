#!/bin/bash
# Audit round 3: synthetic pool with the SAME recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts,
# --text-lines, white plain paper, --perturbations 40/40/20, --unique-texts, 6000 images) from a given text corpus.
# Everything is written under audit_round3/data/synthetic/ (framework output with PNG + metadata.jsonl, Kraken arrows,
# 1:1 subsample of 4152 lines with seed 42, FONTS.txt, GENERATION.md, SHA256.txt). Joaquim's code is used read-only.
# Usage: bash audit_round3/scripts/gen_pool.sh <pool_name> <text_dir> "<description>"
set -eo pipefail
ROOT=/home/joc13/workrepo-htr-synthetic-framework
ENV=/home/joc13/miniconda3/envs/kraken7/bin
cd "$ROOT" || exit 1
POOL=$1; TXT=$(readlink -f "$2"); DESC=$3      # absolute: the generator runs from code/framework
[ -n "$POOL" ] && [ -d "$TXT" ] || { echo "usage: gen_pool.sh <pool_name> <text_dir> <description>"; exit 1; }
cd "$ROOT"
OUTBASE=$ROOT/audit_round3/data/synthetic/output        # build_dataset.py appends _<name>
SRC=${OUTBASE}_${POOL}
DST=$ROOT/audit_round3/data/synthetic/$POOL
SUB=${POOL%_*}; SUB=${SUB}_4152_train_bl.arrow           # e.g. ca_kidstextpert415ext_white_4152_train_bl.arrow
if [ -f "$DST/GENERATION.md" ]; then echo "== $POOL already built"; exit 0; fi
mkdir -p "$DST"
CMD="python build_dataset.py --data-dir $TXT --fonts-dir fonts --language catalan --category-filter KidsCurated --background-color white --background-type plain --total-images 6000 --output-dir $OUTBASE --output-name $POOL --mode lines --style normal --workers 10 --verbose --unique-texts --text-lines --perturbations"
if [ ! -f "$SRC/dataset_info.json" ]; then
  echo "== generating $POOL ($(date -Iseconds)): $CMD"
  (cd code/framework && $ENV/$CMD) > "$DST/generation.log" 2>&1 || { echo "GENERATION FAILED (see $DST/generation.log)"; tail -n 20 "$DST/generation.log"; exit 1; }
fi
echo "== dataset_info"; $ENV/python -c "import json; d=json.load(open('$SRC/dataset_info.json')); print({k: d[k] for k in ('num_fonts','total_samples','perturbations_enabled','splits') if k in d})"
if [ ! -f "$DST/${POOL}_train_bl.arrow" ]; then
  $ENV/python code/scripts/dataset_conversion/convert_hf_to_kraken.py --input "$SRC" --output "$DST"
  for s in train validation test; do
    echo "== compile $s: $(ls "$DST/$s"/*.png | wc -l) png"
    $ENV/ketos compile -f path --force-type baseline -o "$DST/${POOL}_${s}_bl.arrow" "$DST/$s"/*.png 2>&1 | tail -n 1
  done
fi
[ -f "$DST/$SUB" ] || $ENV/python code/scripts/subsample_arrow.py "$DST/${POOL}_train_bl.arrow" "$DST/$SUB" 4152
cp data/synthetic/ca_kidstextpert_white_6k/FONTS.txt "$DST/FONTS.txt"
for s in train validation test; do rm -rf "$DST/$s"; done      # PNG copies: the framework output keeps them
(cd "$ROOT" && sha256sum audit_round3/data/synthetic/$POOL/*.arrow) > "$DST/SHA256.txt"
$ENV/python - "$SRC" "$DST" "$CMD" "$DESC" "$SUB" <<'PYEOF'
import datetime, glob, json, sys, pyarrow as pa
src, dst, cmd, desc, sub = sys.argv[1:6]
d = json.load(open(f"{src}/dataset_info.json"))
rows = []
for f in sorted(glob.glob(f"{dst}/*.arrow")):
    with pa.memory_map(f) as m:
        r = pa.ipc.open_file(m); t = []
        for i in range(r.num_record_batches):
            b = r.get_batch(i); t += b.column(b.schema.get_field_index("lines")).field("text").to_pylist()
    rows.append(f"  {f.split('/')[-1]}: {len(t)} lines, {len(set(t))} distinct texts, {sum(map(len, t))} characters")
open(f"{dst}/GENERATION.md", "w", encoding="utf-8").write(f"""# {dst.split('/')[-1]} — audit round 3 ({datetime.date.today()})

{desc}

Same recipe as data/synthetic/ca_kidstextpert_white_6k (KidsCurated fonts, --text-lines, white plain paper,
--perturbations clean/degraded/severe 40/40/20, --unique-texts, 6000 images); only the text corpus changes.
Generated from code/framework/ (uncommitted --text-lines patch, same as the other matched-text pools):
  {cmd}
dataset_info.json: num_fonts {d.get('num_fonts')}, total_samples {d.get('total_samples')}, splits {d.get('splits')}.
Framework output (PNG + metadata.jsonl kept): {src}
Arrows:
""" + "\n".join(rows) + f"""
1:1 subsample {sub} (code/scripts/subsample_arrow.py, seed 42). Checksums: SHA256.txt. Fonts: FONTS.txt.
Normalization: train and test with -u NFD, as every pool.
""")
PYEOF
echo "=== POOL $POOL DONE ($(date -Iseconds)) ==="
