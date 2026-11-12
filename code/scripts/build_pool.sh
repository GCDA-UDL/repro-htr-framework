#!/bin/bash
# Build a synthetic pool from the framework output, identically for every pool (same steps as build_ca_pool.sh):
# HF layout -> Kraken path format, compile baseline arrows per split, optional subsample of the train split (seed 42).
# Usage: [POOL=<framework output name>] [TAG=<arrow prefix>] bash build_pool.sh <lang> [subsample_n]
#   POOL=ca_school_6k TAG=ca_school_6k bash build_pool.sh ca 4152
#                                  -> data/synthetic/ca_school_6k/ca_school_6k_{train,validation,test}_bl.arrow
#                                     + ca_school_4152_train_bl.arrow (subsample name = TAG without its last token + _<n>)
set -eo pipefail
cd ~/workrepo-htr-synthetic-framework
L=$1; N=$2
[ -n "$L" ] || { echo "usage: [POOL=..] [TAG=..] bash build_pool.sh <lang> [subsample_n]"; exit 1; }
POOL=${POOL:-${L}_pool_30k}; TAG=${TAG:-${L}_synth_30k}; SUB=${TAG%_*}
SRC=code/framework/output_${POOL}
DST=data/synthetic/${POOL}
[ -f "$SRC/dataset_info.json" ] || { echo "generation not finished: $SRC/dataset_info.json missing"; exit 1; }
echo "== dataset_info"; python -c "import json; d=json.load(open('$SRC/dataset_info.json')); print({k: d[k] for k in ('version','num_fonts','total_samples','mode','style','perturbations_enabled','splits') if k in d})"
if [ ! -f "$DST/${TAG}_train_bl.arrow" ]; then
  echo "== convert"; python code/scripts/dataset_conversion/convert_hf_to_kraken.py --input "$SRC" --output "$DST"
  for s in train validation test; do
    echo "== compile $s: $(ls "$DST/$s"/*.png | wc -l) png"
    ketos compile -f path --force-type baseline -o "$DST/${TAG}_${s}_bl.arrow" "$DST/$s"/*.png 2>&1 | tail -n 1
  done
else
  echo "== arrows already built, skipping convert/compile"
fi
if [ -n "$N" ]; then
  echo "== subsample $N (seed 42) -> ${SUB}_${N}_train_bl.arrow"
  python code/scripts/subsample_arrow.py "$DST/${TAG}_train_bl.arrow" "$DST/${SUB}_${N}_train_bl.arrow" "$N"
fi
echo "== verification"
python - "$DST" <<'PYEOF'
import json, glob, sys, pyarrow as pa
for f in sorted(glob.glob(sys.argv[1] + "/*.arrow")):
    r = pa.ipc.open_file(f); t = r.read_all(); m = json.loads(r.schema.metadata[b"lines"])
    texts = []
    for ch in t.column("lines").chunks:
        texts += ch.field("text").to_pylist()
    print(f, "| lines:", t.num_rows, "| unique texts:", len(set(texts)), f"({100*len(set(texts))/max(1,len(texts)):.1f}%)",
          "| chars:", sum(map(len, texts)), "| type:", m.get("type"), "| im_mode:", m.get("im_mode"))
PYEOF
echo "=== BUILD DONE ==="
