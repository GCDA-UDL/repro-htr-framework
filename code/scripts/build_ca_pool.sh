#!/bin/bash
# Build the Catalan synthetic pool from the framework output, identically to fr_pool_30k / en_pool_30k:
# HF layout -> Kraken path format, compile baseline arrows per split, subsample 1:1 with CatK12HW train (4152 lines).
set -e
cd ~/workrepo-htr-synthetic-framework
SRC=code/framework/output_ca_pool_30k
DST=data/synthetic/ca_pool_30k
[ -f "$SRC/dataset_info.json" ] || { echo "generation not finished: $SRC/dataset_info.json missing"; exit 1; }
echo "== dataset_info"; python -c "import json; d=json.load(open('$SRC/dataset_info.json')); print({k: d[k] for k in ('version','num_fonts','total_samples','mode','style','perturbations_enabled','splits') if k in d})"
echo "== convert"; python code/scripts/dataset_conversion/convert_hf_to_kraken.py --input "$SRC" --output "$DST"
for s in train validation test; do
  echo "== compile $s: $(ls "$DST/$s"/*.png | wc -l) png"
  ketos compile -f path --force-type baseline -o "$DST/ca_synth_30k_${s}_bl.arrow" "$DST/$s"/*.png 2>&1 | tail -1
done
echo "== subsample 4152 (1:1 with CatK12HW train)"
python code/scripts/subsample_arrow.py "$DST/ca_synth_30k_train_bl.arrow" "$DST/ca_synth_4152_train_bl.arrow" 4152
echo "== verification"
python - <<'PYEOF'
import json, glob, pyarrow as pa
for f in sorted(glob.glob("data/synthetic/ca_pool_30k/*.arrow")):
    r = pa.ipc.open_file(f); t = r.read_all(); m = json.loads(r.schema.metadata[b"lines"])
    texts = []
    for ch in t.column("lines").chunks:
        texts += ch.field("text").to_pylist()
    print(f, "| lines:", t.num_rows, "| unique texts:", len(set(texts)), f"({100*len(set(texts))/max(1,len(texts)):.1f}%)",
          "| chars:", sum(map(len, texts)), "| type:", m.get("type"), "| im_mode:", m.get("im_mode"))
PYEOF
echo "=== BUILD DONE ==="
