#!/usr/bin/env python3
"""Write data/synthetic/<pool>/GENERATION.md from dataset_info.json, the compiled arrows, FONTS.txt (when present) and a
paper-colour check of 150 sampled training images. Numbers come from the files themselves, never typed by hand.
Usage: python code/scripts/pool_generation_md.py <pool> <language> <category> <total_images> <subsample_n>
         [--bg-color white|grey|beige] [--bg-type plain|grid|lined] [--desc "..."]
  e.g. python code/scripts/pool_generation_md.py ca_white_30k catalan Handwritten 30000 4152 --bg-color white --bg-type plain
"""
import argparse, collections, datetime, glob, json, os, random
import numpy as np
import pyarrow as pa
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("pool"); ap.add_argument("language"); ap.add_argument("category"); ap.add_argument("total", type=int)
ap.add_argument("n", type=int)
ap.add_argument("--bg-color", default=None); ap.add_argument("--bg-type", default=None); ap.add_argument("--desc", default="")
ap.add_argument("--command", default=None, help="exact generation command (replaces the run_pipeline.py template line)")
a = ap.parse_args()
code = {"catalan": "ca", "spanish": "es", "french": "fr", "english": "en"}[a.language]
os.chdir(os.path.expanduser("~/workrepo-htr-synthetic-framework"))
src, dst = f"code/framework/output_{a.pool}", f"data/synthetic/{a.pool}"
d = json.load(open(f"{src}/dataset_info.json"))
sp = d["splits"]
rows = []
for f in sorted(glob.glob(f"{dst}/*.arrow")):
    r = pa.ipc.open_file(f); t = r.read_all(); m = json.loads(r.schema.metadata[b"lines"])
    texts = []
    for ch in t.column("lines").chunks:
        texts += ch.field("text").to_pylist()
    uniq = 100 * len(set(texts)) / max(1, len(texts))
    rows.append(f"  {os.path.basename(f):34s}: {t.num_rows:6d} lines, {uniq:.1f}% unique texts, {sum(map(len, texts))} characters ({m.get('type')}, {m.get('im_mode')})")
# paper-colour check on the compile sources (median colour of the image borders)
random.seed(0)
files = glob.glob(f"{dst}/train/*.png")
cls = collections.Counter()
for f in random.sample(files, min(150, len(files))):
    arr = np.asarray(Image.open(f).convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    r_, g_, b_ = np.median(border, axis=0)
    cls["white" if min(r_, g_, b_) > 235 else "beige" if r_ - b_ > 15 else "grey" if max(r_, g_, b_) - min(r_, g_, b_) < 12 else "other"] += 1
paper_check = f"{sum(cls.values())} sampled training images -> {dict(cls)}"
fonts_file = f"{dst}/FONTS.txt"
nf = sum(1 for l in open(fonts_file, encoding="utf-8") if l.strip() and not l.startswith("#")) if os.path.exists(fonts_file) else None
fonts_line = (f"FONTS.txt in this folder ({nf} fonts chosen by hand: names, files and licence files)" if nf
              else "the whole category as downloaded and verified by the framework")
bg_opts = (f" --background-color {a.bg_color}" if a.bg_color else "") + (f" --background-type {a.bg_type}" if a.bg_type else "")
if a.bg_color or a.bg_type:
    bg_line = (f"Backgrounds: only the {a.bg_color or 'any colour'} {a.bg_type or 'any type'} paper textures of code/framework/backgrounds/ "
               f"(campaign 2: paper matched to the real data; the framework default mixes 45 textures, see the mixedbg archive).")
else:
    bg_line = ("Backgrounds: the 45 paper backgrounds of code/framework/backgrounds/ (plain / grid / lined x white / grey / beige), "
               "mixed at random (framework default).")
today = datetime.date.today().strftime("%d/%m/%Y")
skipped = a.total - int(d.get("total_samples", 0))
sub = a.pool.rsplit("_", 1)[0]
arrows = "\n".join(rows)
gen_cmd = a.command or (f"python run_pipeline.py --language {a.language} --unique-texts --category-filter {a.category}{bg_opts} \\\n"
                        f"    --total-images {a.total} --output-name {a.pool} --skip-fonts --skip-text --workers 10 --verbose")
md = f"""# {a.pool} — {a.language} synthetic pool, fonts of the category {a.category}

{a.desc}

Generated (log: generation.log in this folder): framework v{d.get('version')}, run from code/framework/
  {gen_cmd}
dataset_info.json: num_fonts {d.get('num_fonts')}, total_samples {d.get('total_samples')} ({skipped} of the {a.total} tasks skipped because the
font lacked a required character), splits train {sp['train']['num_samples']} / validation {sp['validation']['num_samples']} / test {sp['test']['num_samples']}.
Perturbations {'ON' if d.get('perturbations_enabled') else 'OFF (Clean)'}, style {d.get('style')}, mode {d.get('mode')}.
{bg_line}
Paper check ({today}, median border colour): {paper_check}.
Fonts: {fonts_line}.

Converted + compiled {today} with `POOL={a.pool} TAG={a.pool} bash code/scripts/build_pool.sh {code} {a.n}` (log: build.log in
this folder; same convert_hf_to_kraken.py + ketos compile -f path --force-type baseline steps as every pool):
{arrows}
Use with the training scripts: SYN_TR_OVR=data/synthetic/{a.pool}/{sub}_{a.n}_train_bl.arrow
SYN_VAL_OVR=data/synthetic/{a.pool}/{a.pool}_validation_bl.arrow (train_blockA.sh synth-only / curriculum; the campaign-2
defaults of the training scripts point here); Block C curriculum runs take the phase-1 checkpoint through PHASE1_DIR.
Checksums: data/synthetic/POOLS_SHA256.txt.

## Normalization
Text is stored as generated; all training and evaluation use -u NFD, as for every pool.
"""
open(f"{dst}/GENERATION.md", "w", encoding="utf-8").write(md)
print(f"{dst}/GENERATION.md written: num_fonts {d.get('num_fonts')}, {len(rows)} arrows, paper {paper_check}, FONTS.txt {nf}")
