#!/usr/bin/env python3
"""Font-count control cell (Article 2, 'hw25'): a seeded random subset of the Handwritten fonts compatible with Catalan.
The candidates are the fonts that the base pool actually used when code/framework/output_<base>/dataset_info.json lists
them; otherwise every folder of code/framework/fonts/Handwritten whose font file covers the Catalan required characters
(checked with fontTools). N folders are sampled with the seed and copied to code/framework/fonts_sel/<dest>/ with a
FONTS.txt (names and files). Idempotent: an existing destination is left untouched.
Usage: python code/scripts/select_fonts_subset.py [N=25] [SEED=42] [--base ca_white_30k] [--dest Handwritten25_ca]
"""
import argparse, glob, json, os, random, shutil

os.chdir(os.path.expanduser("~/workrepo-htr-synthetic-framework/code/framework"))
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=25)
ap.add_argument("seed", nargs="?", type=int, default=42)
ap.add_argument("--base", default="ca_white_30k")
ap.add_argument("--dest", default="Handwritten25_ca")
ap.add_argument("--category", default="Handwritten")
ap.add_argument("--exclude", default="", help="comma-separated font folders never sampled (e.g. unicase fonts)")
a = ap.parse_args()
dest = os.path.join("fonts_sel", a.dest)
if os.path.isdir(dest) and os.listdir(dest):
    print(f"{dest} already exists with {len([d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))])} font folders - nothing done")
    raise SystemExit
folders = sorted(d for d in os.listdir(os.path.join("fonts", a.category)) if os.path.isdir(os.path.join("fonts", a.category, d)))
info_path = f"output_{a.base}/dataset_info.json"
names, source = [], ""
if os.path.exists(info_path):
    info = json.load(open(info_path, encoding="utf-8"))
    print("dataset_info keys:", ", ".join(f"{k}={type(v).__name__}" for k, v in info.items()))
    for key in ("fonts", "fonts_used", "font_list", "font_names", "fonts_list"):
        v = info.get(key)
        if isinstance(v, list) and v:
            cand = [x if isinstance(x, str) else (x.get("name") or x.get("folder") or x.get("font") or "") for x in v]
            cand = [os.path.basename(os.path.dirname(c)) if "/" in c and c.lower().endswith((".ttf", ".otf")) else c for c in cand]
            names = sorted(set(c for c in cand if c in folders))
            if names:
                source = f"{len(names)} fonts listed in {info_path} ({key})"
                break
if not names:
    req = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cfg = "languages/catalan.json"
    if os.path.exists(cfg):
        req |= set(json.load(open(cfg, encoding="utf-8")).get("required_chars", []))
    try:
        from fontTools.ttLib import TTFont
        for d in folders:
            files = sorted(glob.glob(os.path.join("fonts", a.category, d, "*.[to]tf")) + glob.glob(os.path.join("fonts", a.category, d, "*.[TO]TF")))
            if not files:
                continue
            try:
                cmap = TTFont(files[0], fontNumber=0).getBestCmap() or {}
            except Exception:
                continue
            if all(ord(c) in cmap for c in req):
                names.append(d)
        source = f"{len(names)} of {len(folders)} fonts/{a.category} folders cover the Catalan required characters (fontTools check)"
    except ImportError:
        names = folders
        source = f"all {len(folders)} fonts/{a.category} folders (fontTools not available: no glyph check)"
excl = [x for x in a.exclude.split(",") if x]
if excl:
    names = [n for n in names if n not in excl]
    source += f"; {len(excl)} excluded: {', '.join(excl)}"
print("candidates:", source)
assert len(names) >= a.n, f"only {len(names)} candidates for {a.n} fonts"
chosen = sorted(random.Random(a.seed).sample(names, a.n))
os.makedirs(dest, exist_ok=True)
lines = [f"# {a.dest}: {a.n} fonts sampled with seed {a.seed} from {source}", "# name\tfiles"]
for d in chosen:
    shutil.copytree(os.path.join("fonts", a.category, d), os.path.join(dest, d), dirs_exist_ok=True)
    lines.append(d + "\t" + ", ".join(sorted(os.listdir(os.path.join(dest, d)))))
open(os.path.join(dest, "FONTS.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{dest}: {a.n} font folders copied; FONTS.txt written")
print("chosen:", ", ".join(chosen))
