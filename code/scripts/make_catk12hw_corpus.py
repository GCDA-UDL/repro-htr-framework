#!/usr/bin/env python3
"""Matched-text corpus for the Article-2 'text' cells: the transcriptions of the CatK12HW TRAINING split
(data/real/catk12hw/catk12hw_train.arrow; the test split is never read), one per line and repeated REP times in a
seeded shuffle, so that build_dataset.py --text-lines --unique-texts renders every occurrence with a different random
font (with --unique-texts one line = one image, and the 80 % training split must hold >= 4152 lines).
Writes code/framework/data/catk12hw_train/catk12hw_train_x<REP>.txt, prints the corpus statistics and the lines that
the framework's character filter for Catalan (--text-lines mode: allowed characters and their upper-case forms) would
reject, with the offending characters.
Usage: python code/scripts/make_catk12hw_corpus.py [REP=2] [SEED=42]
"""
import collections, json, os, random, sys
import pyarrow as pa

os.chdir(os.path.expanduser("~/workrepo-htr-synthetic-framework"))
rep = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
t = pa.ipc.open_file("data/real/catk12hw/catk12hw_train.arrow").read_all()
texts = []
for ch in t.column("lines").chunks:
    texts += ch.field("text").to_pylist()
texts = [" ".join(s.split()) for s in texts if s and s.strip()]
lines = texts * rep
random.Random(seed).shuffle(lines)
out = "code/framework/data/catk12hw_train"
os.makedirs(out, exist_ok=True)
path = f"{out}/catk12hw_train_x{rep}.txt"
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
w = [len(s.split()) for s in texts]
print(f"{path}: {len(lines)} lines = {len(texts)} training transcriptions x {rep} (shuffle seed {seed}); "
      f"{len(set(texts))} distinct texts, {sum(w) / len(w):.2f} words/line (min {min(w)}, max {max(w)}), "
      f"{sum(map(len, texts)) / len(texts):.1f} chars/line, {100 * sum(1 for s in texts if s == s.upper()) / len(texts):.1f}% all-uppercase")
# the framework's character filter (build_dataset._get_allowed_chars_for_language + upper-case forms in --text-lines mode)
allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") | set(" .,;:!?'\"-()[]{}")
cfg = "code/framework/languages/catalan.json"
if os.path.exists(cfg):
    allowed |= set(json.load(open(cfg, encoding="utf-8")).get("required_chars", []))
allowed |= {c.upper() for c in allowed}
bad = [s for s in texts if any(c not in allowed for c in s)]
chars = collections.Counter(c for s in bad for c in s if c not in allowed)
print(f"lines the framework would reject: {len(bad)} of {len(texts)} ({100 * len(bad) / len(texts):.1f}%); offending characters: "
      + ", ".join(f"{c!r} x{n}" for c, n in chars.most_common()))
for s in bad[:8]:
    print("   rejected:", s)
