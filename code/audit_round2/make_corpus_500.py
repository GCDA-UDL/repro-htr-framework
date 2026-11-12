#!/usr/bin/env python3
"""Matched-text corpus restricted to the 500-line real subset (audit round 2, control B1).

Same construction as code/scripts/make_catk12hw_corpus.py, but the only text the synthetic pool may contain is the
transcriptions of data/real/catk12hw/catk12hw_500_train.arrow (the Block C subset, seed 42): one line per transcription,
repeated REP times in a seeded shuffle, so that build_dataset.py --text-lines --unique-texts renders every occurrence
with a different random font. REP=12 gives 6000 lines -> 80 % train split (4800) >= the 4152-line 1:1 subsample.
The validation and test splits are never read. Output: audit_round2/data/text/catk12hw_500_x<REP>/catk12hw_500_x<REP>.txt
Usage: python make_corpus_500.py [REP=12] [SEED=42]
"""
import collections, os, random, sys
import pyarrow as pa

ROOT = os.path.expanduser("~joc13/workrepo-htr-synthetic-framework")
rep = int(sys.argv[1]) if len(sys.argv) > 1 else 12
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
src = f"{ROOT}/data/real/catk12hw/catk12hw_500_train.arrow"
texts = []
with pa.memory_map(src) as m:
    r = pa.ipc.open_file(m)
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        texts += b.column(b.schema.get_field_index("lines")).field("text").to_pylist()
texts = [" ".join(s.split()) for s in texts if s and s.strip()]
assert len(texts) == 500, len(texts)
lines = texts * rep
random.Random(seed).shuffle(lines)
out = f"{ROOT}/audit_round2/data/text/catk12hw_500_x{rep}"
os.makedirs(out, exist_ok=True)
path = f"{out}/catk12hw_500_x{rep}.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
w = [len(s.split()) for s in texts]
up = sum(1 for s in texts if s == s.upper() and any(c.isalpha() for c in s))
print(f"{path}: {len(lines)} lines = {len(texts)} transcriptions of the 500-line subset x {rep} (shuffle seed {seed}); "
      f"{len(set(texts))} distinct texts ({len(set(s.lower() for s in texts))} case-insensitive), "
      f"{sum(w)/len(w):.2f} words/line, {sum(map(len, texts))/len(texts):.1f} chars/line, {100*up/len(texts):.1f}% all-uppercase")
c = collections.Counter(texts)
print("most repeated texts in the subset:", c.most_common(5))
