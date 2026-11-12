#!/usr/bin/env python3
"""Audit round 3, control B9: text corpus with the SAME structure as the S1 corpus of round 2 (415 distinct texts,
each repeated as often as in the 500-line subset, x12 -> 6000 lines) but made of training transcriptions that are
NOT in the 500-line subset (case-insensitive), so that it adds new in-domain text at equal textual diversity.

  S1      = texts of the 500 labelled lines           (no extra text, low diversity)       -- audit_round2
  S2-415  = 415 other training texts, same profile    (extra text,  low diversity)         -- this corpus
  S2      = all training transcriptions x2             (extra text,  high diversity)        -- ca_kidstextpert_white_6k

S2-415 - S1 isolates the extra text at equal diversity; S2 - S2-415 isolates diversity. Validation and test
transcriptions are never read. Output: audit_round3/data/text/catk12hw_415ext_x12/catk12hw_415ext_x12.txt + stats.
Usage: python make_corpus_415ext.py [SEED=42]
"""
import collections, json, os, random, sys
import pyarrow as pa

ROOT = "/home/joc13/workrepo-htr-synthetic-framework"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42


def texts(p):
    out = []
    with pa.memory_map(p) as m:
        r = pa.ipc.open_file(m)
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            out += b.column(b.schema.get_field_index("lines")).field("text").to_pylist()
    return [" ".join(s.split()) for s in out if s and s.strip()]


train = texts(f"{ROOT}/data/real/catk12hw/catk12hw_train.arrow")
sub = texts(f"{ROOT}/data/real/catk12hw/catk12hw_500_train.arrow")
test = texts(f"{ROOT}/data/real/catk12hw/catk12hw_test.arrow")
assert len(sub) == 500
profile = sorted(collections.Counter(sub).values(), reverse=True)          # 415 counts summing to 500
sub_ci = {s.lower() for s in sub}
eligible = sorted({t for t in train if t.lower() not in sub_ci})
rng = random.Random(seed)
chosen = rng.sample(eligible, len(profile))
counts = profile[:]
rng.shuffle(counts)
lines = [t for t, k in zip(chosen, counts) for _ in range(12 * k)]
rng.shuffle(lines)
out = f"{ROOT}/audit_round3/data/text/catk12hw_415ext_x12"
os.makedirs(out, exist_ok=True)
path = f"{out}/catk12hw_415ext_x12.txt"
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def stats(name, corpus_texts):
    d = set(corpus_texts); dci = {t.lower() for t in d}
    up = sum(1 for t in corpus_texts if t == t.upper() and any(c.isalpha() for c in t))
    cov = 100 * sum(t in d for t in test) / len(test); cov_ci = 100 * sum(t.lower() in dci for t in test) / len(test)
    voc = {w for t in d for w in t.split()}; tok = [w for t in test for w in t.split()]
    return {"corpus": name, "lines": len(corpus_texts), "distinct": len(d), "distinct_ci": len(dci),
            "mean_chars": round(sum(map(len, corpus_texts)) / len(corpus_texts), 2),
            "all_upper_pct": round(100 * up / len(corpus_texts), 1),
            "test_lines_exact_text_pct": round(cov, 1), "test_lines_exact_text_ci_pct": round(cov_ci, 1),
            "test_tokens_in_vocab_pct": round(100 * sum(w in voc for w in tok) / len(tok), 1)}


s1 = [t for t in sub for _ in range(12)]
both = sorted(set(sub) | set(chosen))
# second corpus (B9b): EVERY training line whose text (case-insensitive) is not in the 500-line subset, with its natural
# frequency, replicated so that the corpus has >= 8304 lines like the S2 corpus (4152 x2): high diversity, little extra
# test coverage (the frequent phrases are already in the 500 lines).
ext_lines = [t for t in train if t.lower() not in sub_ci]
rep_x = -(-8304 // len(ext_lines))
extall = ext_lines * rep_x
rng2 = random.Random(seed)
rng2.shuffle(extall)
out2 = f"{ROOT}/audit_round3/data/text/catk12hw_extall_x{rep_x}"
os.makedirs(out2, exist_ok=True)
open(f"{out2}/catk12hw_extall_x{rep_x}.txt", "w", encoding="utf-8").write("\n".join(extall) + "\n")
s2 = [t for t in train for _ in range(2)]
rep = {"seed": seed, "eligible_texts": len(eligible), "S1": stats("S1 (500 subset x12)", s1),
       "S2-415ext": stats("S2-415ext", lines),
       "500 real + S2-415ext (texts seen by the model)": stats("union", both),
       "S2-extall": stats(f"S2-extall ({len(ext_lines)} training lines x{rep_x})", extall),
       "500 real + S2-extall": stats("union", sorted(set(sub) | set(ext_lines))),
       "S2 (all training transcriptions x2)": stats("S2", s2)}
json.dump(rep, open(f"{out}/corpus_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
print(f"written {path}: {len(lines)} lines")
