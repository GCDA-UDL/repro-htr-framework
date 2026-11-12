#!/usr/bin/env python3
"""Error analysis of the Catalan Block C runs at 500 real lines (Article 2): aggregates the ketos test full reports
(eval_full.log) of every seed of each group and compares the groups. Same report parser as parse_ketos_reports.py.
Semantics (verified in section 4.1): Insertions = characters the model omits; Deletions = characters it adds.
Usage: python parse_blockc_reports.py [TOPN]   -> prints the summary and writes results/error_analysis_blockC500.txt"""
import glob, os, re, sys, unicodedata
from collections import Counter

TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
GROUPS = [  # label, folder glob
    ("real-only", "models/blockC/ca_real-only_500_seed*"),
    ("real-only + augment", "models/blockC/ca_real-only_500_aug_seed*"),
    ("kidstext", "models/blockC/ca_curriculum_500_kidstext_seed*"),
    ("kidstext-pert", "models/blockC/ca_curriculum_500_kidstextpert_seed*"),
    ("kidstext-pert + augment", "models/blockC/ca_curriculum_500_kidstextpert-aug_seed*"),
]

def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "Traceback" in txt or "=== report" not in txt:
        return None
    m = re.search(r"(\d+)\s+Characters\s+(\d+)\s+Errors", txt)
    if not m:
        return None
    r = dict(chars=int(m.group(1)), errors=int(m.group(2)))
    m = re.search(r"(\d+)\s+Insertions\s+(\d+)\s+Deletions\s+(\d+)\s+Substitutions", txt)
    r["ins"], r["dele"], r["sub"] = (int(m.group(i)) for i in (1, 2, 3)) if m else (0, 0, 0)
    r["conf"] = Counter({(m.group(2) or "∅", m.group(3) or "∅"): int(m.group(1))
                         for m in re.finditer(r"^(\d+)\s+\{ ?(.*?) ?\} - \{ ?(.*?) ?\}\s*$", txt, re.M)})
    return r

def is_comb(s): return s.startswith("COMBINING") or any(0x300 <= ord(c) <= 0x36F for c in s)   # Kraken prints Unicode names for non-printable chars
def is_space(s): return s in (" ", "SPACE")
def show(a): return "␣" if is_space(a) else a.replace("COMBINING ", "◌")

out = []
def P(s=""):
    out.append(s); print(s)

agg = {}
for label, pat in GROUPS:
    runs = sorted(glob.glob(pat)); reps = []
    for run in runs:
        f = os.path.join(run, "eval_full.log")
        r = parse(f) if os.path.exists(f) else None
        if r: reps.append((os.path.basename(run), r))
        else: P(f"  [skip] {run}: no usable report")
    if not reps:
        continue
    n = len(reps); chars = sum(r["chars"] for _, r in reps); errors = sum(r["errors"] for _, r in reps)
    ins, dele, sub = (sum(r[k] for _, r in reps) for k in ("ins", "dele", "sub"))
    conf = Counter()
    for _, r in reps: conf.update(r["conf"])
    case = sum(c for (a, b), c in conf.items() if a != "∅" and b != "∅" and a.lower() == b.lower() and a != b)
    diac = sum(c for (a, b), c in conf.items() if is_comb(a) or is_comb(b))
    space = sum(c for (a, b), c in conf.items() if is_space(a) or is_space(b))
    agg[label] = dict(n=n, chars=chars, errors=errors, ins=ins, dele=dele, sub=sub, conf=conf, case=case, diac=diac, space=space)

P("=== Block C, ca, 500 real lines: errors pooled over seeds (ketos test -u NFD reports; CER = sum errors / sum characters) ===")
P(f"{'group':26s} {'n':>2s} {'chars':>7s} {'errors':>7s} {'CER':>6s} {'omitted':>9s} {'added':>9s} {'substit.':>10s} {'case-only':>10s} {'diacritic':>10s} {'space':>8s}")
for label, a in agg.items():
    e = a["errors"]; pc = lambda v: f"{v} ({100*v/e:.0f}%)"
    P(f"{label:26s} {a['n']:2d} {a['chars']:7d} {e:7d} {100*e/a['chars']:6.2f} {pc(a['ins']):>9s} {pc(a['dele']):>9s} {pc(a['sub']):>10s} {pc(a['case']):>10s} {pc(a['diac']):>10s} {pc(a['space']):>8s}")
P("omitted/added/substituted = Insertions/Deletions/Substitutions of the report; case-only = substitutions between the same letter in the other case;")
P("diacritic = confusions involving a combining mark (NFD); space = confusions that omit or add a space. Percentages of the group's errors.")
for label, a in agg.items():
    P(f"\n--- top {TOPN} confusions, {label} (correct -> output, count over {a['n']} seeds) ---")
    P("   ".join(f"{show(x)}→{show(y)} {c}" for (x, y), c in a["conf"].most_common(TOPN)))
if "real-only" in agg:
    base = agg["real-only"]["conf"]
    for label in ("kidstext", "kidstext-pert", "kidstext-pert + augment", "real-only + augment"):
        if label not in agg: continue
        d = Counter({k: agg[label]["conf"].get(k, 0) - base.get(k, 0) for k in set(base) | set(agg[label]["conf"])})
        dec = sorted(d.items(), key=lambda kv: kv[1])[:TOPN]; inc = sorted(d.items(), key=lambda kv: -kv[1])[:6]
        P(f"\n--- {label} vs real-only: confusions that decrease most (count difference over 6 seeds) ---")
        P("   ".join(f"{show(x)}→{show(y)} {v:+d}" for (x, y), v in dec if v < 0))
        P(f"--- {label} vs real-only: confusions that increase most ---")
        P("   ".join(f"{show(x)}→{show(y)} {v:+d}" for (x, y), v in inc if v > 0))
os.makedirs("results", exist_ok=True)
open("results/error_analysis_blockC500.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n-> results/error_analysis_blockC500.txt")
