#!/usr/bin/env python3
"""Parse ketos test full reports (Block A) into a comparative error-analysis summary."""
import re, sys, os
ROOT = "models/blockA"
PROTS = ["baseline", "synth-only", "real-only", "mix", "curriculum", "curriculum-LR"]
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "Traceback" in txt or "=== report" not in txt:
        return None
    m = re.search(r"(\d+)\s+Characters\s+(\d+)\s+Errors\s+([\d.]+)%\s+Character Accuracy\s+([\d.]+)%\s+Character Accuracy \(Case-insensitive\)\s+([\d.]+)%\s+Word Accuracy", txt)
    if not m:
        return None
    r = dict(chars=int(m.group(1)), errors=int(m.group(2)), ca=float(m.group(3)), ca_ci=float(m.group(4)), wa=float(m.group(5)))
    m = re.search(r"(\d+)\s+Insertions\s+(\d+)\s+Deletions\s+(\d+)\s+Substitutions", txt)
    r["ins"], r["dele"], r["sub"] = (int(m.group(i)) for i in (1, 2, 3)) if m else (0, 0, 0)
    r["scripts"] = {m.group(4): float(m.group(3)) for m in re.finditer(r"^(\d+)\s+(\d+)\s+([\d.]+)%\s+(\w+)\s*$", txt, re.M)}
    r["conf"] = [(int(m.group(1)), m.group(2) or "∅", m.group(3) or "∅")
                 for m in re.finditer(r"^(\d+)\s+\{ ?(.*?) ?\} - \{ ?(.*?) ?\}\s*$", txt, re.M)]
    return r

for L in ("fr", "en"):
    print(f"\n##### {L.upper()} #####")
    print(f"{'protocol':<14}{'CER-cs':>7}{'WER-cs':>8}{'errors':>8}{'ins':>6}{'del':>6}{'sub':>6}   Latin / Common / Inherited (%right)")
    rows = {}
    for P in PROTS:
        f = f"{ROOT}/{L}_{P}/eval_full.log"
        r = parse(f) if os.path.exists(f) else None
        if not r:
            print(f"{P:<14}   (no valid report)"); continue
        rows[P] = r
        sc = " / ".join(f"{r['scripts'].get(k, float('nan')):.2f}" for k in ("Latin", "Common", "Inherited"))
        print(f"{P:<14}{100-r['ca']:>7.2f}{100-r['wa']:>8.2f}{r['errors']:>8}{r['ins']:>6}{r['dele']:>6}{r['sub']:>6}   {sc}")
    for P, r in rows.items():
        print(f"\n-- {L} {P}: top {TOPN} confusions (count  correct -> generated) --")
        for c, a, b in r["conf"][:TOPN]:
            print(f"  {c:>4}  {a} -> {b}")
