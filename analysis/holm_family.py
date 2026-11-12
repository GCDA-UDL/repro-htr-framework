#!/usr/bin/env python
"""Single family of between-protocol contrasts of the article: Welch on CER-cs between seeds, Holm and Benjamini-Hochberg.

Inputs (captured results, results/kraken/):
  all_results.csv          campaign c2 (synthetic background matched to the real paper)
  all_results_mixedbg.csv  campaign c1 (heterogeneous background)
  results_round2.csv       S1 control with early stopping on the synthetic validation
  results_round3.csv       S1/S2 controls with a phase 1 of exactly 20 epochs
A contrast enters the family when both arms have at least two seeds. Two duplicated contrasts are left out
("Content full/500 Handwritten x generic" are the c2 Catalan curriculum contrasts of blocks A and C), as are the S1 arms
at 1,000 and 2,000 lines, which are not scenario S1. CER-ci and WER of the same contrasts are reported unadjusted.
Outputs: out/holm_family.md and out/holm_family.json. Usage: python analysis/holm_family.py
"""
import csv, json, math, os, re, statistics as st
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "kraken")
OUT = os.path.join(ROOT, "out")
LANGS = ("fr", "en", "ca")


def load(path, pattern=r"^E(\d\d)(v\d+)?(s\d+)?_(\w\w)$"):
    D = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("status", "completed") != "completed":
            continue
        m = re.match(pattern, r["exp_id"])
        if not m:
            continue
        key = (m.group(1), m.group(2) or "", m.group(4))
        D.setdefault(key, []).append({k: float(r[k]) for k in ("cer_cs", "cer_ci", "wer_cs") if r.get(k, "") != ""})
    return D


def load_rounds(path):
    D = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("status") != "completed":
            continue
        key = re.sub(r"_seed\d+$", "", r["exp_name"])
        D.setdefault(key, []).append({k: float(r[k]) for k in ("cer_cs", "cer_ci", "wer_cs") if r.get(k, "") != ""})
    return D


C2 = load(os.path.join(RES, "all_results.csv"))
C1 = load(os.path.join(RES, "all_results_mixedbg.csv"))
R2 = load_rounds(os.path.join(RES, "results_round2.csv"))
R3 = load_rounds(os.path.join(RES, "results_round3.csv"))


def g(D, e, v, l):
    return D.get((e, v, l), [])


rows = []


def welch(a, b):
    d = st.mean(a) - st.mean(b)
    if len(a) < 2 or len(b) < 2:
        return dict(d=d, se=math.nan, t=math.nan, df=math.nan, p=math.nan, lo=math.nan, hi=math.nan)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return dict(d=d, se=0.0, t=math.inf, df=math.nan, p=0.0, lo=d, hi=d)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = d / se
    h = stats.t.ppf(0.975, df) * se
    return dict(d=d, se=se, t=t, df=df, p=2 * stats.t.sf(abs(t), df), lo=d - h, hi=d + h)


def contrast(label, A, B, fam):
    if not A or not B:
        return
    r = dict(label=label, fam=fam, na=len(A), nb=len(B))
    for k in ("cer_cs", "cer_ci", "wer_cs"):
        a = [x[k] for x in A if k in x]; b = [x[k] for x in B if k in x]
        if a and b:
            w = welch(a, b)
            r.update({f"{k}_{n}": v for n, v in w.items()})
            r[f"{k}_a"], r[f"{k}_b"] = st.mean(a), st.mean(b)
    rows.append(r)


# block A (full real set), c2 and c1, vs real-only of the same campaign's reference (c2 real-only)
for l in LANGS:
    ro = g(C2, "03", "", l)
    for e, n in (("04", "mix"), ("05", "curriculum"), ("06", "curriculum-LR")):
        contrast(f"A c2 {l} {n}", g(C2, e, "", l), ro, "A")
        contrast(f"A c1 {l} {n}", g(C1, e, "", l), ro, "A")
# block C: curriculum vs real-only per size, c2 and c1
sizes = {"500": ("11", "12"), "1k": ("13", "14"), "2k": ("15", "16"), "4k": ("17", "18")}
for l in LANGS:
    for s, (r_, c_) in sizes.items():
        contrast(f"C c2 {l} {s}", g(C2, c_, "", l), g(C2, r_, "", l), "C")
        contrast(f"C c1 {l} {s}", g(C1, c_, "", l), g(C1, r_, "", l), "C")
# content (CA, c2) vs the six-seed real-only
ro_full, ro_500 = g(C2, "03", "", "ca"), g(C2, "11", "", "ca")
for v, n in (("v4", "Kids x generic"), ("v6", "HW x matched"), ("v8", "Kids x matched (kidstext)"), ("v11", "kidstext x5"),
             ("v12", "kidstext-pert"), ("v13", "kidstext-pert + aug P2"), ("v17", "kidstext-pert + aug P1")):
    contrast(f"Content full {n}", g(C2, "05", v, "ca"), ro_full, "content")
    contrast(f"Content 500 {n}", g(C2, "12", v, "ca"), ro_500, "content")
contrast("Aug real-only full", g(C2, "03", "v14", "ca"), ro_full, "content")
contrast("Aug real-only 500", g(C2, "11", "v14", "ca"), ro_500, "content")
for s, (r_, c_) in (("1k", ("13", "14")), ("2k", ("15", "16"))):
    for v, n in (("v8", "kidstext"), ("v12", "kidstext-pert")):
        contrast(f"Dose {s} {n}", g(C2, c_, v, "ca"), g(C2, r_, "", "ca"), "content")
contrast("pert+augP2 vs aug-real 500", g(C2, "12", "v13", "ca"), g(C2, "11", "v14", "ca"), "content")
contrast("pert vs kidstext 500", g(C2, "12", "v12", "ca"), g(C2, "12", "v8", "ca"), "content")
contrast("pert+augP1 vs pert 500", g(C2, "12", "v17", "ca"), g(C2, "12", "v12", "ca"), "content")
contrast("pert+augP2 vs aug-real full", g(C2, "05", "v13", "ca"), g(C2, "03", "v14", "ca"), "content")
contrast("pert+augP2 vs pert full", g(C2, "05", "v13", "ca"), g(C2, "05", "v12", "ca"), "content")
contrast("kidstext x5 vs kidstext 500", g(C2, "12", "v11", "ca"), g(C2, "12", "v8", "ca"), "content")
# base-model ablation (campaign 1, FoNDUE-GD_v2), CA
ro2 = g(C1, "03", "v2", "ca") or g(C2, "03", "v2", "ca")
for e, n in (("04", "mix"), ("05", "curriculum"), ("06", "curriculum-LR")):
    contrast(f"Base-v2 c1 full {n}", g(C1, e, "v2", "ca"), ro2, "base")
for s, (r_, c_) in (("500", ("11", "12")), ("1k", ("13", "14")), ("2k", ("15", "16"))):
    contrast(f"Base-v2 c1 {s}", g(C1, c_, "v2", "ca"), g(C1, r_, "v2", "ca") or g(C2, r_, "v2", "ca"), "base")
# from scratch (second recipe)
contrast("Scratch full curriculum(white30k) vs RO", g(C2, "05", "v10", "ca"), g(C2, "03", "v10", "ca"), "scratch")
contrast("Scratch full 2-stage control vs RO", g(C2, "03", "v15", "ca"), g(C2, "03", "v10", "ca"), "scratch")
contrast("Scratch full curriculum vs 2-stage control", g(C2, "05", "v10", "ca"), g(C2, "03", "v15", "ca"), "scratch")
contrast("Scratch full whitepert vs white", g(C2, "05", "v16", "ca"), g(C2, "05", "v10", "ca"), "scratch")
contrast("Scratch 500 curriculum vs RO", g(C2, "12", "v10", "ca"), g(C2, "11", "v10", "ca"), "scratch")
contrast("Scratch 500 2-stage vs RO", g(C2, "11", "v15", "ca"), g(C2, "11", "v10", "ca"), "scratch")
# S1 with early stopping on its own synthetic validation, 500 lines only
s1 = R2.get("ca_curriculum_500_kidstextpert500", [])
s2 = g(C2, "12", "v12", "ca")
contrast("S1 500 vs RO", s1, ro_500, "S1")
contrast("S1 500 vs S2 (kidstext-pert) 500", s1, s2, "S1")
# phase 1 of exactly 20 epochs (last checkpoint), 500 lines
f = {k: R3.get(f"ca_curriculum_500_{k}_fixed20", []) for k in ("s1", "s2", "s2x415", "s2xall")}
contrast("S1f 500 vs RO", f["s1"], ro_500, "R3")
contrast("S2f 500 vs RO", f["s2"], ro_500, "R3")
contrast("B1' S1f vs S2f 500", f["s1"], f["s2"], "R3")
contrast("B9a S2-415ext vs S1f 500", f["s2x415"], f["s1"], "R3")
contrast("B9b S2-extall vs S2-415ext 500", f["s2xall"], f["s2x415"], "R3")
contrast("S2f vs S2-extall 500", f["s2"], f["s2xall"], "R3")

# Holm and BH over every contrast with a CER-cs p value
tests = sorted([r for r in rows if not math.isnan(r.get("cer_cs_p", math.nan))], key=lambda r: r["cer_cs_p"])
m = len(tests)
run = 0.0
for i, r in enumerate(tests):
    run = max(run, min(1.0, (m - i) * r["cer_cs_p"])); r["holm"] = run
prev = 1.0
for i in range(m - 1, -1, -1):
    prev = min(prev, tests[i]["cer_cs_p"] * m / (i + 1)); tests[i]["bh"] = prev
fams = {}
for r in tests:
    fams[r["fam"]] = fams.get(r["fam"], 0) + 1

fmt = lambda x, n=2: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{n}f}"
lines = ["# Single family of contrasts (Welch on CER-cs between seeds), Holm and BH\n",
         f"Contrasts with a p value (n >= 2 per arm): **{m}**; by group: {fams}.\n",
         "| Contrast | n A/B | Δ CER-cs | Welch 95% CI | p | p Holm | q BH | Δ CER-ci (p) | Δ WER (p) |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "cer_cs_d" not in r:
        continue
    p = r.get("cer_cs_p", math.nan)
    ci = f"[{r['cer_cs_lo']:+.2f}; {r['cer_cs_hi']:+.2f}]" if not math.isnan(p) else "-"
    pci = f"{fmt(r.get('cer_ci_d'))} ({r.get('cer_ci_p', math.nan):.3g})" if "cer_ci_d" in r else "-"
    pw = f"{fmt(r.get('wer_cs_d'))} ({r.get('wer_cs_p', math.nan):.3g})" if "wer_cs_d" in r else "-"
    lines.append(f"| {r['label']} | {r['na']}/{r['nb']} | {r['cer_cs_d']:+.2f} | {ci} | {p:.3g} | {r.get('holm', math.nan):.3g} | "
                 f"{r.get('bh', math.nan):.3g} | {pci} | {pw} |")
lines.append("\n## Holm survivors\n")
for r in tests:
    if r["holm"] < 0.05:
        lines.append(f"- {r['label']}: Δ {r['cer_cs_d']:+.2f}, p {r['cer_cs_p']:.3g}, p_H {r['holm']:.3g}")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "holm_family.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({"family_size": m, "groups": fams, "contrasts": rows}, open(os.path.join(OUT, "holm_family.json"), "w"), indent=1, default=float)
print("\n".join(lines[:2] + lines[-8:]))
