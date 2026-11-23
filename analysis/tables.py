#!/usr/bin/env python
"""Regenerate every table and every data figure of the article from the captured results.

Reads results/ and the outputs of analysis/holm_family.py and analysis/lines_analysis.py (run those first; `make all`
does it). Writes
  out/expected.json   per table/figure: the value cells of each row (as printed in the article) and their source
  out/tables.md       the same, human readable
  out/figures/*.csv   the coordinates of the data figures
analysis/check_manuscript.py compares out/expected.json with the LaTeX sources of the article, cell by cell.
Usage: python analysis/tables.py
"""
import csv, json, math, os, re, statistics as st
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "out")
H = {r["label"]: r for r in json.load(open(os.path.join(OUT, "holm_family.json")))["contrasts"]}
LJ = json.load(open(os.path.join(OUT, "lines.json")))
LC = {(c["A"], c["B"], c["metric"]): c for c in LJ["contrasts"]}
CORP = json.load(open(os.path.join(RES, "corpus", "catk12hw.json")))
KS = {r["run"]: r for r in csv.DictReader(open(os.path.join(RES, "kraken", "ketos_test_summaries.csv")))}
P1 = list(csv.DictReader(open(os.path.join(RES, "kraken", "phase1_checkpoints.csv"))))
TR = list(csv.DictReader(open(os.path.join(RES, "trocr", "test_results.csv"))))


def load(path):
    D = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        m = re.match(r"^E(\d\d)(v\d+)?(s\d+)?_(\w\w)$", r["exp_id"])
        if m:
            D.setdefault((m.group(1), m.group(2) or "", m.group(4)), []).append(r)
    return D


def load_rounds(path):
    D = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        D.setdefault(re.sub(r"_seed\d+$", "", r["exp_name"]), []).append(r)
    return D


C2 = load(os.path.join(RES, "kraken", "all_results.csv"))
C1 = load(os.path.join(RES, "kraken", "all_results_mixedbg.csv"))
R2 = load_rounds(os.path.join(RES, "kraken", "results_round2.csv"))
R3 = load_rounds(os.path.join(RES, "kraken", "results_round3.csv"))


def vals(rows, k="cer_cs"):
    return [float(r[k]) for r in rows]


# ------------------------------------------------------------------ formatting (as printed in the article)
def f2(x):
    return f"{x:.2f}"


def s2(x):
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def ms(rows, k="cer_cs"):
    v = vals(rows, k)
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f}" if len(v) > 1 else f"{st.mean(v):.2f}†"


def mean(rows, k):
    v = vals(rows, k)
    return f"{st.mean(v):.2f}" + ("" if len(v) > 1 else "†")


def ph(label):
    p = H[label]["holm"]
    if p >= 0.995:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    return "<0.001"


def welch(a, b):
    d = st.mean(a) - st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return dict(d=d, se=se, lo=d - h, hi=d + h)


def wci(A, B):
    w = welch(vals(A), vals(B))
    return f"{s2(w['d'])} [{s2(w['lo'])}, {s2(w['hi'])}]"


def wse(A, B):
    w = welch(vals(A), vals(B))
    return f"{s2(w['d'])} ({f2(w['se'])})"


def dse(A, B):
    """Δ and SE for arms where either may have one seed (SE from the replicated arm only)."""
    a, b = vals(A), vals(B)
    d = st.mean(a) - st.mean(b)
    v = (st.variance(a) / len(a) if len(a) > 1 else 0) + (st.variance(b) / len(b) if len(b) > 1 else 0)
    return d, math.sqrt(v)


def bci(A, B, key="ci_hier", metric="cs"):
    c = LC[(A, B, metric)]
    return f"[{s2(c[key][0])}, {s2(c[key][1])}]"


def bd(A, B, metric="cs"):
    return s2(LC[(A, B, metric)]["d"])


def cellv(name):
    c = LJ["cells"][name]
    return f"{c['cs']:.2f} ± {c['sd_cs']:.2f}" if c["sd_cs"] is not None else f"{c['cs']:.2f}"


def trocr_runs(prefix):
    return [r for r in TR if re.fullmatch(re.escape(prefix) + r"(_seed\d+)?", r["run"])]


def trs(prefix, k="cer_cs", sd=True):
    v = [float(r[k]) for r in trocr_runs(prefix)]
    if sd and len(v) > 1:
        return f"{st.mean(v):.2f} ± {st.stdev(v):.2f}"
    return f"{st.mean(v):.2f}"


def pct(x, n=1):
    return f"{x:.{n}f}"


def g(D, e, v, l):
    return D.get((e, v, l), [])


X = {}   # table/figure -> {"rows": [[cells]], "source": str, "caption": [tokens]}
SKIP = None


def table(label, source, rows, caption=None):
    X[label] = dict(source=source, rows=rows, caption=caption or [])


# ------------------------------------------------------------------ Table I (CatK12HW at a glance)
sp = CORP["splits"]
table("tab:catk12hw", "results/corpus/catk12hw.json (tools/export_derived.py)", [
    [SKIP, f"{sum(s['lines'] for s in sp.values()):,} / {sum(s['pages'] for s in sp.values())} / {SKIP}"],
    [SKIP, " / ".join(f"{sp[k]['lines']:,}" for k in ("train", "validation", "test"))],
    [SKIP, " / ".join(f"{sp[k]['pages']}" for k in ("train", "validation", "test"))],
    [SKIP, SKIP],
    [SKIP, " / ".join(f"{sp[k]['distinct']:,}" for k in ("train", "validation", "test")) + " (" + " / ".join(f"{sp[k]['distinct_ci']:,}" for k in ("train", "validation", "test")) + ")"],
    [SKIP, f"{pct(CORP['test_text_in_train_pct'])} ({pct(CORP['test_text_in_train_ci_pct'])})"],
    [SKIP, f"{sp['train']['mean_len']:.1f} {SKIP}"],
    [SKIP, pct(sp["train"]["upper_pct"])],
    [SKIP, f"{sp['test']['chars_nfc']:,} ({CORP['test_chars_nfd_ketos']:,})"],
    [SKIP, SKIP], [SKIP, SKIP]])

# ------------------------------------------------------------------ Table II (settings)
table("tab:settings", "results/kraken/all_results.csv (real_train_size) and ketos_test_summaries.csv (Characters)", [
    [SKIP, SKIP, SKIP, SKIP, f"{int(g(C2, '03', '', l)[0]['real_train_size']):,}", f"{int(KS[run]['characters']):,}"]
    for l, run in (("fr", "models/blockA/fr_real-only/eval_full.log"), ("en", "models/iam_aachen/en_real-only_full_seed1/eval_full.log"),
                   ("ca", "models/blockA/ca_real-only_seed1/eval_full.log"))])

# ------------------------------------------------------------------ Table III (six protocols, full real set, c2)
rows = []
for e in ("01", "02", "03", "04", "05", "06"):
    row = [SKIP]
    for l in ("fr", "en", "ca"):
        R = g(C2, e, "", l)
        row += [ms(R) if len(R) > 1 or e != "01" else f2(st.mean(vals(R))), mean(R, "cer_ci") if e != "01" else f2(st.mean(vals(R, "cer_ci")))]
    rows.append(row)
table("tab:full", "results/kraken/all_results.csv (campaign c2), E01-E06", rows)

# ------------------------------------------------------------------ Table IV (size) and Fig. S1
SZ = (("500", "11", "12"), ("1k", "13", "14"), ("2k", "15", "16"), ("4k", "17", "18"))
rows, fig = [], []
for s, r_, c_ in SZ + (("full", "03", "05"),):
    row = [SKIP]
    for l in ("fr", "en", "ca"):
        for camp, D in (("c2", C2), ("c1", C1)):
            A, B = g(D, c_, "", l), g(D, r_, "", l)
            if s == "full" and camp == "c1":
                B = g(C1, "03", "", l)        # c1 reference: the real-only seeds of that campaign
            if not A or not B:
                row.append("---"); continue
            w = welch(vals(A), vals(B))
            row.append(f"{s2(w['d'])} ({f2(w['se'])})")
            fig.append(dict(lang=l, campaign=camp, size=s, d=round(w["d"], 2), se=round(w["se"], 2)))
    rows.append(row)
table("tab:size", "results/kraken/all_results.csv (c2) and all_results_mixedbg.csv (c1): curriculum (E12/14/16/18/05) - real-only (E11/13/15/17/03), Welch", rows)
X["fig:s-size"] = dict(source="as Table IV", points=fig)

# ------------------------------------------------------------------ Fig. 4 (delta_full) and Fig. 5 (curves)
pts = []
for l in ("fr", "en", "ca"):
    for camp, D in (("c1", C1), ("c2", C2)):
        for e, n in (("04", "mixed"), ("05", "curriculum"), ("06", "curriculum-LR")):
            d, se = dse(g(D, e, "", l), g(D, "03", "", l))
            pts.append(dict(lang=l, campaign=camp, protocol=n, d=round(d, 2), se=round(se, 2) if len(g(D, e, "", l)) > 1 else None))
X["fig:deltafull"] = dict(source="all_results*.csv: E04/E05/E06 - E03 of the same campaign; SE of the difference (none with one seed)", points=pts)
pts = []
for l in ("fr", "en", "ca"):
    for s, r_ in (("500", "11"), ("1000", "13"), ("2000", "15"), ("4000", "17"), ("full", "03")):
        R = g(C2, r_, "", l)
        if R:
            pts.append(dict(lang=l, x=int(R[0]["real_train_size"]) if s == "full" else int(s), mean=round(st.mean(vals(R)), 2), sd=round(st.stdev(vals(R)), 2)))
X["fig:curves"] = dict(source="all_results.csv (c2): real-only E11/13/15/17/03, mean and SD over seeds", points=pts)

# ------------------------------------------------------------------ Table V (content)
ro_full, ro_500 = g(C2, "03", "", "ca"), g(C2, "11", "", "ca")
rows = [["real-only: " + ms(ro_full) + " and " + ms(ro_500)]]
for v, hf, h5 in (("", "A c2 ca curriculum", "C c2 ca 500"), ("v4", "Content full Kids x generic", "Content 500 Kids x generic"),
                  ("v6", "Content full HW x matched", "Content 500 HW x matched"), ("v8", "Content full Kids x matched (kidstext)", "Content 500 Kids x matched (kidstext)")):
    rows.append([SKIP, SKIP, ms(g(C2, "02", v, "ca")), ms(g(C2, "05", v, "ca")), wci(g(C2, "05", v, "ca"), ro_full) + "; " + ph(hf),
                 ms(g(C2, "12", v, "ca")), wci(g(C2, "12", v, "ca"), ro_500) + "; " + ph(h5)])
s1 = R2["ca_curriculum_500_kidstextpert500"]
rows.append([SKIP, SKIP, ms(R2["ca_synth-only_kidstextpert500"]), "---", "---", ms(s1), wci(s1, ro_500) + "; " + ph("S1 500 vs RO")])
table("tab:content", "all_results.csv (E02/E05/E12, v4/v6/v8), results_round2.csv (S1); Welch; Holm from out/holm_family.json", rows)

# ------------------------------------------------------------------ Table VI (dose)
cov = CORP["test_text_seen_pct"]
rows = []
for s, r_, c_, lab, hk, hp in (("500", "11", "12", "real 500", "Content 500 Kids x matched (kidstext)", "Content 500 kidstext-pert"),
                               ("1k", "13", "14", "real 1k", "Dose 1k kidstext", "Dose 1k kidstext-pert"),
                               ("2k", "15", "16", "real 2k", "Dose 2k kidstext", "Dose 2k kidstext-pert"),
                               ("full", "03", "05", "real full", "Content full Kids x matched (kidstext)", "Content full kidstext-pert")):
    RO = g(C2, r_, "", "ca")
    a, b = g(C2, c_, "v8", "ca"), g(C2, c_, "v12", "ca")
    c1 = cov[lab]; c2_ = cov[f"{lab} + kidstext"] if s != "full" else cov["real full"]
    rows.append([SKIP, ms(RO), ms(a), wci(a, RO) + "; " + ph(hk), ms(b), wci(b, RO) + "; " + ph(hp), f"{pct(c1)} → {pct(c2_)}"])
table("tab:dose", "all_results.csv (E11-E16/E03/E05, v8/v12), corpus/catk12hw.json (coverage)", rows,
      caption=[pct(cov["real 500 + kidstextpert"]), pct(cov["real 1k + kidstextpert"]), pct(cov["real 2k + kidstextpert"])])

# ------------------------------------------------------------------ Table VII (variants)
rows = [[SKIP, SKIP, f"{ms(ro_500)} ({len(ro_500)})", "---", f"{ms(ro_full)} ({len(ro_full)})", "---"]]
for v5, vf, h5, hf in (("v14", "v14", "Aug real-only 500", "Aug real-only full"), ("v8", "v8", "Content 500 Kids x matched (kidstext)", "Content full Kids x matched (kidstext)"),
                       ("v11", "v11", "Content 500 kidstext x5", "Content full kidstext x5"), ("v12", "v12", "Content 500 kidstext-pert", "Content full kidstext-pert"),
                       ("v13", "v13", "Content 500 kidstext-pert + aug P2", "Content full kidstext-pert + aug P2"),
                       ("v17", "v17", "Content 500 kidstext-pert + aug P1", "Content full kidstext-pert + aug P1")):
    e5, ef = ("11", "03") if v5 == "v14" else ("12", "05")
    A5, Af = g(C2, e5, v5, "ca"), g(C2, ef, vf, "ca")
    rows.append([SKIP, SKIP, f"{ms(A5)} ({len(A5)})", wci(A5, ro_500) + "; " + ph(h5), f"{ms(Af)} ({len(Af)})", wci(Af, ro_full) + "; " + ph(hf)])
table("tab:variants", "all_results.csv (E11/E12/E03/E05, v8/v11-v14/v17); Welch; Holm", rows)

# ------------------------------------------------------------------ Table VIII (hierarchical bootstrap)
rows = []
for A, B, hl in (("RO+aug 500", "RO 500", "Aug real-only 500"), ("kidstext 500", "RO 500", "Content 500 Kids x matched (kidstext)"),
                 ("kidstext-pert 500", "RO 500", "Content 500 kidstext-pert"), ("pert+aug F1 500", "RO 500", "Content 500 kidstext-pert + aug P1"),
                 ("pert+aug F2 500", "RO+aug 500", "pert+augP2 vs aug-real 500"), ("kidstext-pert 500", "kidstext 500", "pert vs kidstext 500"),
                 ("kidstext-pert 1k", "RO 1k", "Dose 1k kidstext-pert"), ("kidstext-pert 2k", "RO 2k", "Dose 2k kidstext-pert"),
                 ("kidstext-pert full", "RO full", "Content full kidstext-pert"), ("pert+aug F2 full", "RO full", "Content full kidstext-pert + aug P2"),
                 ("mix full", "RO full", "A c2 ca mix"), ("curriculum-LR full", "RO full", "A c2 ca curriculum-LR")):
    c = LC[(A, B, "cs")]
    rows.append([SKIP, SKIP, bd(A, B), bci(A, B) + " " + bci(A, B, "ci_writer_bound"), f"{c['worse_pages']}/24", ph(hl)])
table("tab:boot", "out/lines.json (hierarchical bootstrap, writer bound, pages worse); Holm from out/holm_family.json", rows)

# ------------------------------------------------------------------ Table IX (S1 controls)
ep = lambda pat: st.mean(int(r["phase1_epoch"]) for r in P1 if r["size"] == "500" and re.fullmatch(pat, r["run"]))
e_s2 = ep(r"models/blockC/ca_curriculum_500_kidstextpert_seed\d+")
e_s1 = ep(r"audit_round2/models/ca_curriculum_kidstextpert500_seed\d+")
ds = CORP["synthetic_subsample_distinct_texts"]
c2s2 = g(C2, "12", "v12", "ca")
rows = [[SKIP],
        [SKIP, f"{ds['kidstextpert']:,}", pct(cov["real 500 + kidstextpert"]), f"{e_s2:.1f}", "---", ms(c2s2), bd("kidstext-pert 500", "RO 500") + " " + bci("kidstext-pert 500", "RO 500"), ph("Content 500 kidstext-pert")],
        [SKIP, f"{ds['s1']:,}", pct(cov["real 500 + s1"]), f"{e_s1:.1f}", ms(R2["ca_synth-only_kidstextpert500"]), ms(s1), bd("S1 500", "RO 500") + " " + bci("S1 500", "RO 500"), ph("S1 500 vs RO")],
        [SKIP]]
for k, cell, hl, pool in (("s2", "S2f 500", "S2f 500 vs RO", "kidstextpert"), ("s1", "S1f 500", "S1f 500 vs RO", "s1"),
                          ("s2x415", "S2-415ext 500", None, "s2x415"), ("s2xall", "S2-extall 500", None, "s2xall")):
    rows.append([SKIP, f"{ds[pool]:,}", pct(cov[f'real 500 + {pool}']), "19", ms(R3[f"ca_synth-only_{k}_fixed20"]), ms(R3[f"ca_curriculum_500_{k}_fixed20"]),
                 bd(cell, "RO 500") + " " + bci(cell, "RO 500"), ph(hl) if hl else "---"])
rows.append([SKIP])
for A, B, hl in (("S1 500", "kidstext-pert 500", "S1 500 vs S2 (kidstext-pert) 500"), ("S1f 500", "S2f 500", "B1' S1f vs S2f 500"),
                 ("S2-415ext 500", "S1f 500", "B9a S2-415ext vs S1f 500"), ("S2-extall 500", "S2-415ext 500", "B9b S2-extall vs S2-415ext 500"),
                 ("S2f 500", "S2-extall 500", "S2f vs S2-extall 500")):
    rows.append([SKIP, bd(A, B) + " " + bci(A, B), ph(hl)])
table("tab:s1s2", "phase1_checkpoints.csv (epoch; fixed arms: 19 by design), corpus/catk12hw.json (texts, coverage), results_round2/3.csv, out/lines.json, Holm", rows)

# ------------------------------------------------------------------ Table X (TrOCR)
def trow(n, prefix, cell=None, con=None, kr=None):
    cs = cellv(cell) if cell else trs(prefix)
    ci = f"{LJ['cells'][cell]['ci']:.2f}" if cell else trs(prefix, "cer_ci", False)
    wer = f"{LJ['cells'][cell]['wer']:.2f}" if cell else trs(prefix, "wer_cs", False)
    dl = "---" if con is None else f"{bd(*con)} {bci(*con)} ({LC[(con[0], con[1], 'cs')]['worse_pages']})"
    return [SKIP, str(n), cs, ci, wer, dl, kr]


kb = f2(st.mean(vals(g(C2, "01", "", "ca"))))
rows = [trow(1, "trocr_r2/blockA/ca_baseline", kr=kb + " " + SKIP if False else kb),
        trow(3, "trocr_r2/blockA/ca_synth-only_kidstext", kr=ms(g(C2, "02", "v8", "ca"))),
        trow(6, "trocr_r2/blockA/ca_synth-only_kidstextpert", kr="---"), [SKIP]]
for sz, ro_, kro, sfx in (("500", "TrOCR-hw RO 500", ro_500, " 500"), ("full", "TrOCR-hw RO full", ro_full, " full")):
    e = "12" if sz == "500" else "05"
    rows.append(trow(6, None, ro_, kr=ms(kro)))
    for nm, v, n in (("kidstext", "v8", 6), ("x5", "v11", 3), ("kidstext-pert", "v12", 6)):
        c = f"TrOCR-hw {nm}{sfx}"
        rows.append(trow(n, None, c, (c, ro_), ms(g(C2, e, v, "ca"))))
    rows.append([SKIP])
rows.pop()
rows += [[SKIP], trow(1, "trocr_s1/blockA/ca_baseline", kr=kb), trow(6, "trocr_s1/blockA/ca_synth-only_kidstext", kr=ms(g(C2, "02", "v8", "ca"))),
         trow(6, None, "TrOCR-s1 RO 500", kr=ms(ro_500)), trow(6, None, "TrOCR-s1 kidstext 500", ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500"), ms(g(C2, "12", "v8", "ca"))),
         trow(6, None, "TrOCR-s1 RO full", kr=ms(ro_full)), trow(6, None, "TrOCR-s1 kidstext full", ("TrOCR-s1 kidstext full", "TrOCR-s1 RO full"), ms(g(C2, "05", "v8", "ca")))]
table("tab:trocr", "results/trocr/test_results.csv (zero-shot, synthetic-only), out/lines.json (cells, bootstrap), all_results.csv (Kraken column)", rows)

# ------------------------------------------------------------------ Table XI (strata Kraken vs TrOCR) and S4
def strata_rows(name, cons):
    S = [s for s in LJ["strata"] if s["kind"] == "multi" and s["name"] == name and not s["case_insensitive"]][0]
    I = {(it["contrast"], it["strata"]): it for it in S["inter"]}
    out = []
    for A, B in cons:
        row = [SKIP, SKIP] if name == "Kraken vs TrOCR" else [SKIP]
        for lab in ("G1", "G2'", "G3"):
            x = S["strata"][lab]["d"][f"{A} - {B}"]
            row.append(f"{s2(x['d'])} ({100 * x['rel']:+.0f}%) [{s2(x['ci'][0])}, {s2(x['ci'][1])}]")
        it = I[(f"{A} - {B}", "G2' - G3")]
        row.append(f"{s2(it['d'])} [{s2(it['ci'][0])}, {s2(it['ci'][1])}]")
        out.append(row)
    cap = []
    for lab in ("G1", "G2'", "G3"):
        cap += [str(S["strata"][lab]["lines"]), f"{S['strata'][lab]['chars_pct']:.1f}"]
    return out, cap


r, cap = strata_rows("Kraken vs TrOCR", [("kidstext 500", "RO 500"), ("kidstext-pert 500", "RO 500"), ("TrOCR-hw kidstext 500", "TrOCR-hw RO 500"),
                                          ("TrOCR-hw kidstext-pert 500", "TrOCR-hw RO 500"), ("TrOCR-hw x5 500", "TrOCR-hw RO 500"), ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500")])
table("tab:strata", "out/lines.json (strata 'Kraken vs TrOCR', case-sensitive)", r, caption=cap)
ra, cap = strata_rows("early stopping", [("S1 500", "RO 500"), ("S1 500", "kidstext-pert 500")])
rb, _ = strata_rows("fixed 20", [("S2f 500", "RO 500"), ("S1f 500", "RO 500"), ("S1f 500", "S2f 500"), ("S2-415ext 500", "S1f 500"),
                                 ("S2-extall 500", "S2-415ext 500"), ("S2f 500", "S2-extall 500")])
table("tab:s-strata", "out/lines.json (strata 'early stopping' and 'fixed 20', case-sensitive)", [[SKIP]] + ra + [[SKIP]] + rb, caption=cap)

# ------------------------------------------------------------------ Table XII (starting points)
# Kraken cells as logged by ketos test (all_results.csv), as in every other table; TrOCR cells from the per-line scores
KCSV = {"RO 500": ("11", ""), "RO full": ("03", ""), "kidstext 500": ("12", "v8"), "kidstext full": ("05", "v8"),
        "kidstext-pert 500": ("12", "v12"), "kidstext-pert full": ("05", "v12"), "pert+aug F1 500": ("12", "v17"),
        "pert+aug F1 full": ("05", "v17"), "scratch RO 500": ("11", "v10"), "scratch RO full": ("03", "v10"),
        "scratch cur 500": ("12", "v10"), "scratch cur full": ("05", "v10"), "scratch curpert full": ("05", "v16")}


def kcell(name):
    return ms(g(C2, *KCSV[name], "ca")) if name in KCSV else cellv(name)


def xrow(c5, c5d, cf, cfd):
    return [SKIP, kcell(c5) if c5 else "---", (bd(*c5d) + " " + bci(*c5d)) if c5d else "---", kcell(cf), (bd(*cfd) + " " + bci(*cfd)) if cfd else "---"]


SRCDIFF = []   # Kraken cells whose per-line recomputation differs from the logged value at the printed precision
for name, (e, v) in KCSV.items():
    a, b = ms(g(C2, e, v, "ca")), cellv(name)
    if a != b:
        SRCDIFF.append(dict(cell=name, logged=a, per_line=b))


rows = [[SKIP], xrow("scratch RO 500", None, "scratch RO full", None),
        xrow("scratch cur 500", ("scratch cur 500", "scratch RO 500"), "scratch cur full", ("scratch cur full", "scratch RO full")),
        xrow(None, None, "scratch curpert full", ("scratch curpert full", "scratch RO full")), [SKIP],
        xrow("RO 500", None, "RO full", None)]
for k in ("kidstext", "kidstext-pert", "pert+aug F1"):
    rows.append(xrow(f"{k} 500", (f"{k} 500", "RO 500"), f"{k} full", (f"{k} full", "RO full")))
rows += [[SKIP], xrow("TrOCR-s1 RO 500", None, "TrOCR-s1 RO full", None),
         xrow("TrOCR-s1 kidstext 500", ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500"), "TrOCR-s1 kidstext full", ("TrOCR-s1 kidstext full", "TrOCR-s1 RO full")), [SKIP],
         xrow("TrOCR-hw RO 500", None, "TrOCR-hw RO full", None)]
for k in ("kidstext", "kidstext-pert"):
    rows.append(xrow(f"TrOCR-hw {k} 500", (f"TrOCR-hw {k} 500", "TrOCR-hw RO 500"), f"TrOCR-hw {k} full", (f"TrOCR-hw {k} full", "TrOCR-hw RO full")))
w5 = H["Scratch 500 curriculum vs RO"]; wf = H["Scratch full curriculum(white30k) vs RO"]
table("tab:cross", "out/lines.json (cells and hierarchical bootstrap); caption: Welch from out/holm_family.json", rows,
      caption=[f"{w5['cer_cs_lo']:+.1f}", f"{w5['cer_cs_hi']:+.1f}", f"{wf['cer_cs_lo']:+.1f}", f"{wf['cer_cs_hi']:+.1f}"])

# ------------------------------------------------------------------ Supplementary Table S3 (base model) and S5 (scratch)
rows = [[SKIP]]
for e in ("02", "03", "04", "05", "06"):
    row = [SKIP]
    for v in ("", "v2"):
        A, B = g(C1, e, v, "ca"), g(C1, "03", v, "ca")
        row += [ms(A), "---" if e == "03" else wse(A, B)]
    rows.append(row)
rows.append([SKIP])
for r_, c_ in (("11", "12"), ("13", "14"), ("15", "16")):
    row = [SKIP]
    for v in ("", "v2"):
        A, B = g(C1, c_, v, "ca"), g(C1, r_, v, "ca")
        row += [ms(A), wse(A, B)]
    rows.append(row)
table("tab:s-base", "all_results_mixedbg.csv (c1), CA, base FoNDUE-GD_v2_fr ('') and FoNDUE-GD_v2 ('v2'); Welch SE", rows)
ro5, rof = g(C2, "11", "v10", "ca"), g(C2, "03", "v10", "ca")
rows = [[SKIP, ms(ro5), "---", ms(rof), "---", f"{ms(ro_500)} / {ms(ro_full)}"],
        [SKIP, ms(g(C2, "11", "v15", "ca")), wci(g(C2, "11", "v15", "ca"), ro5), ms(g(C2, "03", "v15", "ca")), wci(g(C2, "03", "v15", "ca"), rof), "---"],
        [SKIP, ms(g(C2, "12", "v10", "ca")), wci(g(C2, "12", "v10", "ca"), ro5), ms(g(C2, "05", "v10", "ca")), wci(g(C2, "05", "v10", "ca"), rof),
         f"{ms(g(C2, '12', '', 'ca'))} / {ms(g(C2, '05', '', 'ca'))} {SKIP}"],
        [SKIP, "---", "---", ms(g(C2, "05", "v16", "ca")), wci(g(C2, "05", "v16", "ca"), rof), "---"]]
table("tab:s-scratch", "all_results.csv (c2), v10 (second from-scratch recipe), v15 (two-stage control), v16 (perturbed pool); Welch", rows)

# ------------------------------------------------------------------ Fig. S2 (error composition)
pts = []
for l, ro, so in (("FR", "models/blockA/fr_real-only/eval_full.log", "models/mixedbg/blockA/fr_synth-only/eval_full.log"),
                  ("EN", "models/iam_aachen/en_real-only_full_seed1/eval_full.log", "models/iam_aachen/c1_en_synth-only_seed1/eval_full.log"),
                  ("CA", "models/blockA/ca_real-only_seed1/eval_full.log", "models/mixedbg/blockA/ca_synth-only_seed1/eval_full.log")):
    for prot, run in (("real-only", ro), ("synth.-only", so)):
        k = KS[run]; E = int(k["errors"])
        # article convention: deletions = reference characters the model does not emit = ketos "Insertions" column
        pts.append(dict(label=f"{l} {prot}", errors=E, deletions=round(100 * int(k["ketos_insertions"]) / E),
                        insertions=round(100 * int(k["ketos_deletions"]) / E), substitutions=round(100 * int(k["substitutions"]) / E)))
X["fig:s-errors"] = dict(source="results/kraken/ketos_test_summaries.csv (heterogeneous-background synthetic-only; real-only seed 1)", points=pts)

# ------------------------------------------------------------------ values quoted in the prose that no table holds
PV = {"intervals": [], "values": {}, "source_differences": SRCDIFF}
fac = {"HWgen": ("", "C"), "KIDgen": ("v4", ""), "HWtxt": ("v6", ""), "KIDtxt": ("v8", "")}


def lin(e, w):
    c = {k: vals(g(C2, e, fac[k][0], "ca")) for k in w}
    est = sum(w[k] * st.mean(c[k]) for k in w)
    vs = {k: (w[k] ** 2) * st.variance(c[k]) / len(c[k]) for k in w}
    se = math.sqrt(sum(vs.values()))
    df = sum(vs.values()) ** 2 / sum(vs[k] ** 2 / (len(c[k]) - 1) for k in w)
    h = stats.t.ppf(.975, df) * se
    return est, est - h, est + h


for lab, w in (("interaction font x text", {"KIDtxt": 1, "HWtxt": -1, "KIDgen": -1, "HWgen": 1}),
               ("text effect, Handwritten", {"HWtxt": 1, "HWgen": -1}), ("text effect, KidsCurated", {"KIDtxt": 1, "KIDgen": -1}),
               ("font effect, generic text", {"KIDgen": 1, "HWgen": -1}), ("font effect, domain text", {"KIDtxt": 1, "HWtxt": -1}),
               ("main effect of text", {"HWtxt": .5, "KIDtxt": .5, "HWgen": -.5, "KIDgen": -.5}),
               ("main effect of fonts", {"KIDgen": .5, "KIDtxt": .5, "HWgen": -.5, "HWtxt": -.5})):
    est, lo, hi = lin("12", w)
    PV["values"][f"factorial 500 lines: {lab}"] = f"{s2(est)} [{s2(lo)}, {s2(hi)}]"
    PV["intervals"].append([lo, hi])
w = welch(vals(s1), vals(g(C2, "12", "v8", "ca")))
PV["values"]["S1 (perturbed) - kidstext, 500 lines (Welch)"] = f"{s2(w['d'])} [{s2(w['lo'])}, {s2(w['hi'])}]"
PV["intervals"].append([w["lo"], w["hi"]])
cur = [r for r in C2.values() for r in r if r["language"] == "ca" and r["model_base"] == "FoNDUE-GD_v2_fr" and r["best_stage"]
       and r["protocol"] == "curriculum" and r["real_train_size"] == "4152"]
rho, p = stats.spearmanr([int(r["best_stage"]) for r in cur], [float(r["cer_cs"]) for r in cur])
PV["values"][f"Spearman(best epoch, CER-cs), {len(cur)} Catalan curricula, full real set"] = f"{rho:+.2f} (p = {p:.2f})"
json.dump(PV, open(os.path.join(OUT, "prose_values.json"), "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ write
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
json.dump(X, open(os.path.join(OUT, "expected.json"), "w"), indent=1, ensure_ascii=False)
md = ["# Regenerated tables and figure data\n"]
for k, v in X.items():
    md.append(f"\n## {k}\n\nSource: {v['source']}\n")
    if "rows" in v:
        for r in v["rows"]:
            md.append("| " + " | ".join("·" if c is None else c for c in r) + " |")
        if v["caption"]:
            md.append(f"\nCaption values: {', '.join(v['caption'])}")
    else:
        keys = list(v["points"][0])
        md.append("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
        for p in v["points"]:
            md.append("| " + " | ".join(str(p[c]) for c in keys) + " |")
        with open(os.path.join(OUT, "figures", k.split(":")[1] + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(v["points"])
md.append("\n## Values quoted in the prose\n")
md += [f"- {k}: {v}" for k, v in PV["values"].items()]
md.append("\n## Kraken cells whose per-line recomputation differs from the value logged by ketos test\n")
md += [f"- {d['cell']}: logged {d['logged']}, per line {d['per_line']}" for d in SRCDIFF] or ["- none"]
open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print(f"{sum('rows' in v for v in X.values())} tables and {sum('points' in v for v in X.values())} figures -> out/expected.json, out/tables.md, out/figures/")
