#!/usr/bin/env python
"""Per-line analysis of the CatK12HW-actions test predictions (Kraken and TrOCR): hierarchical bootstrap over test
pages x seeds, Welch intervals, writer-clustering bound, strata by the exact text of the test line and the S1/S2 ratio.

Same algorithm and random streams as the analysis behind the article (each comparison draws its replicas from a
generator seeded with zlib.crc32 of its tag; NB = 4000), but reading the text-free tables of results/perline/
(errors per line, pseudonymous page index and precomputed strata) instead of the private predictions.
  * Hierarchical bootstrap: the 24 test pages are resampled with replacement (the same pages for every arm of the
    comparison: paired) and, inside each replica, the seeds of each arm are resampled with replacement.
  * Writer-clustering bound: the pages-only variance is added to the hierarchical variance (normal approximation).
  * Strata by exact line text: G1 the text is in the 500 real lines; G2 only in the arm's synthetic subsample; G2b in
    the training set but in neither; G3 not in the training set (G2' = G2 + G2b in the multi-arm strata).
Outputs: out/lines.json and out/lines.md. Usage: python analysis/lines_analysis.py [--nb 4000]
"""
import argparse, csv, json, math, os, statistics as st, zlib
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "results", "perline")
OUT = os.path.join(ROOT, "out")
Z = 1.959963984540054

LINES = list(csv.DictReader(open(os.path.join(PL, "lines.csv"))))
L = len(LINES)
page_of = np.array([int(r["page"]) for r in LINES])
P = int(page_of.max()) + 1
ALL = np.ones(L, bool)
CELLS = {c["cell"]: c for c in json.load(open(os.path.join(PL, "cells.json")))}


def label(col):
    return np.array([r[col] for r in LINES])


def load_cell(name):
    c = CELLS[name]
    rows = list(csv.DictReader(open(os.path.join(PL, "runs", c["file"]))))
    runs = []
    for k, rn in enumerate(c["runs"]):
        R = sorted((r for r in rows if int(r["run"]) == k), key=lambda r: int(r["line"]))
        assert len(R) == L
        d = {key: np.array([float(r[key]) for r in R]) for key in ("n", "e", "eci", "w", "ew")}
        d["name"] = rn
        runs.append(d)
    return runs


METRIC = {"cs": ("e", "n"), "ci": ("eci", "n"), "wer": ("ew", "w")}


def page_sums(runs, mask, metric):
    num, den = METRIC[metric]
    E = np.stack([np.bincount(page_of[mask], weights=r[num][mask], minlength=P) for r in runs])
    N = np.stack([np.bincount(page_of[mask], weights=r[den][mask], minlength=P) for r in runs])
    return E, N


def point(runs, mask=ALL, metric="cs"):
    E, N = page_sums(runs, mask, metric)
    return float(np.mean(100 * E.sum(1) / N.sum(1)))


def per_seed(runs, mask=ALL, metric="cs"):
    E, N = page_sums(runs, mask, metric)
    return list(100 * E.sum(1) / N.sum(1))


class Draws:
    """Replicas for one comparison: page counts shared by every arm, seed indices per arm."""

    def __init__(self, tag, arms, nb):
        rng = np.random.default_rng(zlib.crc32(tag.encode("utf-8")))
        idx = rng.integers(0, P, size=(nb, P))
        self.C = np.stack([np.bincount(row, minlength=P) for row in idx]).astype(float)
        self.S = {k: rng.integers(0, len(v), size=(nb, len(v))) for k, v in sorted(arms.items())}
        self.nb, self.arms = nb, arms

    def rates(self, key, mask=ALL, metric="cs", seeds=True):
        E, N = page_sums(self.arms[key], mask, metric)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = 100 * (E @ self.C.T) / (N @ self.C.T)          # (runs, nb)
        if seeds:
            return r[self.S[key].T, np.arange(self.nb)].mean(0)
        return r.mean(0)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def welch(a, b):
    d = st.mean(a) - st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return dict(d=d, se=se, df=df, p=float(2 * stats.t.sf(abs(d / se), df)), lo=d - h, hi=d + h)


def contrast(tag, A, B, nb, metric="cs"):
    D = Draws(tag, {"A": A, "B": B}, nb)
    ds = D.rates("A", metric=metric) - D.rates("B", metric=metric)
    dp = D.rates("A", metric=metric, seeds=False) - D.rates("B", metric=metric, seeds=False)
    d0 = point(A, metric=metric) - point(B, metric=metric)
    w = welch(per_seed(A, metric=metric), per_seed(B, metric=metric))
    sd_w = math.sqrt(np.nanvar(ds) + np.nanvar(dp))
    worse = sum(point(A, page_of == k, metric) > point(B, page_of == k, metric) for k in range(P))
    return dict(tag=tag, metric=metric, nA=len(A), nB=len(B), cer_A=point(A, metric=metric), cer_B=point(B, metric=metric), d=d0,
                welch=w, ci_hier=ci(ds), ci_pages=ci(dp), ci_writer_bound=[d0 - Z * sd_w, d0 + Z * sd_w],
                var_ratio=float(np.nanvar(ds) / w["se"] ** 2), worse_pages=int(worse))


def strata(tag, arms, contrasts, masks, nb, metric, inter=()):
    """Δ per stratum for several contrasts on the same replicas. masks: {label: bool array}."""
    D = Draws(tag, arms, nb)
    res = {"tag": tag, "metric": metric, "strata": {}, "inter": []}
    cache = {}
    for lab, m in masks.items():
        if not m.any():
            continue
        row = {"lines": int(m.sum()), "chars_pct": float(100 * arms[next(iter(arms))][0]["n"][m].sum() / arms[next(iter(arms))][0]["n"].sum()),
               "cer": {k: point(v, m, metric) for k, v in arms.items()}, "d": {}}
        for A, B in contrasts:
            a, b = D.rates(A, m, metric), D.rates(B, m, metric)
            cache[(A, B, lab)] = a - b
            d0 = row["cer"][A] - row["cer"][B]
            rel = (a - b) / b
            row["d"][f"{A} - {B}"] = dict(d=d0, ci=ci(a - b), rel=d0 / row["cer"][B], rel_ci=ci(100 * rel))
        res["strata"][lab] = row
    for (A, B), l1, l2 in inter:
        if (A, B, l1) in cache and (A, B, l2) in cache:
            x = cache[(A, B, l1)] - cache[(A, B, l2)]
            d0 = res["strata"][l1]["d"][f"{A} - {B}"]["d"] - res["strata"][l2]["d"][f"{A} - {B}"]["d"]
            res["inter"].append(dict(contrast=f"{A} - {B}", strata=f"{l1} - {l2}", d=d0, ci=ci(x)))
    return res


def ratio(tag, RO, S1, S2, nb, metric="cs"):
    """(S1 - RO) / (S2 - RO): share of the S2 effect kept by S1."""
    D = Draws(tag, {"RO": RO, "S1": S1, "S2": S2}, nb)
    ro, a, b = D.rates("RO", metric=metric), D.rates("S1", metric=metric), D.rates("S2", metric=metric)
    r0 = (point(S1, metric=metric) - point(RO, metric=metric)) / (point(S2, metric=metric) - point(RO, metric=metric))
    ros, as_, bs = (D.rates(k, metric=metric, seeds=False) for k in ("RO", "S1", "S2"))
    # seeds only: pages fixed (all), seeds resampled
    rng = np.random.default_rng(zlib.crc32((tag + "/seeds").encode()))
    sr = lambda runs: np.array(per_seed(runs, metric=metric))
    v_ro, v_a, v_b = sr(RO), sr(S1), sr(S2)
    m = lambda v: v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    rs = (m(v_a) - m(v_ro)) / (m(v_b) - m(v_ro))
    return dict(tag=tag, metric=metric, ratio=r0, ci_hier=ci((a - ro) / (b - ro)), ci_pages=ci((as_ - ros) / (bs - ros)), ci_seeds=ci(rs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nb", type=int, default=4000); a = ap.parse_args()
    nb = a.nb
    data = {k: load_cell(k) for k in CELLS}
    data = {k: v for k, v in data.items() if v}
    R = {"cells": {k: dict(n=len(v), cs=point(v), ci=point(v, metric="ci"), wer=point(v, metric="wer"),
                           sd_cs=float(np.std(per_seed(v), ddof=1)) if len(v) > 1 else None) for k, v in data.items()},
         "contrasts": [], "strata": [], "ratios": [], "g3_words": {}}
    pairs = [("RO+aug 500", "RO 500"), ("HWxgen 500", "RO 500"), ("Kidsxgen 500", "RO 500"), ("HWxtxt 500", "RO 500"), ("kidstext 500", "RO 500"),
             ("kidstext x5 500", "RO 500"), ("kidstext-pert 500", "RO 500"), ("pert+aug F2 500", "RO 500"), ("pert+aug F1 500", "RO 500"),
             ("pert+aug F2 500", "RO+aug 500"), ("kidstext-pert 500", "kidstext 500"), ("S1 500", "RO 500"), ("S1 500", "kidstext-pert 500"),
             ("S1 500", "RO+aug 500"),
             ("S1f 500", "RO 500"), ("S2f 500", "RO 500"), ("S2-415ext 500", "RO 500"), ("S2-extall 500", "RO 500"),
             ("S1f 500", "S2f 500"), ("S2-415ext 500", "S1f 500"), ("S2-extall 500", "S2-415ext 500"), ("S2f 500", "S2-extall 500"),
             ("S1f 500", "S1 500"), ("S2f 500", "kidstext-pert 500"),
             ("kidstext 1k", "RO 1k"), ("kidstext-pert 1k", "RO 1k"), ("kidstext 2k", "RO 2k"), ("kidstext-pert 2k", "RO 2k"),
             ("RO+aug full", "RO full"), ("HWxgen full", "RO full"), ("Kidsxgen full", "RO full"), ("HWxtxt full", "RO full"),
             ("kidstext full", "RO full"), ("kidstext x5 full", "RO full"), ("kidstext-pert full", "RO full"), ("pert+aug F2 full", "RO full"),
             ("pert+aug F1 full", "RO full"), ("pert+aug F2 full", "RO+aug full"), ("mix full", "RO full"), ("curriculum-LR full", "RO full"),
             ("TrOCR-hw kidstext 500", "TrOCR-hw RO 500"), ("TrOCR-hw kidstext-pert 500", "TrOCR-hw RO 500"), ("TrOCR-hw x5 500", "TrOCR-hw RO 500"),
             ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500"), ("TrOCR-hw kidstext full", "TrOCR-hw RO full"),
             ("TrOCR-hw kidstext-pert full", "TrOCR-hw RO full"), ("TrOCR-hw x5 full", "TrOCR-hw RO full"), ("TrOCR-s1 kidstext full", "TrOCR-s1 RO full"),
             ("TrOCR-s1 RO 500", "TrOCR-hw RO 500"), ("TrOCR-s1 RO full", "TrOCR-hw RO full"),
             ("scratch cur 500", "scratch RO 500"), ("scratch cur full", "scratch RO full"), ("scratch curpert full", "scratch RO full")]
    for A, B in pairs:
        if A in data and B in data and len(data[A]) > 1 and len(data[B]) > 1:
            for metric in ("cs", "ci", "wer"):
                R["contrasts"].append(dict(A=A, B=B, **contrast(f"{A} | {B} | {metric}", data[A], data[B], nb, metric)))
    # strata, single arm vs its reference (fine strata: G2 = only in this arm's pool)
    single = [("kidstext 500", "RO 500", "500", "kidstext"), ("kidstext-pert 500", "RO 500", "500", "kidstextpert"),
              ("pert+aug F1 500", "RO 500", "500", "kidstextpert"), ("kidstext x5 500", "RO 500", "500", "kidstext30k"),
              ("S1 500", "RO 500", "500", "s1"), ("S1f 500", "RO 500", "500", "s1"), ("S2f 500", "RO 500", "500", "s2"),
              ("S2-415ext 500", "RO 500", "500", "s2x415"), ("S2-extall 500", "RO 500", "500", "s2xall"),
              ("TrOCR-hw kidstext 500", "TrOCR-hw RO 500", "500", "kidstext"), ("TrOCR-hw kidstext-pert 500", "TrOCR-hw RO 500", "500", "kidstextpert"),
              ("TrOCR-hw x5 500", "TrOCR-hw RO 500", "500", "kidstext30k"), ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500", "500", "kidstext")]
    for A, B, n, pool in single:
        if A not in data or B not in data:
            continue
        for ci_ in (False, True):
            g = label(f"single_{pool}_{'ci' if ci_ else 'cs'}")
            masks = {k: g == k for k in ("G1", "G2", "G2b", "G3")}
            masks["G2+G2b"] = (g == "G2") | (g == "G2b")
            metric = "ci" if ci_ else "cs"
            R["strata"].append(dict(kind="single", A=A, B=B, pool=pool, case_insensitive=ci_,
                                    **strata(f"strata | {A} | {B} | {metric}", {A: data[A], B: data[B]}, [(A, B)], masks, nb, metric,
                                             inter=[((A, B), "G2", "G3"), ((A, B), "G1", "G3")])))
    # strata common to several arms (G2 and G2b merged: train text outside the 500 lines)
    multi = [("early stopping", ["RO 500", "S1 500", "kidstext-pert 500"], [("kidstext-pert 500", "RO 500"), ("S1 500", "RO 500"), ("S1 500", "kidstext-pert 500")]),
             ("fixed 20", ["RO 500", "S1f 500", "S2f 500", "S2-415ext 500", "S2-extall 500"],
              [("S2f 500", "RO 500"), ("S1f 500", "RO 500"), ("S1f 500", "S2f 500"), ("S2-415ext 500", "S1f 500"),
               ("S2-extall 500", "S2-415ext 500"), ("S2f 500", "S2-extall 500")]),
             ("Kraken vs TrOCR", ["RO 500", "kidstext 500", "kidstext-pert 500", "kidstext x5 500", "TrOCR-hw RO 500", "TrOCR-hw kidstext 500",
                                  "TrOCR-hw kidstext-pert 500", "TrOCR-hw x5 500", "TrOCR-s1 RO 500", "TrOCR-s1 kidstext 500"],
              [("kidstext 500", "RO 500"), ("kidstext-pert 500", "RO 500"), ("kidstext x5 500", "RO 500"), ("TrOCR-hw kidstext 500", "TrOCR-hw RO 500"),
               ("TrOCR-hw kidstext-pert 500", "TrOCR-hw RO 500"), ("TrOCR-hw x5 500", "TrOCR-hw RO 500"), ("TrOCR-s1 kidstext 500", "TrOCR-s1 RO 500")])]
    for name, arms, cons in multi:
        if not all(k in data for k in arms):
            continue
        for ci_ in (False, True):
            g = label(f"multi_{'ci' if ci_ else 'cs'}")
            masks = {k: g == k for k in ("G1", "G2'", "G3")}
            if not ci_:   # G3 lines with at least one word seen neither in the 500 lines nor in the S2 subsample
                masks["G3 (palabra nueva)"] = label("g3_new_word") == "1"
            metric = "ci" if ci_ else "cs"
            inter = [(c, "G2'", "G3") for c in cons] + [(c, "G1", "G3") for c in cons]
            R["strata"].append(dict(kind="multi", name=name, case_insensitive=ci_,
                                    **strata(f"multi | {name} | {metric}", {k: data[k] for k in arms}, cons, masks, nb, metric, inter)))
    R["g3_words"] = json.load(open(os.path.join(PL, "g3_words.json")))
    for tag, ro, s1, s2 in (("ratio early stopping", "RO 500", "S1 500", "kidstext-pert 500"), ("ratio fixed 20", "RO 500", "S1f 500", "S2f 500")):
        if all(k in data for k in (ro, s1, s2)):
            for metric in ("cs", "ci", "wer"):
                R["ratios"].append(ratio(f"{tag} | {metric}", data[ro], data[s1], data[s2], nb, metric))
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(f"{OUT}/lines.json", "w"), indent=1)
    write_md(R, nb)


def f2(x):
    return f"{x:+.2f}"


def write_md(R, nb):
    o = [f"# Predicciones por línea en CatK12HW-actions: contrastes, estratos y razones (bootstrap de {nb} réplicas)\n",
         "Generador propio por comparación (zlib.crc32 de la etiqueta). IC jerárquico = hojas × semillas, emparejado por hojas. "
         "Cota por escritor = varianza jerárquica + varianza solo de hojas.\n", "## Celdas\n", "| Celda | n | CER-cs | DT | CER-ci | WER |", "|---|---|---|---|---|---|"]
    for k, v in R["cells"].items():
        o.append(f"| {k} | {v['n']} | {v['cs']:.2f} | {'' if v['sd_cs'] is None else f'{v['sd_cs']:.2f}'} | {v['ci']:.2f} | {v['wer']:.2f} |")
    o += ["\n## Contrastes (A − B)\n", "| A − B | métrica | n | Δ | IC Welch (p) | IC hojas × semillas | IC solo hojas | cota escritor | razón var. | hojas A peor |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in R["contrasts"]:
        w = c["welch"]
        o.append(f"| {c['A']} − {c['B']} | {c['metric']} | {c['nA']}/{c['nB']} | {f2(c['d'])} | [{f2(w['lo'])}; {f2(w['hi'])}] ({w['p']:.3g}) | "
                 f"[{f2(c['ci_hier'][0])}; {f2(c['ci_hier'][1])}] | [{f2(c['ci_pages'][0])}; {f2(c['ci_pages'][1])}] | "
                 f"[{f2(c['ci_writer_bound'][0])}; {f2(c['ci_writer_bound'][1])}] | {c['var_ratio']:.2f} | {c['worse_pages']}/24 |")
    o.append("\n## Estratos por exposición textual\n")
    for s in R["strata"]:
        title = f"{s['A']} − {s['B']} (pool {s['pool']})" if s["kind"] == "single" else f"{s['name']}"
        o.append(f"\n### {title}; {'sin caja, CER-ci' if s['case_insensitive'] else 'con caja, CER-cs'}\n")
        cons = list(next(iter(s["strata"].values()))["d"].keys()) if s["strata"] else []
        o.append("| Estrato | líneas | % caract. | " + " | ".join(f"CER {k}" for k in next(iter(s['strata'].values()))['cer']) + " | " + " | ".join(f"Δ {c} [IC] (rel. %)" for c in cons) + " |")
        o.append("|---" * (3 + len(next(iter(s['strata'].values()))['cer']) + len(cons)) + "|")
        for lab, row in s["strata"].items():
            cells = " | ".join(f"{v:.2f}" for v in row["cer"].values())
            ds = " | ".join(f"{f2(x['d'])} [{f2(x['ci'][0])}; {f2(x['ci'][1])}] ({100 * x['rel']:+.0f} %)" for x in row["d"].values())
            o.append(f"| {lab} | {row['lines']} | {row['chars_pct']:.1f} | {cells} | {ds} |")
        for it in s["inter"]:
            o.append(f"\nInteracción {it['contrast']}, {it['strata']}: {f2(it['d'])} [{f2(it['ci'][0])}; {f2(it['ci'][1])}]")
    o.append("\n## Palabras de las líneas G3 (texto exacto ausente del entrenamiento)\n")
    o.append(json.dumps(R["g3_words"], ensure_ascii=False))
    o.append("\n## Razón (S1 − RO) / (S2 − RO)\n\n| Comparación | métrica | razón | IC hojas × semillas | IC solo hojas | IC solo semillas |\n|---|---|---|---|---|---|")
    for r in R["ratios"]:
        o.append(f"| {r['tag'].split(' | ')[0]} | {r['metric']} | {r['ratio']:.2f} | [{r['ci_hier'][0]:.2f}; {r['ci_hier'][1]:.2f}] | "
                 f"[{r['ci_pages'][0]:.2f}; {r['ci_pages'][1]:.2f}] | [{r['ci_seeds'][0]:.2f}; {r['ci_seeds'][1]:.2f}] |")
    open(f"{OUT}/lines.md", "w", encoding="utf-8").write("\n".join(o) + "\n")
    print(f"written {OUT}/lines.md and lines.json")


if __name__ == "__main__":
    main()
