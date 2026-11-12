#!/usr/bin/env python
"""Compare the regenerated tables and figure data (out/expected.json) with the LaTeX sources of the article.

For every table, the data rows (after the first \\midrule) are split into cells and each cell into numeric tokens;
each regenerated cell must match the article cell token by token (a regenerated cell ending in "None" is checked
as a prefix: the rest of the article cell is not derivable from the captured results). Figures are compared on the
coordinates of their pgfplots sources. Prose: every interval "[$a$, $b$]" and every "\\pH = x" outside the tables
must be one of the regenerated values. Nothing is corrected: every mismatch is reported with file and line.
Output: out/check_manuscript.md (exit status 1 if any table or figure cell differs).
Usage: python analysis/check_manuscript.py <paper-repo>   (the repository root that contains ieee-access/)
"""
import json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
PAPER = sys.argv[1]
IA = os.path.join(PAPER, "ieee-access")
X = json.load(open(os.path.join(OUT, "expected.json")))
FILES = [os.path.join(IA, "sections", f) for f in sorted(os.listdir(os.path.join(IA, "sections")))] + [os.path.join(IA, "supplementary.tex")]
TOK = re.compile(r"<0\.001|†|[+-]?\d[\d,]*(?:\.\d+)?")


def strip_group(s, cmd):
    """Remove \\cmd{...} with balanced braces."""
    out, i = [], 0
    while True:
        j = s.find(cmd + "{", i)
        if j < 0:
            return "".join(out) + s[i:]
        out.append(s[i:j]); k = j + len(cmd) + 1; d = 1
        while d and k < len(s):
            d += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        i = k


def clean(s):
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", s)
    for c in ("\\texttt", "\\cite", "\\ref", "\\label"):
        s = strip_group(s, c)
    s = s.replace("CatK12HW", "").replace("\\one", " † ").replace("\\pm", " ± ").replace("\\rightarrow", " → ")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("$", "").replace("{", " ").replace("}", " ").replace("\\,", "").replace("\\%", "%")
    return s


def norm(t):
    t = t.lstrip("+")
    return "0" + t[2:] if t.startswith("-0") and set(t[1:]) <= set("0.") else t


def toks(s):
    return [norm(t) for t in TOK.findall(clean(s))]


def split_top(s, sep):
    out, d, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
        if d == 0 and s.startswith(sep, i):
            out.append("".join(cur)); cur = []; i += len(sep); continue
        cur.append(ch); i += 1
    out.append("".join(cur))
    return out


def read(f):
    return open(f, encoding="utf-8").read()


def find_table(label):
    for f in FILES:
        s = read(f)
        i = s.find("\\label{" + label + "}")
        if i < 0:
            continue
        b = s.rfind("\\begin{table", 0, i); e = s.find("\\end{table", i)
        return f, s, b, e
    return None


def lineno(s, pos):
    return s.count("\n", 0, pos) + 1


def strip_comments(s):
    return re.sub(r"(?<!\\)%[^\n]*", "", s)


report, bad, checked = ["# Comparison of the regenerated values with the manuscript\n"], 0, 0
for label, spec in X.items():
    if "rows" not in spec:
        continue
    hit = find_table(label)
    if not hit:
        report.append(f"\n## {label}: NOT FOUND in the manuscript"); bad += 1; continue
    f, s, b, e = hit
    rel = os.path.relpath(f, PAPER)
    tab0 = s.find("\\begin{tabular}", b); tab1 = s.find("\\end{tabular}", tab0)
    mid = s.find("\\midrule", tab0)
    body_start = mid + len("\\midrule")
    body = s[body_start:tab1]
    rows, pos = [], body_start
    for chunk in split_top(body, "\\\\"):
        start = pos; pos += len(chunk) + 2
        c = strip_comments(chunk)
        c = re.sub(r"\\(midrule|toprule|bottomrule)|\\cmidrule\([^)]*\)\{[^}]*\}|\\cmidrule\{[^}]*\}|\\addlinespace(\[[^\]]*\])?", "", c).strip()
        if not c:
            continue
        rows.append((lineno(s, start + len(chunk) - len(chunk.lstrip())), split_top(c, "&")))
    exp = spec["rows"]
    out = [f"\n## {label} ({rel}, line {lineno(s, b)})\n", f"Source: {spec['source']}\n"]
    nbad = 0
    if len(rows) != len(exp):
        out.append(f"- ROW COUNT: manuscript {len(rows)}, regenerated {len(exp)}"); nbad += 1
    for (ln, cells), ecells in zip(rows, exp):
        for k, ec in enumerate(ecells):
            if ec is None:
                continue
            if k >= len(cells):
                out.append(f"- line {ln}, cell {k + 1}: missing in the manuscript (regenerated `{ec}`)"); nbad += 1; continue
            pt, et = toks(cells[k]), toks(ec.replace("None", "")) if "None" in ec else toks(ec)
            prefix = "None" in ec
            checked += 1
            ok = pt[:len(et)] == et if prefix else pt == et
            if not ok:
                out.append(f"- line {ln}, cell {k + 1}: manuscript `{' '.join(pt)}` vs regenerated `{' '.join(et)}`"); nbad += 1
    if spec.get("caption"):
        cap0 = s.find("\\caption{", b); cap1 = s.find("\\label{", cap0)
        ct = toks(s[cap0:cap1]); checked += 1
        miss = [t for t in (norm(x) for x in spec["caption"]) if t not in ct]
        if miss:
            out.append(f"- caption (line {lineno(s, cap0)}): regenerated values not found: {miss}"); nbad += 1
    out.insert(2, f"Result: {'OK' if not nbad else f'{nbad} discrepancies'}\n")
    report += out; bad += nbad

# ------------------------------------------------------------------ figures (pgfplots sources)
SRC = os.path.join(IA, "figures", "src")


def coords(path):
    s = read(path)
    plots = []
    for m in re.finditer(r"\\addplot\[[^\]]*\]\s*coordinates\s*\{(.*?)\};", s, re.S):
        pts = re.findall(r"\(([^,()]+),([^()]+)\)(?:\s*\+-\s*\(0,([^()]+)\))?(?:\s*\[([^\]]+)\])?", m.group(1))
        plots.append([(a.strip(), b.strip(), c.strip() or None) for a, b, c, _ in pts])
    return plots


def fcmp(name, got, exp):
    global bad, checked
    out = [f"\n## {name} (ieee-access/figures/src/{os.path.basename(name)})\n"]
    n = 0
    for (lab, gy, ge), (elab, ey, ee) in zip(got, exp):
        checked += 1
        ok = abs(float(gy) - ey) < 0.0051 and (ee is None or ge is None and ee is None or (ge is not None and abs(float(ge) - ee) < 0.0051))
        if ee is not None and ge is None:
            ok = False
        if not ok:
            out.append(f"- {elab}: figure ({gy}, ±{ge}) vs regenerated ({ey:.2f}, ±{ee})"); n += 1
    if len(got) != len(exp):
        out.append(f"- POINT COUNT: figure {len(got)}, regenerated {len(exp)}"); n += 1
    out.insert(1, f"Result: {'OK' if not n else f'{n} discrepancies'}\n")
    report.extend(out); bad += n


P = coords(os.path.join(SRC, "delta_full.tex"))
got = [p for pl in P for p in pl]
exp = [(f"{q['lang']} {q['campaign']} {q['protocol']}", q["d"], q["se"]) for l in ("fr", "en", "ca") for c in ("c1", "c2")
       for q in X["fig:deltafull"]["points"] if q["lang"] == l and q["campaign"] == c]
fcmp("delta_full.tex", got, exp)
P = coords(os.path.join(SRC, "curves.tex"))
got = [p for pl in P for p in pl]
exp = [(f"{q['lang']} {q['x']}", q["mean"], q["sd"]) for q in X["fig:curves"]["points"]]
fcmp("curves.tex", got, exp)
P = coords(os.path.join(SRC, "delta_size.tex"))
got = [p for pl in P for p in pl]
exp = [(f"{q['lang']} {q['campaign']} {q['size']}", q["d"], q["se"]) for l in ("fr", "en", "ca") for c in ("c1", "c2")
       for q in X["fig:s-size"]["points"] if q["lang"] == l and q["campaign"] == c]
fcmp("delta_size.tex", got, exp)
P = coords(os.path.join(SRC, "errors.tex"))
got = [(b, a, None) for pl in P for a, b, _ in pl]
E = {q["label"]: q for q in X["fig:s-errors"]["points"]}
exp = [(f"{lab} {k}", E[lab][k], None) for k in ("deletions", "insertions", "substitutions") for lab in (g[0] for g in got[:6])]
fcmp("errors.tex", got, exp)

# ------------------------------------------------------------------ prose: intervals and adjusted p values
H = json.load(open(os.path.join(OUT, "holm_family.json")))["contrasts"]
L = json.load(open(os.path.join(OUT, "lines.json")))
iv, phs = set(), set()


def add(lo, hi):
    if lo is None or hi is None or (isinstance(lo, float) and (math.isnan(lo) or math.isnan(hi))):
        return
    iv.add((norm(f"{lo:+.2f}"), norm(f"{hi:+.2f}")))


def fph(p):
    return "1" if p >= 0.995 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}" if p >= 0.001 else "0.001"


for r in H:
    for k in ("cer_cs", "cer_ci", "wer_cs"):
        if f"{k}_lo" in r:
            add(r[f"{k}_lo"], r[f"{k}_hi"])
    if "holm" in r:
        phs.add(fph(r["holm"]))
for c in L["contrasts"]:
    add(c["welch"]["lo"], c["welch"]["hi"])
    for k in ("ci_hier", "ci_pages", "ci_writer_bound"):
        add(*c[k])
for s_ in L["strata"]:
    for row in s_["strata"].values():
        for d in row["d"].values():
            add(*d["ci"])
    for it in s_["inter"]:
        add(*it["ci"])
for extra in json.load(open(os.path.join(OUT, "prose_values.json"))).get("intervals", []) if os.path.exists(os.path.join(OUT, "prose_values.json")) else []:
    add(*extra)
report.append("\n## Prose (sections and supplementary, outside tables)\n")
miss = []
for f in FILES:
    s = read(f)
    spans = [(m.start(), m.end()) for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", s, re.S)]
    inside = lambda p: any(a <= p < b for a, b in spans)
    for m in re.finditer(r"\[(?:95\\,\\% CI )?\$([+-]?\d+\.\d+)\$, \$([+-]?\d+\.\d+)\$\]", s):
        if inside(m.start()):
            continue
        checked += 1
        if (norm(m.group(1)), norm(m.group(2))) not in iv:
            miss.append(f"- {os.path.relpath(f, PAPER)}:{lineno(s, m.start())}: interval [{m.group(1)}, {m.group(2)}] not among the regenerated intervals")
    for m in re.finditer(r"\\pH (?:=|<|\\leq) ([\d.]+)", s):
        if inside(m.start()):
            continue
        checked += 1
        if m.group(1) not in phs:
            miss.append(f"- {os.path.relpath(f, PAPER)}:{lineno(s, m.start())}: p_H {m.group(1)} not among the regenerated Holm values")
report += miss or ["All intervals and adjusted p values found."]
report.insert(1, f"Cells, captions, figure points and prose values checked: {checked}. Table/figure discrepancies: {bad}. "
                 f"Prose values not regenerated: {len(miss)}.\n")
open(os.path.join(OUT, "check_manuscript.md"), "w", encoding="utf-8").write("\n".join(report) + "\n")
print("\n".join(report[:2]))
sys.exit(1 if bad else 0)
