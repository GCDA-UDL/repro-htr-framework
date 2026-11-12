#!/usr/bin/env python
"""Summary of a TrOCR arm root (default models/trocr): CER-cs mean +/- sd per cell from <root>/block*/*/test_results.json,
and curriculum <variant> vs real-only at each size (delta, SE, t), like the Kraken tables of eval_register.py."""
import glob, json, os, re, sys, statistics as st
ROOT = sys.argv[1] if len(sys.argv) > 1 else "models/trocr"
rows = {}
for f in sorted(glob.glob(f"{ROOT}/block*/*/test_results.json")):
    d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
    m = re.match(r"ca_(baseline|synth-only|real-only|curriculum)(?:_(500|1k|2k))?(?:_(kidstext30k|kidstextpert|kidstext))?(?:_seed(\d+))?$", name)
    if not m:
        print("skip", name); continue
    prot, size, var, seed = m.group(1), m.group(2) or "full", m.group(3), int(m.group(4) or 1)
    cell = "zero-shot" if prot == "baseline" else f"{prot}{'_' + var if var else ''} {size}"
    rows.setdefault(cell, []).append((seed, d["cer_cs"], d["cer_ci"]))
def fmt(v): return f"{st.mean(v):.2f} +/- {st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"
def lst(v): return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
print(f"=== TrOCR arm {ROOT}, ca (base: see run_info.json; greedy, NFD; CER-cs mean +/- sd [seeds]; CER-ci mean) ===")
for cell in sorted(rows):
    r = sorted(rows[cell]); cs, ci = [x[1] for x in r], [x[2] for x in r]
    print(f"{cell:28s}: n={len(cs)}  CER-cs {fmt(cs):15s} {lst(cs):44s}  CER-ci {st.mean(ci):.2f}")
print("\n=== curriculum vs real-only (delta = curriculum - real-only; SE = sqrt(sd_c^2/n_c + sd_r^2/n_r); t = delta/SE) ===")
for var in ("kidstext", "kidstext30k", "kidstextpert"):
    for size in ("500", "full"):
        ro = [x[1] for x in rows.get(f"real-only {size}", [])]; cu = [x[1] for x in rows.get(f"curriculum_{var} {size}", [])]
        if not cu:
            continue
        if len(ro) > 1 and len(cu) > 1:
            d = st.mean(cu) - st.mean(ro); se = (st.variance(cu) / len(cu) + st.variance(ro) / len(ro)) ** 0.5
            print(f"{var:12s} {size:>4}: real-only {fmt(ro):15s} curriculum {fmt(cu):15s} delta {d:+.2f}  SE {se:.2f}  t {d / se:+.1f}  (n {len(cu)} vs {len(ro)})")
        else:
            print(f"{var:12s} {size:>4}: real-only n={len(ro)} {lst(ro)}  curriculum n={len(cu)} {lst(cu)}  (more than one seed per arm needed)")
