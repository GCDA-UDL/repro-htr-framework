#!/usr/bin/env python3
"""Evaluate finished runs on the test set, register them in results/all_results.csv and print summary tables.

Campaign 2 (05/10/2026 ->): synthetic pools with paper matched to the real data (fr/en/ca *_white_30k).
Campaign 1 (mixed paper backgrounds) is archived in _archive/results/all_results_mixedbg.csv and models/mixedbg/.

Runs covered (folder -> exp_id), every environment with explicit metadata (table ENV, no template rows):
  models/blockA/<lang>_<protocol>[_<tag>][_seed<S>]         Block A: E02 synth-only, E03 real-only, E04 mix, E05 curriculum,
                                              E06 curriculum-LR. fr = RIMES, en = IAM, ca = CatK12HW.
                                              Folder tags: base-v2 / altbase-v2 = generic base FoNDUE-GD_v2 (ca only,
                                              ids E<nn>v2[sS]_ca, exp_name *_altbase-v2); a key of VARIANTS = synthetic-
                                              content variant (Article 2), ids E<nn>v<k>[sS]_<lang>, exp_name *_<tag>.
                                              Unknown tags are reported and skipped.
  models/blockC/<lang>_<protocol>_<size>[_<tag>][_seed<S>]  Block C: E11/E13/E15/E17 real-only 500/1k/2k/4k, E12/E14/E16/E18
                                              curriculum; same tags.
  Seed 1 (or no seed) -> E<nn>_<lang>; seed S > 1 -> E<nn>sS_<lang>. The fonts count of a pool is read from
  code/framework/output_<pool>/dataset_info.json (num_fonts); a run whose pool has no dataset_info.json is not registered.
A run is skipped when a ketos training process is still writing into it, or when its exp_id is already in the CSV
(the CSV values are shown instead). Otherwise:
  1. best checkpoint = highest validation accuracy in the file name (Block A curriculum*: phase2/m). Files that do not
     match checkpoint_<epoch>-<acc>.ckpt (e.g. checkpoint_abort.ckpt) and files older than the start of the last launch
     recorded in run_info.txt (leftovers of an aborted earlier attempt) are listed and ignored.
  2. the run must have finished: ketos writes m/best_<acc>.safetensors when training ends. Runs without that file are
     reported and skipped (--allow-incomplete evaluates them but never registers them).
  3. `ketos test -f binary -u NFD` on the language's test arrow -> eval_full.log (reused when it already holds a report
     for the same checkpoint; a log with a traceback is never used).
  4. register_result.py with the fields of the protocol (ENV / VARIANTS). eval_pipeline is set to ketos_test.
Usage: python code/scripts/eval_register.py [--dry-run] [--device cuda:0] [--allow-incomplete]
"""
import argparse, csv, datetime as dt, glob, json, os, re, statistics, subprocess

ROOT = os.path.expanduser("~/workrepo-htr-synthetic-framework")
CSV = "results/all_results.csv"
TEST = {"ca": "data/real/catk12hw/catk12hw_test.arrow",
        "fr": "data/real/rimes/rimes_test.arrow",
        "en": "data/real/iam/iam_test.arrow"}
A_IDS = {"synth-only": "E02", "real-only": "E03", "mix": "E04", "curriculum": "E05", "curriculum-LR": "E06"}
C_IDS = {"500": {"real-only": "E11", "curriculum": "E12"}, "1k": {"real-only": "E13", "curriculum": "E14"},
         "2k": {"real-only": "E15", "curriculum": "E16"}, "4k": {"real-only": "E17", "curriculum": "E18"}}
SIZE_N = {"500": 500, "1k": 1000, "2k": 2000, "4k": 4000}
# Environments (campaign 2 pools). fonts = num_fonts of the pool, read from its dataset_info.json at run time.
ENV = {
    "fr": dict(base="FoNDUE-GD_v2_fr", alt_base=None, real="rimes", real_desc="RIMES", n=10188,
               synth="fr_white_30k", cat="Handwritten", env="1st env"),
    "en": dict(base="FoNDUE-GD_v2", alt_base=None, real="iam", real_desc="IAM", n=8917,
               synth="en_white_30k", cat="Handwritten", env="2nd env"),
    "ca": dict(base="FoNDUE-GD_v2_fr", alt_base="FoNDUE-GD_v2", real="catk12hw", real_desc="CatK12HW", n=4152,
               synth="ca_white_30k", cat="Handwritten", env="3rd env"),
}
# Synthetic-content variants (Article 2): folder tag -> id mark and the pool that replaces the environment's default
VARIANTS = {
    "fr": {}, "en": {},
    "ca": {   # Article 2 content cells (white plain paper, 6000 images, 1:1 subsample 4152): ids E<nn>v<k>[sS]_ca
        "school":     dict(v="v3", synth="ca_school_white_6k",     cat="School",      desc="School fonts (DaFont category, Catalan glyph check)"),
        "kids":       dict(v="v4", synth="ca_kids_white_6k",       cat="KidsCurated", desc="KidsCurated fonts (25 child-like fonts chosen by hand)"),
        "hw25":       dict(v="v5", synth="ca_hw25_white_6k",       cat="Handwritten", desc="Handwritten, 25 random fonts (font-count control)"),
        "text":       dict(v="v6", synth="ca_text_white_6k",       cat="Handwritten", desc="Handwritten fonts, matched text (CatK12HW training transcriptions)"),
        "schooltext": dict(v="v7", synth="ca_schooltext_white_6k", cat="School",      desc="School fonts, matched text"),
        "kidstext":   dict(v="v8", synth="ca_kidstext_white_6k",   cat="KidsCurated", desc="KidsCurated fonts, matched text"),
        "kidstext30k": dict(v="v11", synth="ca_kidstext_white_30k", cat="KidsCurated", desc="KidsCurated fonts, matched text, 30k pool (5x volume; phase1 on the full train split)"),
        "kidstextpert": dict(v="v12", synth="ca_kidstextpert_white_6k", cat="KidsCurated", q="Degraded", desc="KidsCurated fonts, matched text, --perturbations (clean/degraded/severe 40/40/20)"),
        "kidstextpert-aug": dict(v="v13", synth="ca_kidstextpert_white_6k", cat="KidsCurated", q="Degraded", desc="KidsCurated fonts, matched text, --perturbations; phase 1 shared with kidstextpert, phase 2 on real with ketos --augment"),
        "kidstextpert-augp1": dict(v="v17", synth="ca_kidstextpert_white_6k", cat="KidsCurated", q="Degraded", desc="KidsCurated fonts, matched text, --perturbations; ketos --augment in phase 1 (synthetic) only, phase 2 standard"),
    },
}


def _fonts_of_json(pool):
    """num_fonts of a pool from code/framework/output_<pool>/dataset_info.json, or None when the pool is not generated."""
    p = f"code/framework/output_{pool}/dataset_info.json"
    if not os.path.exists(p):
        return None
    return str(json.load(open(p, encoding="utf-8")).get("num_fonts"))
A_NOTE = {
    "synth-only":    "Block A synth-only ({env}): {synth} 1:1 ({n}), no real, seed {s}. ketos test -u NFD best ckpt (val {v}). {d}",
    "real-only":     "Block A real-only ({env}): {real} train {n}, seed {s}. ketos test -u NFD best ckpt (val {v}). {d}",
    "mix":           "Block A mix ({env}): {real} {n} + {synth} {n}, single phase, seed {s}. ketos test -u NFD best ckpt (val {v}). {d}",
    "curriculum":    "Block A curriculum ({env}): phase1 synth {n} -> phase2 real {n}, both lr 1e-4, seed {s}. ketos test phase2 best ckpt (val {v}). {d}",
    "curriculum-LR": "Block A curriculum-LR ({env}): phase1 synth {n} -> phase2 real {n} lr 1e-5, seed {s}. ketos test phase2 best ckpt (val {v}). {d}",
}
C_NOTE = {
    "real-only":  "Block C real-only arm ({env}), real={n}, no synthetic, seed {s}. ketos test -u NFD best ckpt (val {v}). {d}",
    "curriculum": "Block C curriculum arm ({env}), real={n}, phase1 synth {nsyn} reused from blockA {p1}, lr 1e-4. ketos test best ckpt (val {v}). {d}",
}
SCRATCH2_TAGS = {  # from-scratch recipe-2 folder tags -> (id mark, phase-1 pool, quality level)
    "scratchr2": ("v10", "ca_white_30k", "Clean"), "scratchr2-white30k": ("v10", "ca_white_30k", "Clean"),
    "scratchr2-2stage": ("v15", "ca_white_30k", "Clean"),            # control: real-only r2 + a second stage with the phase-2 recipe
    "scratchr2-whitepert30k": ("v16", "ca_whitepert_30k", "Degraded")}  # quality cell: generic pool regenerated with --perturbations
SCRATCH2_SYN = "data/synthetic/ca_white_30k/ca_white_30k_train_bl.arrow"   # phase1 of the from-scratch recipe-2 curriculum: full train split


def arrow_rows(path):
    """Row count of a Kraken arrow (memory-mapped, no copy)."""
    import pyarrow as pa
    return pa.ipc.open_file(pa.memory_map(path)).read_all().num_rows


def set_lr(fields, flag, value):
    out = list(fields)
    for i, x in enumerate(out[:-1]):
        if x == flag:
            out[i + 1] = value
    return out


TAG = r"(?:_(?!seed\d+(?:_|$))([A-Za-z0-9-]+?))?"          # optional folder tag that can never swallow _seed<S>
A_RE = re.compile(r"(fr|en|ca)_(synth-only|real-only|mix|curriculum|curriculum-LR)" + TAG + r"(?:_seed(\d+))?$")
C_RE = re.compile(r"(fr|en|ca)_(real-only|curriculum)_(500|1k|2k|4k)" + TAG + r"(?:_seed(\d+))?$")
CKPT_RE = re.compile(r"checkpoint_(\d+)-([\d.]+)\.ckpt$")
REPORT_RE = re.compile(r"^\s*(-?[\d.]+)%?\s+(Characters|Errors|Character Accuracy|Character Accuracy \(Case-insensitive\)|Word Accuracy)\s*$")
NEED = ("Characters", "Errors", "Character Accuracy", "Character Accuracy (Case-insensitive)", "Word Accuracy")


def run_start(run):
    """Timestamp of the last launch recorded in run_info.txt (None when unknown) and its gpu= field."""
    ri = os.path.join(run, "run_info.txt")
    start, gpu = None, "cuda:?"
    if os.path.exists(ri):
        for line in open(ri, encoding="utf-8", errors="replace"):
            m = re.search(r"date=(\S+)", line)
            if m:
                try:
                    start = dt.datetime.fromisoformat(m.group(1)).timestamp()
                except ValueError:
                    pass
                g = re.search(r"gpu=(\S+)", line)
                gpu = g.group(1) if g else gpu
    return start, gpu


def checkpoints(ckdir, start):
    """(valid, ignored): valid = [(path, acc, epoch)] matching the name pattern and not older than the run start."""
    valid, ignored = [], []
    for f in sorted(glob.glob(os.path.join(ckdir, "checkpoint_*.ckpt"))):
        m = CKPT_RE.search(f)
        if not m:
            ignored.append(os.path.basename(f) + " (name)"); continue
        if start is not None and os.path.getmtime(f) < start:
            ignored.append(os.path.basename(f) + " (older than run start)"); continue
        valid.append((f, float(m.group(2)), int(m.group(1))))
    return valid, ignored


def finished(ckdir, start):
    best = [f for f in glob.glob(os.path.join(ckdir, "best_*")) if start is None or os.path.getmtime(f) >= start]
    return sorted(best)


def parse_report(path):
    vals, ckpt = {}, None
    if not os.path.exists(path):
        return None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("# checkpoint: "):
            ckpt = line[len("# checkpoint: "):].strip()
        m = REPORT_RE.match(line)
        if m:
            vals[m.group(2)] = float(m.group(1))
    return (vals if all(k in vals for k in NEED) else None), ckpt


def has_error(path):
    txt = open(path, encoding="utf-8", errors="replace").read().lower() if os.path.exists(path) else ""
    return any(k in txt for k in ("traceback", "runtimeerror", "exception"))


def busy_dirs():
    out = subprocess.run(["pgrep", "-af", "ketos_safe"], capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if " train " in l]           # training processes only
    return "\n".join(re.sub(r"\s-i\s+\S+", " ", l) for l in lines)  # without the -i (loaded checkpoint) path


def csv_rows():
    if not os.path.exists(CSV):
        return []
    return list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))


def set_eval_pipeline(ids):
    rows = csv_rows(); n = 0
    for r in rows:
        if r["exp_id"] in ids and not r.get("eval_pipeline"):
            r["eval_pipeline"] = "ketos_test"; n += 1
    if n:
        with open(CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n


def fonts_of(pool):
    """Fonts count of a synthetic pool: data/synthetic/<pool>/dataset_info.json, else code/framework/output_<pool>/dataset_info.json
    (the original source; those folders were deleted on 18/10/2026 after backing up the pools), else the 'fonts' column of a row
    already registered with the same dataset_synth in the CSV."""
    import json as _json
    p = f"data/synthetic/{pool}/dataset_info.json"
    if os.path.exists(p):
        try:
            v = _json.load(open(p)).get("num_fonts")
            if v: return str(v)
        except Exception:
            pass
    v = _fonts_of_json(pool)
    if v is not None:
        return v
    for r in csv_rows():
        if r.get("dataset_synth") == pool and (r.get("fonts_count") or r.get("fonts")):
            return r.get("fonts_count") or r.get("fonts")
    return None


def tag_info(lang, tag):
    """Folder tag -> (alt_base, variant) or None when the tag is unknown for this language."""
    if not tag:
        return False, None
    if tag in ("base-v2", "altbase-v2"):
        return (True, None) if ENV[lang]["alt_base"] else None
    if tag == "scratch" and lang == "ca":
        return "scratch", None
    if tag == "scratch-kidstext" and lang == "ca":
        return "scratch", "kidstext"
    if tag == "aug":
        return "aug", None
    if tag in SCRATCH2_TAGS and lang == "ca":
        return "scratch2", None
    if tag in VARIANTS.get(lang, {}):
        return False, tag
    return None


def strip_lr(fields, phase1, phase2):
    """Drop lr flags that do not apply: from-scratch steps use Kraken's default lr, not a fixed value."""
    drop = ({"--phase1-lr"} if phase1 else set()) | ({"--phase2-lr"} if phase2 else set())
    out, skip = [], False
    for x in fields:
        if skip:
            skip = False; continue
        if x in drop:
            skip = True; continue
        out.append(x)
    return out


def env_fields(lang, alt, var, tag=None):
    """(common options, synthetic options, id mark, name suffix, env description, pool name) for ca runs."""
    e = ENV[lang]
    base = e["base"] if alt == "aug" else "none-scratch-r2" if alt == "scratch2" else "none-scratch" if alt == "scratch" else (e["alt_base"] if alt else e["base"])
    pool = VARIANTS[lang][var] if var else e
    if alt == "scratch2":
        pool = dict(pool, synth=SCRATCH2_TAGS[tag][1], q=SCRATCH2_TAGS[tag][2])
    common = ["--language", lang, "--status", "completed", "--model-base", base, "--decoder", "greedy_ctc", "--filter-qmark", "no"]
    nf = fonts_of(pool["synth"])
    n_syn = arrow_rows(f"data/synthetic/{pool['synth']}/{pool['synth']}_train_bl.arrow") if alt == "scratch2" else arrow_rows(f"data/synthetic/{pool['synth']}/{pool['synth']}_train_bl.arrow") if var == "kidstext30k" else e["n"]
    syn = None if nf is None else ["--dataset-synth", pool["synth"], "--synth-samples-count", str(n_syn), "--fonts-count", nf,
                                   "--font-category", pool["cat"], "--quality-level", pool.get("q", "Clean")]
    mark = "v14" if alt == "aug" else SCRATCH2_TAGS[tag][0] if alt == "scratch2" else "v9" if alt == "scratch" else "v2" if alt else (VARIANTS[lang][var]["v"] if var else "")
    sfx = "_aug" if alt == "aug" else f"_{tag}" if alt == "scratch2" else ("_scratch" + (f"-{var}" if var else "")) if alt == "scratch" else "_altbase-v2" if alt else (f"_{var}" if var else "")
    env = e["env"] + ((", ketos --augment on the real training lines" if alt == "aug" else ", trained from scratch, recipe 2 (no base model; lr 1e-3 warmup + reduceonplateau, B16, augment, lag 20 / min 50; phase1 = full " + pool["synth"] + " train)" if alt == "scratch2" else ", trained from scratch (no base model)" if alt == "scratch" else f", alt base {base}") if alt else "") + (f", synthetic content: {VARIANTS[lang][var]['desc']}" if var else "")
    return common, syn, mark, sfx, env, pool["synth"]


def runs(by_id):
    """Yield dicts describing every run folder handled by this script."""
    # Block A
    for run in sorted(glob.glob("models/blockA/*")):
        m = A_RE.match(os.path.basename(run))
        if not m or not os.path.isdir(run):
            continue
        lang, prot, tag = m.group(1), m.group(2), m.group(3)
        seed = int(m.group(4)) if m.group(4) else 1
        ti = tag_info(lang, tag)
        if ti is None:
            print(f"[{os.path.basename(run)}] unknown folder tag '{tag}' for {lang} - skipped"); continue
        alt, var = ti
        e = ENV[lang]
        common, syn, mark, sfx, env, synth = env_fields(lang, alt, var, tag)
        real = ["--dataset-real", e["real"], "--real-train-size", str(e["n"])]
        if syn is None and prot != "real-only":
            a_fields = None
        else:
            a_fields = {"synth-only": (syn or []) + ["--phase2-lr", "0.0001"],
                        "real-only": real + ["--phase2-lr", "0.0001"],
                        "mix": real + (syn or []) + ["--phase2-lr", "0.0001"],
                        "curriculum": real + (syn or []) + ["--phase1-lr", "0.0001", "--phase2-lr", "0.0001"],
                        "curriculum-LR": real + (syn or []) + ["--phase1-lr", "0.0001", "--phase2-lr", "0.00001"]}[prot]
        if alt == "scratch" and a_fields is not None:
            a_fields = strip_lr(a_fields, phase1=True, phase2=(prot == "real-only"))
        note = A_NOTE[prot].replace("{env}", env).replace("{synth}", synth).replace("{real}", e["real_desc"])
        if alt == "scratch":
            note = note.replace("both lr 1e-4", "phase1 from scratch (Kraken default lr, lag 10, min-epochs 20), phase2 lr 1e-4") + (
                " From scratch: Kraken default lr, lag 10, min-epochs 20." if prot == "real-only" else "")
        if alt == "scratch2":
            if a_fields is not None:
                a_fields = set_lr(a_fields, "--phase1-lr" if prot.startswith("curriculum") else "--phase2-lr", "0.001")
            note = note.replace("phase1 synth {n}", f"phase1 synth (full {synth} train split)").replace(
                "both lr 1e-4", "phase1 recipe 2 (lr 1e-3 warmup + reduceonplateau, B16, augment, lag 20 / min 50), phase2 lr 1e-4") + (
                " From scratch, recipe 2: lr 1e-3 warmup + reduceonplateau, B16, augment, lag 20 / min 50." if prot == "real-only" else "")
        if tag == "scratchr2-2stage":
            note += (" Two-stage control: the real-only recipe-2 model of the same seed (E03v10) trained again on the same real lines with the"
                     " phase-2 recipe (lr 1e-4, lag 5, min-epochs 3, B1, resize union); isolates the effect of a second training stage from that of the synthetic phase 1.")
        if tag == "scratchr2-whitepert30k":
            note += " Phase 1 on ca_whitepert_30k: the generic base pool regenerated with --perturbations (clean/degraded/severe 40/40/20)."
        if alt == "aug":
            note += " ketos --augment on the real training lines."
        yield dict(run=run, lang=lang, prot=prot, size=None, seed=seed, n=e["n"],
                   exp_id=f"{A_IDS[prot]}{mark}{'' if seed == 1 else f's{seed}'}_{lang}",
                   exp_name=f"{lang}_{prot}{sfx}" + ("" if seed == 1 else f"_seed{seed}"),
                   ckdir=os.path.join(run, "phase2", "m") if prot.startswith("curriculum") else os.path.join(run, "m"),
                   fields=None if a_fields is None else ["--protocol", prot] + common + a_fields, note=note, pool=synth)
    # Block C
    for run in sorted(glob.glob("models/blockC/*_*")):
        m = C_RE.match(os.path.basename(run))
        if not m or not os.path.isdir(run):
            continue
        lang, prot, size, tag = m.group(1), m.group(2), m.group(3), m.group(4)
        seed = int(m.group(5)) if m.group(5) else 1
        sfx = "" if seed == 1 else f"s{seed}"
        n = SIZE_N[size]
        ti = tag_info(lang, tag)
        if ti is None:
            print(f"[{os.path.basename(run)}] unknown folder tag '{tag}' for {lang} - skipped"); continue
        alt, var = ti
        e = ENV[lang]
        common, syn, mark, nsfx, env, synth = env_fields(lang, alt, var, tag)
        if prot == "real-only":
            fields = ["--protocol", prot] + common + ["--dataset-real", e["real"], "--real-train-size", str(n), "--phase2-lr", "0.0001"]
        elif syn is None:
            fields = None
        else:
            fields = ["--protocol", prot] + common + ["--dataset-real", e["real"], "--real-train-size", str(n)] + syn + [
                "--phase1-lr", "0.0001", "--phase2-lr", "0.0001"]
        if alt == "scratch" and fields is not None:
            fields = strip_lr(fields, phase1=True, phase2=(prot == "real-only"))
        if alt == "scratch2" and fields is not None:
            fields = set_lr(fields, "--phase2-lr" if prot == "real-only" else "--phase1-lr", "0.001")
        p1 = (("ca_curriculum_scratch-kidstext phase1 " if alt == "scratch" else f"ca_curriculum_{tag} phase1 " if alt == "scratch2" else "base-v2 ") if alt else (f"{lang}_curriculum_{var} " if var else "")) + "seed {s}"
        note = C_NOTE[prot].replace("{env}", env).replace("{nsyn}", str(arrow_rows(f"data/synthetic/{synth}/{synth}_train_bl.arrow") if var == "kidstext30k" or alt == "scratch2" else e["n"])).replace("{p1}", p1)
        if alt == "scratch":
            note += " From scratch (no base model): real-only / phase1 Kraken default lr, lag 10, min-epochs 20; phase2 lr 1e-4."
        if alt == "scratch2":
            note += " From scratch, recipe 2 (no base model; lr 1e-3 warmup + reduceonplateau, B16, augment, lag 20 / min 50); phase2 lr 1e-4, resize union."
            if tag == "scratchr2-2stage":
                note += " Two-stage control: the real-only-500 recipe-2 model of the same seed (E11v10) trained again on the same 500 lines with the phase-2 recipe."
        if alt == "aug":
            note += " ketos --augment on the real training lines."
        exp_name = f"{lang}_{prot.replace('-', '')}_{size}{nsfx}" + ("" if seed == 1 else f"_seed{seed}")
        exp_id = f"{C_IDS[size][prot]}{mark}{sfx}_{lang}"
        yield dict(run=run, lang=lang, prot=prot, size=size, seed=seed, n=n, exp_id=exp_id,
                   exp_name=exp_name, ckdir=os.path.join(run, "m"), fields=fields, note=note, pool=synth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-incomplete", action="store_true", help="evaluate (never register) runs without best_* file")
    a = ap.parse_args()
    os.chdir(ROOT)
    running = busy_dirs()
    rows = csv_rows()
    by_id = {r["exp_id"]: r for r in rows}
    registered = []
    for r in runs(by_id):
        name, exp_id = os.path.basename(r["run"]), r["exp_id"]
        if f"{r['run']}/" in running:
            print(f"[{name}] RUNNING - skipped"); continue
        if exp_id in by_id:
            c = by_id[exp_id]
            print(f"[{name}] {exp_id} already in CSV: CER-cs {c['cer_cs']} CER-ci {c['cer_ci']} WER-cs {c['wer_cs']} (best_stage {c['best_stage']}, val {c['best_val_acc']})")
            continue
        start, gpu = run_start(r["run"])
        valid, ignored = checkpoints(r["ckdir"], start)
        for ig in ignored:
            print(f"[{name}] ignored {r['ckdir']}/{ig}")
        if not valid:
            print(f"[{name}] no usable checkpoint in {r['ckdir']} (not started or aborted?) - skipped"); continue
        ckpt, val, epoch = max(valid, key=lambda t: t[1])
        last = max(t[2] for t in valid)
        done = finished(r["ckdir"], start)
        if not done:
            print(f"[{name}] NOT FINISHED: no best_* file in {r['ckdir']} (newest kept epoch {last}, best {epoch}, val {val})"
                  + ("" if a.allow_incomplete else " - skipped"))
            if not a.allow_incomplete:
                continue
        log = os.path.join(r["run"], "eval_full.log")
        rep, logck = parse_report(log)
        if rep is not None and logck is not None and logck != ckpt:
            print(f"[{name}] eval_full.log was made from {os.path.basename(logck)}, best is now {os.path.basename(ckpt)} - re-evaluating")
            rep = None
        if rep is None:
            cmd = ["python", "code/scripts/ketos_safe.py", "-d", a.device, "test", "-m", ckpt, "-f", "binary", "-u", "NFD", TEST[r["lang"]]]
            print(f"[{name}] evaluating {os.path.basename(ckpt)} (epoch {epoch}, val {val}) on {TEST[r['lang']]} -> {log}")
            if a.dry_run:
                continue
            with open(log, "w", encoding="utf-8") as fp:
                fp.write(f"# checkpoint: {ckpt}\n# command: {' '.join(cmd)}\n"); fp.flush()
                subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
            if has_error(log):
                print(f"[{name}] ERROR in {log} (traceback/exception) - not registered"); continue
            rep, _ = parse_report(log)
            if rep is None:
                print(f"[{name}] ERROR: no report block in {log} - not registered"); continue
        cer_cs = round(100 - rep["Character Accuracy"], 2)
        cer_ci = round(100 - rep["Character Accuracy (Case-insensitive)"], 2)
        wer_cs = round(100 - rep["Word Accuracy"], 2)
        print(f"[{name}] {exp_id}: chars {rep['Characters']:.0f} errors {rep['Errors']:.0f} CER-cs {cer_cs:.2f} CER-ci {cer_ci:.2f} "
              f"WER-cs {wer_cs:.2f} (best epoch {epoch}, val {val}, {os.path.basename(done[0]) if done else 'unfinished'})")
        if not done:
            print(f"[{name}] unfinished run - not registered"); continue
        if r["fields"] is None:
            print(f"[{name}] pool {r['pool']} has no code/framework/output_{r['pool']}/dataset_info.json (fonts count unknown) - not registered"); continue
        note = r["note"].format(s=r["seed"], v=val, d=name, n=r["n"])
        cmd = ["python", "code/scripts/register_result.py", "--exp-id", exp_id, "--exp-name", r["exp_name"]] + r["fields"] + [
               "--cer-cs", f"{cer_cs:.2f}", "--cer-ci", f"{cer_ci:.2f}", "--wer-cs", f"{wer_cs:.2f}",
               "--best-stage", str(epoch), "--best-val-acc", str(val), "--gpu-used", gpu, "--notes", note]
        if a.dry_run:
            print("  would run:", " ".join(cmd[2:])); continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("  " + ((res.stdout.strip() or res.stderr.strip()).splitlines() or ["(no output)"])[-1])
        if res.returncode == 0:
            registered.append(exp_id)
    if registered:
        print("registered:", ", ".join(registered), "- eval_pipeline set on", set_eval_pipeline(set(registered)), "rows")
    # summaries
    rows = csv_rows()
    def sel(eid, lang, var=""): return [r for r in rows if re.match(rf"^{eid}{var}(s\d+)?_{lang}$", r["exp_id"])]
    def fmt(v): return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if v else "-")
    def lst(v): return "[" + ", ".join(f"{x:.2f}" for x in v) + "]" if v else ""
    for lang, e in ENV.items():
        for var, base in (("", f"base {e['base']}"), ("v2", f"generic base {e['alt_base']}")):
            if var and not any(sel(eid, lang, var) for eid in A_IDS.values()):
                continue
            print(f"\n=== Block A {lang} ({e['real_desc']}), {base} (CER-cs / CER-ci per seed; mean +/- sd; delta = mean - real-only mean) ===")
            ro = [float(r["cer_cs"]) for r in sel("E03", lang, var)]
            for prot, eid in A_IDS.items():
                s = sel(eid, lang, var)
                if not s:
                    print(f"{prot:14s}: -"); continue
                cs, ci = [float(r["cer_cs"]) for r in s], [float(r["cer_ci"]) for r in s]
                d = f"  delta vs real-only {statistics.mean(cs) - statistics.mean(ro):+.2f}" if ro and prot != "real-only" else ""
                print(f"{prot:14s}: n={len(s)}  CER-cs {fmt(cs):15s} {lst(cs):20s}  CER-ci {fmt(ci):15s} {lst(ci)}{d}")
    for lang in ("fr", "en", "ca"):
        print(f"\n=== Block C dose-response, {lang} (CER-cs per seed; mean +/- sd; delta = curriculum - real-only, CER-cs / CER-ci) ===")
        for size in ("500", "1k", "2k", "4k"):
            ro, cu = sel(C_IDS[size]["real-only"], lang), sel(C_IDS[size]["curriculum"], lang)
            if not ro and not cu:
                continue
            rcs, ccs = [float(r["cer_cs"]) for r in ro], [float(r["cer_cs"]) for r in cu]
            rci, cci = [float(r["cer_ci"]) for r in ro], [float(r["cer_ci"]) for r in cu]
            d = f"{statistics.mean(ccs) - statistics.mean(rcs):+.2f} / {statistics.mean(cci) - statistics.mean(rci):+.2f}" if ro and cu else "-"
            print(f"{size:>4}: real-only {fmt(rcs):15s} {lst(rcs):20s} curriculum {fmt(ccs):15s} {lst(ccs):20s} delta {d}")
    print("\n=== Base ablation, ca: FR base (FoNDUE-GD_v2_fr, E<nn>) vs generic base (FoNDUE-GD_v2, E<nn>v2); CER-cs per seed; "
          "mean +/- sd; delta = curriculum - real-only, CER-cs / CER-ci ===")
    for size in ("500", "1k", "2k", "full"):
        ids = C_IDS[size] if size != "full" else {"real-only": "E03", "curriculum": "E05"}
        for var, base in (("", "FR base  "), ("v2", "generic  ")):
            ro, cu = sel(ids["real-only"], "ca", var), sel(ids["curriculum"], "ca", var)
            if not ro and not cu:
                print(f"{size:>4} {base}: -"); continue
            rcs, ccs = [float(r["cer_cs"]) for r in ro], [float(r["cer_cs"]) for r in cu]
            rci, cci = [float(r["cer_ci"]) for r in ro], [float(r["cer_ci"]) for r in cu]
            d = f"{statistics.mean(ccs) - statistics.mean(rcs):+.2f} / {statistics.mean(cci) - statistics.mean(rci):+.2f}" if ro and cu else "-"
            print(f"{size:>4} {base}: real-only {fmt(rcs):15s} {lst(rcs):20s} curriculum {fmt(ccs):15s} {lst(ccs):20s} delta {d}")
    for lang, variants in VARIANTS.items():
        cells = [("", f"reference ({ENV[lang]['cat']} x Wikipedia)")] + [(v["v"], f"{tag} ({v['v']}: {v['desc']})") for tag, v in variants.items()]
        if not any(sel(eid, lang, var) for var, _ in cells[1:] for eid in ("E02", "E05", "E12")):
            continue
        print(f"\n=== Synthetic-content ablation, {lang} (CER-cs mean +/- sd [seeds]; curriculum full = real {ENV[lang]['n']}; "
              f"curriculum 500 = Block C 500; reference = default pool) ===")
        for var, label in cells:
            parts = []
            for eid, what in (("E02", "synth-only"), ("E05", "curriculum full"), ("E12", "curriculum 500")):
                s = [float(r["cer_cs"]) for r in sel(eid, lang, var)]
                parts.append(f"{what} {fmt(s):15s} {lst(s):20s}")
            print(f"{label}\n    " + " | ".join(parts))
    for tag, v in VARIANTS["ca"].items():
        vm = v["v"]
        rows_c = [(sz, sel(C_IDS[sz]["real-only"], "ca"), sel(C_IDS[sz]["curriculum"], "ca", vm)) for sz in ("500", "1k", "2k", "4k")]
        rows_c = [(sz, ro, cu) for sz, ro, cu in rows_c if cu] + [("full", sel("E03", "ca"), sel("E05", "ca", vm))]
        if len(rows_c) == 1 and not rows_c[0][2]:
            continue
        print(f"\n=== Block C dose-response, ca, curriculum {tag} ({vm}) vs real-only (CER-cs mean +/- sd [seeds]; "
              "delta = curriculum - real-only; SE = sqrt(sd_c^2/n_c + sd_r^2/n_r); t = delta / SE) ===")
        for sz, ro, cu in rows_c:
            rcs, ccs = [float(r["cer_cs"]) for r in ro], [float(r["cer_cs"]) for r in cu]
            if len(rcs) > 1 and len(ccs) > 1:
                d = statistics.mean(ccs) - statistics.mean(rcs)
                se = (statistics.variance(ccs) / len(ccs) + statistics.variance(rcs) / len(rcs)) ** 0.5
                st = f"delta {d:+.2f}  SE {se:.2f}  t {d / se:+.1f}  (n {len(ccs)} vs {len(rcs)})"
            else:
                st = "-"
            print(f"{sz:>4}: real-only {fmt(rcs):15s} {lst(rcs):44s} curriculum {fmt(ccs):15s} {lst(ccs):44s} {st}")
    print("\n=== Augmentation control, ca: ketos --augment on the real lines (v14) vs plain real-only; kidstext-pert curriculum 500 with (v13) / without (v12) --augment in phase 2; CER-cs mean +/- sd [seeds] ===")
    for size, eid in (("full", "E03"), ("500", "E11")):
        ro = [float(r["cer_cs"]) for r in sel(eid, "ca")]; au = [float(r["cer_cs"]) for r in sel(eid, "ca", "v14")]
        print(f"{size:>4} real-only     {fmt(ro):15s} {lst(ro):44s} +augment    {fmt(au):15s} {lst(au)}")
    kp = [float(r["cer_cs"]) for r in sel("E12", "ca", "v12")]; ka = [float(r["cer_cs"]) for r in sel("E12", "ca", "v13")]
    print(f" 500 kidstext-pert {fmt(kp):15s} {lst(kp):44s} +augment p2 {fmt(ka):15s} {lst(ka)}")
    k1 = [float(r["cer_cs"]) for r in sel("E12", "ca", "v17")]; k1f = [float(r["cer_cs"]) for r in sel("E05", "ca", "v17")]
    print(f" 500 kidstext-pert +augment p1 (synthetic only) {fmt(k1):15s} {lst(k1):44s} full {fmt(k1f):15s} {lst(k1f)}")
    print("\n=== From-scratch arm, ca: no base model: r1 = E<nn>v9 (Kraken default lr, phase1 kidstext 1:1), r2 = E<nn>v10 (lr 1e-3 warmup+plateau, B16, augment, phase1 full ca_white_30k) "
          "vs FoNDUE-GD_v2_fr fine-tuning (real-only E<nn>; curriculum = kidstext cell); "
          "CER-cs mean +/- sd [seeds] ===")
    kv = VARIANTS["ca"]["kidstext"]["v"]
    for size, ids in (("full", {"real-only": "E03", "curriculum": "E05"}), ("500", C_IDS["500"])):
        for prot, eid in ids.items():
            scr = [float(r["cer_cs"]) for r in sel(eid, "ca", "v9")]
            scr2 = [float(r["cer_cs"]) for r in sel(eid, "ca", "v10")]
            fon = [float(r["cer_cs"]) for r in sel(eid, "ca", kv if prot == "curriculum" else "")]
            print(f"{size:>4} {prot:10s}: scratch-r1 {fmt(scr):15s} {lst(scr):22s} scratch-r2 {fmt(scr2):15s} {lst(scr2):22s} FoNDUE {fmt(fon):15s} {lst(fon)}")
    print("\n=== From-scratch arm, ca, control and quality cell (full real): E03v10 real-only r2; E03v15 = E03v10 + second stage with the phase-2 recipe on the same real lines (control); "
          "E05v10 curriculum r2 (phase1 ca_white_30k); E05v16 curriculum r2 (phase1 ca_whitepert_30k, --perturbations); CER-cs mean +/- sd [seeds] ===")
    for label, eid, mk in (("real-only r2", "E03", "v10"), ("real-only r2 + 2nd stage", "E03", "v15"),
                           ("curriculum r2 white30k", "E05", "v10"), ("curriculum r2 whitepert30k", "E05", "v16"),
                           ("500: real-only r2", "E11", "v10"), ("500: real-only r2 + 2nd stage", "E11", "v15"), ("500: curriculum r2 white30k", "E12", "v10")):
        v = [float(r["cer_cs"]) for r in sel(eid, "ca", mk)]
        print(f"{label:28s}: {fmt(v):15s} {lst(v)}")


if __name__ == "__main__":
    main()
