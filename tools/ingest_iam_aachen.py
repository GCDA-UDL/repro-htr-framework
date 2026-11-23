#!/usr/bin/env python
"""Replace the English (IAM) rows of the captured results with the re-run on the standard Aachen split.

The original IAM experiments used a non-standard 8,917 / 2,732 / 1,704 split of the 13,353 IAM lines. Every English
run was repeated with the standard Aachen split (Teklia/IAM-line: 6,482 / 976 / 2,915), the same recipe and three
explicit seeds (code/rerun_iam/iam_queue.py). This script reads the run folders of that re-run and rewrites, in place:
  results/kraken/all_results.csv          English rows of campaign c2 (matched background, en_white_30k)
  results/kraken/all_results_mixedbg.csv  English rows of campaign c1 (heterogeneous background, en_pool_30k)
  results/kraken/ketos_test_summaries.csv English rows (full ketos test counts), run = models/iam_aachen/<job>/...
The experiment ids (E01-E06, E11-E18, suffixes s2/s3) are unchanged, so holm_family.py and tables.py need no change.
Usage: python tools/ingest_iam_aachen.py <rerun dir>      (the folder that holds models/ and logs/)
"""
import csv, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "kraken")
RERUN = sys.argv[1]
M, L = os.path.join(RERUN, "models"), os.path.join(RERUN, "logs")
NREAL = {"full": 6482, "500": 500, "1k": 1000, "2k": 2000, "4k": 4000}
EID = {"500": ("11", "12"), "1k": ("13", "14"), "2k": ("15", "16"), "4k": ("17", "18")}
SYN = {"c2": ("en_white_30k", "187", "Handwritten", "Clean"), "c1": ("en_pool_30k", "", "", "")}
NSYN = 6482


def report(path):
    s = open(path, encoding="utf-8").read()
    g = lambda pat: re.search(pat, s).group(1)
    r = dict(characters=int(g(r"(\d+)\tCharacters")), errors=int(g(r"(\d+)\tErrors")),
             char_accuracy=float(g(r"([\d.]+)%\tCharacter Accuracy\n")), char_accuracy_ci=float(g(r"([\d.]+)%\tCharacter Accuracy \(Case-insensitive\)")),
             word_accuracy=float(g(r"([\d.]+)%\tWord Accuracy")), ketos_insertions=int(g(r"(\d+)\tInsertions")), ketos_deletions=int(g(r"(\d+)\tDeletions")))
    m = re.search(r"(\d+)\tSubstitutions", s)
    r["substitutions"] = int(m.group(1)) if m else r["errors"] - r["ketos_insertions"] - r["ketos_deletions"]
    return r


def run_info(d):
    info = dict(re.findall(r"^(\w+)=(.*)$", open(os.path.join(d, "run_info.txt"), encoding="utf-8").read(), re.M))
    m = re.search(r"checkpoint_(\d+)-([\d.]+)\.ckpt", info.get("model", ""))
    info["best_stage"], info["best_val_acc"] = (m.group(1), m.group(2)) if m else ("", "")
    info["timestamp"] = re.sub(r"T(\d\d:\d\d):\d\d.*$", r" \1", info["date"])
    info["gpu"] = info.get("gpu", "")
    return info


def hours(job):
    p = os.path.join(L, job + ".DONE")
    m = re.search(r"hours=([\d.]+)", open(p).read()) if os.path.exists(p) else None
    return m.group(1) if m else ""


FIELDS = ["exp_id", "exp_name", "protocol", "language", "status", "model_base", "dataset_real", "real_train_size", "dataset_synth",
          "synth_samples_count", "fonts_count", "font_category", "quality_level", "phase1_lr", "phase2_lr", "decoder", "filter_qmark",
          "cer_cs", "cer_ci", "wer_cs", "wer_ci", "best_stage", "best_val_acc", "wall_time_hours", "gpu_used", "commit_sha", "timestamp",
          "notes", "eval_pipeline"]
summ = {}


def row(eid, job, protocol, camp, nreal, folder, evalsub="", p1lr="", p2lr="", note=""):
    d = os.path.join(M, folder)
    ev = os.path.join(d, evalsub, "eval_full.log") if evalsub else os.path.join(d, "eval_full.log")
    k = report(ev); info = run_info(os.path.join(d, evalsub) if evalsub else d)
    key = f"models/iam_aachen/{folder}/{evalsub + '/' if evalsub else ''}eval_full.log"
    summ[key] = dict(run=key, **{a: k[a] for a in ("characters", "errors", "ketos_insertions", "ketos_deletions", "substitutions")},
                     char_accuracy=f"{k['char_accuracy']:.2f}", char_accuracy_ci=f"{k['char_accuracy_ci']:.2f}", word_accuracy=f"{k['word_accuracy']:.2f}")
    syn = SYN[camp] if camp else ("", "", "", "")
    val = f" best ckpt (val {info['best_val_acc']})" if info["best_val_acc"] else ""
    return dict(exp_id=eid, exp_name=job, protocol=protocol, language="en", status="completed", model_base="FoNDUE-GD_v2",
                dataset_real="iam" if nreal else "", real_train_size=str(nreal) if nreal else "", dataset_synth=syn[0],
                synth_samples_count=str(NSYN) if syn[0] else "", fonts_count=syn[1], font_category=syn[2], quality_level=syn[3],
                phase1_lr=p1lr, phase2_lr=p2lr, decoder="greedy_ctc", filter_qmark="no",
                cer_cs=f"{100 - k['char_accuracy']:.2f}", cer_ci=f"{100 - k['char_accuracy_ci']:.2f}", wer_cs=f"{100 - k['word_accuracy']:.2f}", wer_ci="",
                best_stage=info["best_stage"], best_val_acc=info["best_val_acc"], wall_time_hours=hours(job), gpu_used=info["gpu"], commit_sha="",
                timestamp=info["timestamp"], notes=f"IAM Aachen split (Teklia/IAM-line, 6,482 / 976 / 2,915). {note} ketos test -u NFD{val}. {job}",
                eval_pipeline="ketos_test")


shared = [row("E01_en", "en_baseline", "baseline", None, 0, "en_baseline", note="Block A baseline: FoNDUE-GD_v2 evaluated on the test set, no fine-tuning.")]
for size in ("full", "500", "1k", "2k", "4k"):
    for s in (1, 2, 3):
        eid = "E" + ("03" if size == "full" else EID[size][0]) + ("" if s == 1 else f"s{s}") + "_en"
        shared.append(row(eid, f"en_real-only_{size}_seed{s}", "real-only", None, NREAL[size], f"en_real-only_{size}_seed{s}", p2lr="0.0001",
                          note=f"Block {'A' if size == 'full' else 'C'} real-only, real={NREAL[size]:,}, no synthetic, seed {s}."))
camp_rows = {}
for camp in ("c2", "c1"):
    R = []
    R.append(row("E02_en", f"{camp}_en_synth-only_seed1", "synth-only", camp, 0, f"{camp}_en_synth-only_seed1", p2lr="0.0001",
                 note=f"Block A synth-only: {SYN[camp][0]} 1:1 subsample ({NSYN:,}), no real, seed 1, early stopping on the synthetic validation."))
    for s in (1, 2, 3):
        sfx = "" if s == 1 else f"s{s}"
        R.append(row(f"E04{sfx}_en", f"{camp}_en_mix_seed{s}", "mix", camp, NREAL["full"], f"{camp}_en_mix_seed{s}", p2lr="0.0001",
                     note=f"Block A mix: IAM {NREAL['full']:,} + {SYN[camp][0]} {NSYN:,}, single phase, seed {s}."))
        R.append(row(f"E05{sfx}_en", f"{camp}_en_curriculum_full_seed{s}", "curriculum", camp, NREAL["full"], f"{camp}_en_curriculum_seed{s}", "phase2",
                     p1lr="0.0001", p2lr="0.0001", note=f"Block A curriculum: phase 1 synthetic {NSYN:,} -> phase 2 real {NREAL['full']:,}, both lr 1e-4, seed {s}."))
        for size in ("500", "1k", "2k", "4k"):
            R.append(row(f"E{EID[size][1]}{sfx}_en", f"{camp}_en_curriculum_{size}_seed{s}", "curriculum", camp, NREAL[size], f"{camp}_en_curriculum_{size}_seed{s}",
                         p1lr="0.0001", p2lr="0.0001", note=f"Block C curriculum arm, real={NREAL[size]:,}, phase 1 reused from the block A curriculum of seed {s}, lr 1e-4, seed {s}."))
    R.append(row("E06_en", f"{camp}_en_curriculum-LR_seed1", "curriculum-LR", camp, NREAL["full"], f"{camp}_en_curriculum-LR_seed1", p1lr="0.0001", p2lr="1e-05",
                 note=f"Block A curriculum-LR: phase 1 synthetic {NSYN:,} -> phase 2 real {NREAL['full']:,} lr 1e-5, seed 1."))
    camp_rows[camp] = R

for fname, camp in (("all_results.csv", "c2"), ("all_results_mixedbg.csv", "c1")):
    p = os.path.join(RES, fname)
    with open(p, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh); fields = rd.fieldnames; old = list(rd)
    assert fields == FIELDS, fields
    keep = [r for r in old if r["language"] != "en"]
    new = shared + camp_rows[camp]
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(keep + new)
    print(f"{fname}: {len(old) - len(keep)} English rows removed, {len(new)} written ({len(keep) + len(new)} rows)")

p = os.path.join(RES, "ketos_test_summaries.csv")
with open(p, encoding="utf-8", newline="") as fh:
    rd = csv.DictReader(fh); fields = rd.fieldnames; old = list(rd)
keep = [r for r in old if "/en_" not in r["run"]]
with open(p, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(keep + list(summ.values()))
print(f"ketos_test_summaries.csv: {len(old) - len(keep)} English rows removed, {len(summ)} written")
