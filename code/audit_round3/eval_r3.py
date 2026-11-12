#!/usr/bin/env python
"""Evaluate and register the audit-round-3 runs (never touches results/all_results.csv).

For every audit_round3/models/<run>/phase2_<size>/m (and phase1/m, registered as synth-only) whose phase is DONE
(train_fixed.sh writes the marker only after pruning, so runs still training on the other GPU are never read):
  1. checkpoint = phase 2: highest validation accuracy in the file name, lowest epoch on ties (eval_register.py rule);
     phase 1 (fixed 20 epochs): last.ckpt, the checkpoint that starts phase 2
  2. ketos test -f binary -u NFD on data/real/catk12hw/catk12hw_test.arrow -> <dir>/eval_full.log (reused when it already
     holds a complete report for the same checkpoint)
  3. one row in audit_round3/results/results_round3.csv (same columns as results/all_results.csv) and a copy of the log
     in audit_round3/results/eval_logs/.
Usage: python eval_r3.py [--device cuda:0]   (run under flock audit_round3/results/.eval.lock)
"""
import argparse, csv, datetime, glob, os, re, shutil, subprocess

ROOT = "/home/joc13/workrepo-htr-synthetic-framework"
PY = "/home/joc13/miniconda3/envs/kraken7/bin/python"
TEST = "data/real/catk12hw/catk12hw_test.arrow"
CSV = "audit_round3/results/results_round3.csv"
COLS = ["exp_id", "exp_name", "protocol", "language", "status", "model_base", "dataset_real", "real_train_size", "dataset_synth",
        "synth_samples_count", "fonts_count", "font_category", "quality_level", "phase1_lr", "phase2_lr", "decoder", "filter_qmark",
        "cer_cs", "cer_ci", "wer_cs", "wer_ci", "best_stage", "best_val_acc", "wall_time_hours", "gpu_used", "commit_sha",
        "timestamp", "notes", "eval_pipeline"]
CK = re.compile(r"checkpoint_(\d+)-([\d.]+)\.ckpt$")
REP = re.compile(r"^\s*(-?[\d.]+)%?\s+(Characters|Errors|Character Accuracy|Character Accuracy \(Case-insensitive\)|Word Accuracy)\s*$")
NEED = ("Characters", "Errors", "Character Accuracy", "Character Accuracy (Case-insensitive)", "Word Accuracy")
SIZES = {"500": 500, "1k": 1000, "2k": 2000, "4k": 4000, "full": 4152}


def best_ckpt(d):
    c = [(float(m.group(2)), -int(m.group(1)), f) for f in glob.glob(f"{d}/checkpoint_*.ckpt") for m in [CK.search(f)] if m]
    return sorted(c)[-1][2] if c else None


def parse(path):
    vals, ck = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("# checkpoint: "):
            ck = line[14:].strip()
        m = REP.match(line)
        if m:
            vals[m.group(2)] = float(m.group(1))
    return (vals if all(k in vals for k in NEED) else None), ck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    os.chdir(ROOT)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8"))) if os.path.exists(CSV) else []
    done = {r["exp_name"] for r in rows}
    sha = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    runs = []
    for run in sorted(glob.glob("audit_round3/models/*_seed*")):
        m = re.match(r".*/ca_curriculum_(\w+?_fixed20)_seed(\d+)$", run)
        if not m:
            continue
        variant, seed = m.group(1), int(m.group(2))
        for sub in sorted(glob.glob(f"{run}/phase2_*")) + [f"{run}/phase1"]:
            if not os.path.exists(f"{sub}/DONE"):
                continue
            size = os.path.basename(sub).split("_", 1)[1] if sub.endswith(("500", "1k", "2k", "4k")) else "full"
            if sub.endswith("phase1"):
                name, prot, nreal = f"ca_synth-only_{variant}_seed{seed}", "synth-only", ""
            else:
                name, prot, nreal = f"ca_curriculum_{size}_{variant}_seed{seed}", "curriculum", str(SIZES[size])
            runs.append((name, prot, nreal, variant, seed, f"{sub}/m", sub))
    for name, prot, nreal, variant, seed, ckdir, sub in runs:
        if name in done:
            continue
        ck = f"{ckdir}/last.ckpt" if sub.endswith("phase1") else best_ckpt(ckdir)
        ck = ck if os.path.exists(ck) else None
        if not ck:
            print(f"[{name}] no checkpoint"); continue
        log = f"{sub}/eval_full.log"
        vals, logck = parse(log) if os.path.exists(log) else (None, None)
        if vals is None or logck != ck:
            cmd = [PY, "code/scripts/ketos_safe.py", "-d", a.device, "test", "-m", ck, "-f", "binary", "-u", "NFD", TEST]
            print(f"[{name}] ketos test {os.path.basename(ck)}", flush=True)
            with open(log, "w", encoding="utf-8") as fp:
                fp.write(f"# checkpoint: {ck}\n# command: {' '.join(cmd)}\n"); fp.flush()
                subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, check=False)
            vals, _ = parse(log)
            if vals is None:
                print(f"[{name}] no valid report in {log}"); continue
        m = CK.search(ck)
        epoch, val = (int(m.group(1)), float(m.group(2))) if m else (19, "")
        cer_cs, cer_ci, wer = round(100 - vals["Character Accuracy"], 2), round(100 - vals["Character Accuracy (Case-insensitive)"], 2), round(100 - vals["Word Accuracy"], 2)
        shutil.copy(log, f"audit_round3/results/eval_logs/{name}.eval_full.log")
        pool = {"s1_fixed20": "ca_kidstextpert500_white_6k", "s2_fixed20": "ca_kidstextpert_white_6k", "s2x415_fixed20": "ca_kidstextpert415ext_white_6k", "s2xall_fixed20": "ca_kidstextpertextall_white_6k"}.get(variant, variant)
        row = dict.fromkeys(COLS, "")
        row.update(exp_id=f"R3_{name}", exp_name=name, protocol=prot, language="ca", status="completed", model_base="FoNDUE-GD_v2_fr",
                   dataset_real="catk12hw" if nreal else "", real_train_size=nreal, dataset_synth=pool, synth_samples_count=4152,
                   fonts_count=23, font_category="KidsCurated", quality_level="Degraded", phase1_lr=0.0001, phase2_lr=0.0001 if nreal else "",
                   decoder="greedy_ctc", filter_qmark="no", cer_cs=cer_cs, cer_ci=cer_ci, wer_cs=wer, best_stage=epoch, best_val_acc=val,
                   gpu_used=a.device, commit_sha=sha, timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   notes=f"Audit round 3 (B1'/B9): phase 1 of fixed length (20 epochs, last.ckpt) on {pool}; "
                         + ("phase 1 only, evaluated as synth-only" if not nreal else f"phase 2 on {nreal} real lines from the phase-1 last checkpoint") + f", seed {seed}. ketos test -u NFD.",
                   eval_pipeline="ketos_test")
        rows.append(row); done.add(name)
        print(f"[{name}] CER-cs {cer_cs} CER-ci {cer_ci} WER-cs {wer} (epoch {epoch}, val {val})")
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows in {CSV}")


if __name__ == "__main__":
    main()
