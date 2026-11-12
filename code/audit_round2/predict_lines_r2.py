#!/usr/bin/env python
"""Line-by-line test predictions of Kraken checkpoints with exactly the criterion of `ketos test -f binary -u NFD`
(audit round 2). Same mechanism as code/scripts/predict_lines.py: the real ketos test command is run through
kraken.ketos:cli and every (target, prediction, edit cost) that VGSLRecognitionModel.test_step passes to global_align
is recorded. One subprocess per checkpoint. The global CER recomputed from the costs must equal the registered CER.

Usage: python predict_lines_r2.py --device cuda:1 --list runs.tsv
  runs.tsv: <name>\t<checkpoint dir or file>   (dir: the checkpoint named in its eval_full.log header, else highest val acc)
Output: audit_round2/results/predictions/<name>.tsv (target, prediction, cost, nchars) + <name>.json (checkpoint, CER).
"""
import argparse, csv, glob, inspect, json, os, re, subprocess, sys

ROOT = "/home/joc13/workrepo-htr-synthetic-framework"
OUT = f"{ROOT}/audit_round2/results/predictions"
TEST = f"{ROOT}/data/real/catk12hw/catk12hw_test.arrow"
CK = re.compile(r"checkpoint_(\d+)-([\d.]+)\.ckpt$")


def resolve(path):
    if os.path.isfile(path):
        return path
    d = path.rstrip("/")
    for _ in range(3):   # eval_full.log sits in the run folder: <run>/m or <run>/phase2/m (eval_register.py) or <run>/phase2_<size>/m (round 2)
        d = os.path.dirname(d)
        log = os.path.join(d, "eval_full.log")
        if os.path.exists(log):
            for line in open(log, encoding="utf-8", errors="replace"):
                if line.startswith("# checkpoint: "):
                    ck = line[14:].strip(); ck = ck if os.path.isabs(ck) else os.path.join(ROOT, ck)
                    if os.path.exists(ck):
                        return ck
            break
    c = [(float(m.group(2)), -int(m.group(1)), f) for f in glob.glob(f"{path}/checkpoint_*.ckpt") for m in [CK.search(f)] if m]
    return sorted(c)[-1][2] if c else None


def run_one(name, ck, device):
    import torch, kraken.configs.vgsl as kvgsl
    torch.serialization.add_safe_globals([obj for _, obj in inspect.getmembers(kvgsl, inspect.isclass)])
    import kraken.train.vgsl as V
    rec, orig = [], V.global_align

    def recorder(target, pred):
        c, a1, a2 = orig(target, pred)
        rec.append((target, pred, c))
        return c, a1, a2
    V.global_align = recorder
    from kraken.ketos import cli
    cli.main(args=["-d", device, "test", "-m", ck, "-f", "binary", "-u", "NFD", TEST], standalone_mode=False)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(["target", "prediction", "cost", "nchars"])
        for t, p, c in rec:
            w.writerow([t, p, c, len(t)])
    chars = sum(len(t) for t, _, _ in rec); err = sum(c for _, _, c in rec)
    json.dump({"name": name, "checkpoint": os.path.relpath(ck, ROOT), "lines": len(rec), "chars": chars, "errors": err,
               "cer_cs": round(100 * err / chars, 4) if chars else None, "criterion": "ketos test -f binary -u NFD, costs from global_align"},
              open(f"{OUT}/{name}.json", "w"), indent=1)
    print(f"[{name}] {len(rec)} lines, {chars} chars, CER-cs {100 * err / chars:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--list", required=True)
    ap.add_argument("--one", nargs=2, metavar=("NAME", "CKPT")); a = ap.parse_args()
    if a.one:
        run_one(a.one[0], a.one[1], a.device); return
    for line in open(a.list, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        name, path = line.rstrip("\n").split("\t")
        if os.path.exists(f"{OUT}/{name}.json"):
            continue
        ck = resolve(path if os.path.isabs(path) else os.path.join(ROOT, path))
        if not ck:
            print(f"[{name}] no checkpoint in {path}", flush=True); continue
        r = subprocess.run([sys.executable, __file__, "--device", a.device, "--list", a.list, "--one", name, ck])
        if r.returncode:
            print(f"[{name}] FAILED rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()
