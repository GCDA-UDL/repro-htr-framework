#!/usr/bin/env python3
"""Re-run of every English (IAM) Kraken experiment of the article with the standard Aachen split (Teklia/IAM-line).

Same commands and hyperparameters as code/training/train_blockA.sh and train_blockC.sh of the work repository
(ketos train -f binary --resize add -r 1e-4 --lag 5 --min-epochs 3 -u NFD; curriculum-LR phase 2 at 1e-5; Block C
curriculum = phase 2 only from the Block A phase-1 checkpoint of the same seed), and the same evaluation
(ketos test -f binary -u NFD on the best checkpoint, full report in eval_full.log). Differences: real data = Aachen
split (6,482 / 976 / 2,915 lines), synthetic 1:1 subsample = 6,482 lines (subsample_arrow.py, seed 42), and every run
has an explicit seed (1, 2, 3).

Scheduler: one job per GPU, dependencies respected, resumable (a job with DONE is skipped; a job that failed is not
retried). Control files in logs/: STOP (start no new job), GPUS (comma-separated GPU indices to use, default 0,1).
Usage: setsid nohup python -I scripts/iam_queue.py > logs/queue.out 2>&1 &          (baseline, real-only, campaign c2)
       setsid nohup python -I scripts/iam_queue.py c1only > logs/queue_c1.out 2>&1 &  (campaign c1; second job per GPU)
"""
import json, os, subprocess, sys, time, datetime

ROOT = "/home/vitor.luiz/joaquim/rerun-iam"
PY = "/home/joc13/miniconda3/envs/kraken7/bin/python"
KS = f"{ROOT}/scripts/ketos_safe.py"
BASE = "/home/joc13/workrepo-htr-synthetic-framework/models/base/FoNDUE-GD_v2.mlmodel"
SYN = "/home/joc13/workrepo-htr-synthetic-framework/data/synthetic"
D = f"{ROOT}/data"
REAL_VAL, TEST = f"{D}/iam_val.arrow", f"{D}/iam_test.arrow"
REAL = {"full": f"{D}/iam_train.arrow", **{s: f"{D}/iam_{s}_train.arrow" for s in ("500", "1k", "2k", "4k")}}
CAMPAIGN = {  # synthetic train (1:1 subsample) and synthetic validation of each campaign
    "c2": (f"{D}/synthetic/en_white_6482_train_bl.arrow", f"{SYN}/en_white_30k/en_white_30k_validation_bl.arrow"),
    "c1": (f"{D}/synthetic/en_synth_6482_train_bl.arrow", f"{SYN}/en_pool_30k/en_synth_30k_validation_bl.arrow")}
LOGS, MODELS = f"{ROOT}/logs", f"{ROOT}/models"

BEST = r"""best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_[0-9]*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }"""


def train(out, init, lr, val, data, seed):
    vl = f"{out}/val_list.txt"
    return (f'mkdir -p {out}\necho "{val}" > {vl}\n'
            f'{PY} {KS} --seed {seed} -d $GPU train -i "{init}" -f binary --resize add -r {lr} --lag 5 --min-epochs 3 -u NFD '
            f'-o {out}/m -e {vl} {" ".join(data)} > {out}/train.log 2>&1\n')


def evaluate(out, model_expr):
    return (f'mkdir -p {out}\nM={model_expr}; [ -n "$M" ] || {{ echo "no checkpoint in {out}"; exit 1; }}\n'
            f'echo "model=$M" >> {out}/run_info.txt\n'
            f'{PY} {KS} -d $GPU test -m "$M" -f binary -u NFD {TEST} > {out}/eval_full.log 2>&1\n'
            f'grep -q "Character Accuracy" {out}/eval_full.log\n')


def jobs(campaigns, shared=True):
    J = []  # (id, deps, top folder, script)
    if shared:
        J.append(("en_baseline", [], f"{MODELS}/en_baseline", evaluate(f"{MODELS}/en_baseline", BASE)))
    for c in campaigns:
        syn_tr, syn_val = CAMPAIGN[c]
        for s in (1, 2, 3):
            o = f"{MODELS}/{c}_en_curriculum_seed{s}/phase1"
            J.append((f"{c}_en_p1_seed{s}", [], o, train(o, BASE, "0.0001", syn_val, [syn_tr], s)))
        o = f"{MODELS}/{c}_en_curriculum-LR_seed1"
        J.append((f"{c}_en_curriculum-LR_seed1", [], o,
                  train(f"{o}/phase1", BASE, "0.0001", syn_val, [syn_tr], 1)
                  + f'B1=$(best {o}/phase1/m); [ -n "$B1" ] || exit 1; echo "phase1=$B1" >> {o}/run_info.txt\n'
                  + train(f"{o}/phase2", "$B1", "0.00001", REAL_VAL, [REAL["full"]], 1) + evaluate(o, f"$(best {o}/phase2/m)")))
        o = f"{MODELS}/{c}_en_synth-only_seed1"
        J.append((f"{c}_en_synth-only_seed1", [], o, train(o, BASE, "0.0001", syn_val, [syn_tr], 1) + evaluate(o, f"$(best {o}/m)")))
        for s in (1, 2, 3):
            o = f"{MODELS}/{c}_en_mix_seed{s}"
            J.append((f"{c}_en_mix_seed{s}", [], o, train(o, BASE, "0.0001", REAL_VAL, [REAL["full"], syn_tr], s) + evaluate(o, f"$(best {o}/m)")))
        for s in (1, 2, 3):
            p1 = f"{MODELS}/{c}_en_curriculum_seed{s}/phase1/m"
            for size in ("full", "4k", "2k", "1k", "500"):
                o = f"{MODELS}/{c}_en_curriculum_seed{s}/phase2" if size == "full" else f"{MODELS}/{c}_en_curriculum_{size}_seed{s}"
                J.append((f"{c}_en_curriculum_{size}_seed{s}", [f"{c}_en_p1_seed{s}"], o,
                          f'B1=$(best {p1}); [ -n "$B1" ] || exit 1; echo "phase1=$B1" >> {o}/run_info.txt\n'
                          + train(o, "$B1", "0.0001", REAL_VAL, [REAL[size]], s) + evaluate(o, f"$(best {o}/m)")))
    for s in (1, 2, 3) if shared else ():
        for size in ("full", "4k", "2k", "1k", "500"):
            o = f"{MODELS}/en_real-only_{size}_seed{s}"
            J.append((f"en_real-only_{size}_seed{s}", [], o, train(o, BASE, "0.0001", REAL_VAL, [REAL[size]], s) + evaluate(o, f"$(best {o}/m)")))
    return J


def main():
    only_c1 = "c1only" in sys.argv[1:]   # second instance: campaign-1 jobs only, one more job per GPU
    campaigns = ["c1"] if only_c1 else ["c2"]
    J = jobs(campaigns, shared=not only_c1)
    status = f"{LOGS}/status{'_c1' if only_c1 else ''}.json"
    os.makedirs(LOGS, exist_ok=True)
    state = {j[0]: "pending" for j in J}
    for jid, *_ in J:
        if os.path.exists(f"{LOGS}/{jid}.DONE"):
            state[jid] = "done"
        elif os.path.exists(f"{LOGS}/{jid}.FAILED"):
            state[jid] = "failed"
    running = {}  # gpu -> (jid, Popen, start)
    log = lambda m: print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {m}", flush=True)
    log(f"campaigns={campaigns} jobs={len(J)} done={sum(v == 'done' for v in state.values())}")
    while True:
        for g, (jid, p, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                state[jid] = "done" if rc == 0 else "failed"
                open(f"{LOGS}/{jid}.{'DONE' if rc == 0 else 'FAILED'}", "w").write(f"rc={rc} hours={(time.time() - t0) / 3600:.2f}\n")
                log(f"{state[jid].upper()} {jid} gpu={g} rc={rc} {(time.time() - t0) / 3600:.2f} h")
                del running[g]
        try:
            gpus = [int(x) for x in open(f"{LOGS}/GPUS").read().strip().split(",") if x.strip()]
        except (FileNotFoundError, ValueError):
            gpus = [0, 1]
        if not os.path.exists(f"{LOGS}/STOP"):
            for g in gpus:
                if g in running:
                    continue
                nxt = next((j for j in J if state[j[0]] == "pending" and all(state[d] == "done" for d in j[1])), None)
                if not nxt:
                    break
                jid, _, top, script = nxt
                head = (f"set -eo pipefail\nexport GPU=cuda:{g}\n{BEST}\ncd {ROOT}\nmkdir -p {top}\n"
                        f'echo "date=$(date -Iseconds) job={jid} gpu=$GPU" >> {top}/run_info.txt\n')
                open(f"{LOGS}/{jid}.sh", "w").write(head + script)
                p = subprocess.Popen(["bash", f"{LOGS}/{jid}.sh"], stdout=open(f"{LOGS}/{jid}.log", "w"), stderr=subprocess.STDOUT, cwd=ROOT)
                running[g] = (jid, p, time.time()); state[jid] = "running"
                log(f"START {jid} gpu={g}")
        json.dump(dict(updated=datetime.datetime.now().isoformat(timespec="seconds"), state=state,
                       running={g: v[0] for g, v in running.items()}), open(status, "w"), indent=1)
        runnable = any(state[j[0]] == "pending" and all(state[d] == "done" for d in j[1]) for j in J)
        if not running and (os.path.exists(f"{LOGS}/STOP") or not runnable):
            log("stopped (STOP file)" if os.path.exists(f"{LOGS}/STOP") else "queue finished")
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
