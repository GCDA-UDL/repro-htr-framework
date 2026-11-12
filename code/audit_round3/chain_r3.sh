#!/bin/bash
# Audit round 3: run <arm>:<seed> jobs one after the other on one GPU. After each job: evaluate and register every finished
# phase (eval_r3.py, under a lock shared by both chains) and record the per-line test predictions of the two checkpoints
# of the run (phase 1 last.ckpt = synth-only fixed 20; phase 2 best = curriculum 500). B9 jobs wait until their pool exists.
# Usage: bash chain_r3.sh <cuda:N> <arm>:<seed> [<arm>:<seed> ...]
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
GPU=$1; shift
cd "$ROOT" || exit 1
mkdir -p audit_round3/logs audit_round3/results/eval_logs audit_round3/results/predictions audit_round3/results/predict_lists
exec > >(tee -a "audit_round3/logs/chain_${GPU//:/}.log") 2>&1
echo "$(date -Iseconds) chain on $GPU: $*"
for J in "$@"; do
  ARM=${J%%:*}; S=${J##*:}
  case $ARM in
    s2x415) POOL=ca_kidstextpert415ext_white_6k ;;
    s2xall) POOL=ca_kidstextpertextall_white_6k ;;
    *) POOL= ;;
  esac
  if [ -n "$POOL" ]; then
    D=audit_round3/data/synthetic
    until [ -f "$D/$POOL/GENERATION.md" ] || [ -f "$D/$POOL.FAILED" ]; do sleep 60; done
    if [ -f "$D/$POOL.FAILED" ]; then echo "$(date -Iseconds) === $J SKIPPED: pool $POOL failed ==="; continue; fi
  fi
  echo "$(date -Iseconds) === $J START ==="
  if bash audit_round3/scripts/train_fixed.sh "$GPU" "$ARM" "$S" > "audit_round3/logs/job_${ARM}_seed${S}.log" 2>&1; then
    echo "$(date -Iseconds) === $J TRAINED ==="
  else
    echo "$(date -Iseconds) === $J FAILED (see audit_round3/logs/job_${ARM}_seed${S}.log) ==="; tail -n 5 "audit_round3/logs/job_${ARM}_seed${S}.log"
  fi
  flock audit_round3/results/.eval.lock $PY audit_round3/scripts/eval_r3.py --device "$GPU"
  RUN=audit_round3/models/ca_curriculum_${ARM}_fixed20_seed${S}
  L=audit_round3/results/predict_lists/${ARM}_seed${S}.tsv
  : > "$L"
  [ -f "$RUN/phase1/DONE" ]     && printf 'ca_synth-only_%s_fixed20_seed%s\t%s\n' "$ARM" "$S" "$RUN/phase1/m/last.ckpt" >> "$L"
  [ -f "$RUN/phase2_500/DONE" ] && printf 'ca_curriculum_500_%s_fixed20_seed%s\t%s\n' "$ARM" "$S" "$RUN/phase2_500/m" >> "$L"
  [ -s "$L" ] && $PY audit_round3/scripts/predict_lines_r3.py --device "$GPU" --list "$L" 2>&1 | grep -E '^\[|FAILED|Error' 
done
echo "$(date -Iseconds) chain on $GPU finished"
