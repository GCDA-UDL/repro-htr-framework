#!/bin/bash
# Audit round 2: run the B1 curriculum for several seeds on one GPU, evaluating and registering after each seed.
# Usage: bash chain_b1.sh <cuda:N> <seed> [<seed> ...]
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
GPU=$1; shift
cd "$ROOT" || exit 1
LOG=audit_round2/logs/chain_b1_${GPU//:/}.log
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Iseconds) chain on $GPU, seeds: $*"
for S in "$@"; do
  echo "$(date -Iseconds) === seed $S START ==="
  if bash audit_round2/scripts/train_b1.sh "$GPU" "$S" 500 1k 2k; then
    echo "$(date -Iseconds) === seed $S TRAINED ==="
  else
    echo "$(date -Iseconds) === seed $S FAILED (rc=$?) ==="
  fi
  flock audit_round2/results/.eval.lock $PY audit_round2/scripts/eval_r2.py --device "$GPU"
done
echo "$(date -Iseconds) chain on $GPU finished"
