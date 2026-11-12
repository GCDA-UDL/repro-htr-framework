#!/bin/bash
# Audit round 2, control B1: curriculum with the 500-text pool. Usage: bash train_b1.sh <cuda:N> <seed> [sizes...]
#   phase 1: FoNDUE-GD_v2_fr fine-tuned on the 4152-line 1:1 subsample of ca_kidstextpert500_white_6k (synthetic
#            validation split of the same pool), lr 1e-4, lag 5, min-epochs 3, --resize add, -u NFD   [as train_blockA.sh]
#   phase 2: from the best phase-1 checkpoint (highest validation accuracy in the file name), on the real subset of
#            500 / 1k / 2k lines (real validation), lr 1e-4, same options                                [as train_blockC.sh]
# Everything under audit_round2/models/ca_curriculum_kidstextpert500_seed<S>/{phase1,phase2_<size>}/m
set -eo pipefail
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
KS=$ROOT/code/scripts/ketos_safe.py
GPU=$1; SEED=$2; shift 2; SIZES=${*:-500 1k 2k}
[ -n "$GPU" ] && [ -n "$SEED" ] || { echo "usage: bash train_b1.sh <cuda:N> <seed> [sizes...]"; exit 1; }
cd "$ROOT"
BASE=models/base/FoNDUE-GD_v2_fr.mlmodel
POOL=audit_round2/data/synthetic/ca_kidstextpert500_white_6k
SYN_TR=$POOL/ca_kidstextpert500_white_4152_train_bl.arrow
SYN_VAL=$POOL/ca_kidstextpert500_white_6k_validation_bl.arrow
REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
for f in $BASE $SYN_TR $SYN_VAL $REAL_VAL; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
OUT=audit_round2/models/ca_curriculum_kidstextpert500_seed${SEED}
MAN=audit_round2/data/manifests; mkdir -p "$OUT" "$MAN"
echo "$SYN_VAL" > "$MAN/syn_val_b1.txt"; echo "$REAL_VAL" > "$MAN/real_val.txt"
best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_[0-9]*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }
COMMON="-f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD"
if [ ! -f "$OUT/phase1/DONE" ]; then
  mkdir -p "$OUT/phase1"
  echo "date=$(date -Iseconds) lang=ca protocol=curriculum-b1 phase=1 base=$BASE syn=$SYN_TR seed=$SEED gpu=$GPU" | tee -a "$OUT/run_info.txt"
  $PY $KS --seed $SEED -d $GPU train -i $BASE $COMMON -o "$OUT/phase1/m" -e "$MAN/syn_val_b1.txt" $SYN_TR 2>&1 | tee "$OUT/phase1.log"
  [ -n "$(best "$OUT/phase1/m")" ] && ls "$OUT"/phase1/m/best_* >/dev/null 2>&1 || { echo "PHASE1 FAIL"; exit 1; }
  touch "$OUT/phase1/DONE"
fi
B1=$(best "$OUT/phase1/m")
for SIZE in $SIZES; do
  REAL_TR=data/real/catk12hw/catk12hw_${SIZE}_train.arrow
  [ -f "$REAL_TR" ] || { echo "missing $REAL_TR"; exit 1; }
  P2=$OUT/phase2_${SIZE}
  if [ -f "$P2/DONE" ]; then echo "== phase2 $SIZE seed $SEED already done"; continue; fi
  mkdir -p "$P2"
  echo "date=$(date -Iseconds) lang=ca protocol=curriculum-b1 phase=2 size=$SIZE phase1=$B1 lr=0.0001 seed=$SEED gpu=$GPU" | tee -a "$OUT/run_info.txt"
  $PY $KS --seed $SEED -d $GPU train -i "$B1" $COMMON -o "$P2/m" -e "$MAN/real_val.txt" $REAL_TR 2>&1 | tee "$P2/train.log"
  ls "$P2"/m/best_* >/dev/null 2>&1 || { echo "PHASE2 $SIZE FAIL"; exit 1; }
  touch "$P2/DONE"
done
echo "=== DONE b1 seed $SEED ($(date -Iseconds)) ==="
