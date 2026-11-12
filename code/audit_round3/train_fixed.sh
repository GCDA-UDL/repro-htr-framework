#!/bin/bash
# Audit round 3, controls B1' and B9: curriculum with a phase 1 of FIXED length, identical for every arm.
#   phase 1: FoNDUE-GD_v2_fr fine-tuned for exactly 20 epochs (-q fixed -N 20) on the 4152-line 1:1 subsample of the
#            arm's pool, lr 1e-4, --resize add, -u NFD (the synthetic validation split is only logged, never used to
#            select); the checkpoint of the last epoch (last.ckpt, kept by ketos_safe_last.py) starts phase 2.
#   phase 2: on the 500 real lines (real validation), lr 1e-4, lag 5, min-epochs 3, as train_blockC.sh.
# Arms: s1 = texts of the 500 labelled lines (audit_round2 pool); s2 = all training transcriptions (kidstext-pert pool of
# the article); s2x415 = 415 other training texts with the S1 frequency profile; s2xall = every training line whose text
# is not in the 500 subset. Usage: bash train_fixed.sh <cuda:N> <arm> <seed>
set -eo pipefail
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
KS=$ROOT/audit_round3/scripts/ketos_safe_last.py
GPU=$1; ARM=$2; SEED=$3
[ -n "$GPU" ] && [ -n "$ARM" ] && [ -n "$SEED" ] || { echo "usage: train_fixed.sh <cuda:N> <arm> <seed>"; exit 1; }
cd "$ROOT"
case $ARM in
  s1)     P=audit_round2/data/synthetic/ca_kidstextpert500_white_6k;     TR=$P/ca_kidstextpert500_white_4152_train_bl.arrow;     VA=$P/ca_kidstextpert500_white_6k_validation_bl.arrow ;;
  s2)     P=data/synthetic/ca_kidstextpert_white_6k;                    TR=$P/ca_kidstextpert_white_4152_train_bl.arrow;        VA=$P/ca_kidstextpert_white_6k_validation_bl.arrow ;;
  s2x415) P=audit_round3/data/synthetic/ca_kidstextpert415ext_white_6k; TR=$P/ca_kidstextpert415ext_white_4152_train_bl.arrow; VA=$P/ca_kidstextpert415ext_white_6k_validation_bl.arrow ;;
  s2xall) P=audit_round3/data/synthetic/ca_kidstextpertextall_white_6k; TR=$P/ca_kidstextpertextall_white_4152_train_bl.arrow; VA=$P/ca_kidstextpertextall_white_6k_validation_bl.arrow ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
BASE=models/base/FoNDUE-GD_v2_fr.mlmodel
REAL_TR=data/real/catk12hw/catk12hw_500_train.arrow
REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
for f in $BASE $TR $VA $REAL_TR $REAL_VAL; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
OUT=audit_round3/models/ca_curriculum_${ARM}_fixed20_seed${SEED}
MAN=audit_round3/data/manifests; mkdir -p "$OUT" "$MAN"
echo "$VA" > "$MAN/syn_val_${ARM}.txt"; echo "$REAL_VAL" > "$MAN/real_val.txt"
best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\)-\([0-9.]*\)\.ckpt/\2 \1 &/' | sort -k1,1nr -k2,2n | head -1 | cut -d' ' -f3; }
if [ ! -f "$OUT/phase1/DONE" ]; then
  rm -rf "$OUT/phase1"; mkdir -p "$OUT/phase1"
  echo "date=$(date -Iseconds) lang=ca protocol=curriculum-fixed20 arm=$ARM phase=1 base=$BASE syn=$TR epochs=20 (fixed) seed=$SEED gpu=$GPU" | tee -a "$OUT/run_info.txt"
  $PY $KS --seed $SEED -d $GPU train -i $BASE -f binary --resize add -r 0.0001 -q fixed -N 20 -u NFD \
      -o "$OUT/phase1/m" -e "$MAN/syn_val_${ARM}.txt" $TR 2>&1 | tee "$OUT/phase1.log"
  [ -f "$OUT/phase1/m/last.ckpt" ] || { echo "PHASE1 FAIL: no last.ckpt"; exit 1; }
  EP=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$OUT/phase1/m/last.ckpt")
  echo "phase1 last.ckpt epoch=$EP" | tee -a "$OUT/run_info.txt"
  [ "$EP" -ge 19 ] 2>/dev/null || { echo "PHASE1 FAIL: last.ckpt is epoch $EP, expected 19"; exit 1; }
  find "$OUT/phase1/m" -name 'checkpoint_*.ckpt' -delete      # keep last.ckpt (epoch 19) and best_*.safetensors
  touch "$OUT/phase1/DONE"
fi
P2=$OUT/phase2_500
if [ ! -f "$P2/DONE" ]; then
  rm -rf "$P2"; mkdir -p "$P2"
  echo "date=$(date -Iseconds) lang=ca protocol=curriculum-fixed20 arm=$ARM phase=2 size=500 phase1=$OUT/phase1/m/last.ckpt lr=0.0001 lag=5 min_epochs=3 seed=$SEED gpu=$GPU" | tee -a "$OUT/run_info.txt"
  $PY $KS --seed $SEED -d $GPU train -i "$OUT/phase1/m/last.ckpt" -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$P2/m" -e "$MAN/real_val.txt" $REAL_TR 2>&1 | tee "$P2/train.log"
  ls "$P2"/m/best_* >/dev/null 2>&1 || { echo "PHASE2 FAIL"; exit 1; }
  B=$(best "$P2/m"); echo "selected=$B" >> "$OUT/run_info.txt"
  find "$P2/m" -name 'checkpoint_*.ckpt' ! -path "$B" -delete; rm -f "$P2/m/last.ckpt"
  touch "$P2/DONE"
fi
echo "=== DONE $ARM seed $SEED ($(date -Iseconds)) ==="
