#!/bin/bash
# TrOCR arm (microsoft/trocr-base-handwritten), Catalan; mirrors the Kraken protocols on the same arrows.
# Usage: TROCR_RECIPE=r2 bash train_trocr.sh <GPU> <real-only|curriculum> <full|500> <seed>
#   r1 (2026-10-14, unstable): lr 5e-5 constant, batch 8, lag 5 / min 3            -> models/trocr/
#   r2 (2026-10-15): lr 2e-5 linear decay, batch 8 x accum 2, lag 10 / min 10, no_repeat_ngram 3 -> models/trocr_r2/
# curriculum = phase1 on synthetic kidstext (shared per seed), then phase2 on real; same recipe both phases.
set -eo pipefail
cd ~/workrepo-htr-synthetic-framework
PY=~/miniconda3/envs/trocr/bin/python; RUN="$PY code/trocr/trocr_run.py"
GPU=$1; PROT=$2; SIZE=$3; SEED=$4; R=${TROCR_RECIPE:-r2}
[ -n "$SEED" ] || { echo "usage: $0 GPU real-only|curriculum full|500 SEED"; exit 1; }
BASE=${TROCR_BASE:-microsoft/trocr-base-handwritten}   # TROCR_BASE=microsoft/trocr-base-stage1: printed-text pre-training only (2026-10-19)
case $R in
  r1) HP="--lr 5e-5 --batch 8 --warmup 100 --lag 5 --min-epochs 3 --max-epochs 40"; TESTOPT=""; ROOT=models/trocr ;;
  r2) HP="--lr 2e-5 --batch 8 --accum 2 --warmup 200 --schedule linear --lag 10 --min-epochs 10 --max-epochs 40 --no-repeat-ngram 3"
      TESTOPT="--no-repeat-ngram 3"; ROOT=models/trocr_r2 ;;
  *) echo "unknown recipe $R"; exit 1 ;;
esac
[ -n "$TROCR_ROOT" ] && ROOT=$TROCR_ROOT   # e.g. models/trocr_s1 for the stage-1 base
REAL_VAL=data/real/catk12hw/catk12hw_val.arrow; TEST=data/real/catk12hw/catk12hw_test.arrow
if [ "$SIZE" = full ]; then REAL_TR=data/real/catk12hw/catk12hw_train.arrow; BLK=blockA; TAG=""
else REAL_TR=data/real/catk12hw/catk12hw_${SIZE}_train.arrow; BLK=blockC; TAG="_${SIZE}"; fi
SYN=${TROCR_SYN:-kidstext}; P1MAX=""
case $SYN in
  kidstext)    SYN_TR=data/synthetic/ca_kidstext_white_6k/ca_kidstext_white_4152_train_bl.arrow; SYN_VAL=data/synthetic/ca_kidstext_white_6k/ca_kidstext_white_6k_validation_bl.arrow ;;
  kidstextpert) SYN_TR=data/synthetic/ca_kidstextpert_white_6k/ca_kidstextpert_white_4152_train_bl.arrow; SYN_VAL=data/synthetic/ca_kidstextpert_white_6k/ca_kidstextpert_white_6k_validation_bl.arrow ;;
  kidstext30k) SYN_TR=data/synthetic/ca_kidstext_white_30k/ca_kidstext_white_30k_train_bl.arrow; SYN_VAL=data/synthetic/ca_kidstext_white_30k/ca_kidstext_white_30k_validation_bl.arrow; P1MAX="--max-epochs 10" ;;
  *) echo "unknown TROCR_SYN $SYN"; exit 1 ;;
esac
for f in "$REAL_TR" "$REAL_VAL" "$TEST" "$SYN_TR" "$SYN_VAL"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
A_CUR=$ROOT/blockA/ca_curriculum_${SYN}_seed${SEED}; P1=$A_CUR/phase1
case $PROT in
  real-only) OUT=$ROOT/$BLK/ca_real-only${TAG}_seed${SEED}; START=$BASE; MDIR=$OUT ;;
  curriculum)
    if [ ! -f "$P1"/DONE ]; then
      [ "$SIZE" = full ] || { echo "phase1 not finished: $P1"; exit 1; }
      mkdir -p "$P1"
      echo "date=$(date -Iseconds) arch=trocr recipe=$R phase1 base=$BASE syn=$SYN_TR hp=[$HP] seed=$SEED gpu=$GPU" | tee -a "$A_CUR"/run_info.txt
      $RUN train --model $BASE --train "$SYN_TR" --val "$SYN_VAL" --out "$P1" --seed $SEED $HP $P1MAX --device $GPU
      [ -f "$P1"/best/val.json ] || { echo "PHASE1 FAIL"; exit 1; }
      touch "$P1"/DONE
      SO=$ROOT/blockA/ca_synth-only_${SYN}_seed${SEED}; mkdir -p "$SO"
      $RUN test --model "$P1"/best --test "$TEST" --out "$SO" --device $GPU $TESTOPT && echo "date=$(date -Iseconds) arch=trocr recipe=$R protocol=synth-only syn=$SYN model=$P1/best" >> "$SO"/run_info.txt
    fi
    START=$P1/best
    if [ "$SIZE" = full ]; then OUT=$A_CUR; MDIR=$OUT/phase2; else OUT=$ROOT/blockC/ca_curriculum${TAG}_${SYN}_seed${SEED}; MDIR=$OUT; fi ;;
  *) echo "unknown protocol: $PROT"; exit 1 ;;
esac
[ -f "$OUT"/test_results.json ] && { echo "already done: $OUT"; exit 0; }
mkdir -p "$OUT" "$MDIR"
echo "date=$(date -Iseconds) arch=trocr recipe=$R base=$BASE protocol=$PROT size=$SIZE seed=$SEED start=$START train=$REAL_TR hp=[$HP] gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
$RUN train --model "$START" --train "$REAL_TR" --val "$REAL_VAL" --out "$MDIR" --seed $SEED $HP --device $GPU
[ -f "$MDIR"/best/val.json ] || { echo "TRAIN FAIL"; exit 1; }
$RUN test --model "$MDIR"/best --test "$TEST" --out "$OUT" --device $GPU $TESTOPT
if [ "${TROCR_KEEP:-1}" = 0 ]; then rm -f "$MDIR"/best/model.safetensors
  [ "$PROT" = curriculum ] && [ "$SIZE" = 500 ] && rm -f "$P1"/best/model.safetensors
  echo "weights removed (TROCR_KEEP=0); logs, val.json and test results kept"; fi
echo "=== DONE trocr $R ca $PROT $SIZE seed$SEED ==="
