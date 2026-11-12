#!/bin/bash
# Phase-2-only variant of train_blockA.sh (2026-10-08): for a curriculum run whose phase 1 finished and whose phase 2
# crashed (disk full), it reuses the phase-1 checkpoints already in $OUT/phase1/m and reruns only phase 2. Same arguments.
# Block A: 6 protocols. Usage: bash train_blockA.sh <GPU> <fr|en|ca> <baseline|synth-only|real-only|mix|curriculum|curriculum-LR>
# Optional environment variables:
#   SEED=<int>        passed to ketos as --seed (reproducible runs; used for the Catalan replicates)
#   BASE_CA=<path>    base model for ca (default models/base/FoNDUE-GD_v2_fr.mlmodel)
#   OUT_SUFFIX=<str>  appended to the output folder name, e.g. _base-v2 or _seed2
#   SYN_TR_OVR=<path>, SYN_VAL_OVR=<path>  replace the synthetic train / validation arrows (content ablations)
#   PHASE1_DIR=<dir>  phase-1 checkpoint dir to start from (default $OUT/phase1/m), e.g. the phase 1 of another run of the same seed
#   AUGMENT=1         ketos --augment in phase 2 (2026-10-18: kidstext-pert + augment with the full real data)
set -e
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=$1; LANG=$2; PROT=$3
SEEDOPT=""; [ -n "$SEED" ] && SEEDOPT="--seed $SEED"

if [ "$LANG" = "fr" ]; then
  BASE=models/base/FoNDUE-GD_v2_fr.mlmodel
  REAL_TR=data/real/rimes/rimes_only_train.arrow
  REAL_VAL=data/real/rimes/rimes_only_val.arrow
  TEST=data/real/rimes/rimes_test.arrow
  SYN_TR=data/synthetic/fr_pool_30k/fr_synth_10188_train_bl.arrow
  SYN_VAL=data/synthetic/fr_pool_30k/fr_synth_30k_validation_bl.arrow
elif [ "$LANG" = "en" ]; then
  BASE=models/base/FoNDUE-GD_v2.mlmodel
  REAL_TR=data/real/iam/iam_train.arrow
  REAL_VAL=data/real/iam/iam_val.arrow
  TEST=data/real/iam/iam_test.arrow
  SYN_TR=data/synthetic/en_pool_30k/en_synth_8917_train_bl.arrow
  SYN_VAL=data/synthetic/en_pool_30k/en_synth_30k_validation_bl.arrow
elif [ "$LANG" = "ca" ]; then
  BASE=${BASE_CA:-models/base/FoNDUE-GD_v2_fr.mlmodel}
  REAL_TR=data/real/catk12hw/catk12hw_train.arrow
  REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
  TEST=data/real/catk12hw/catk12hw_test.arrow
  SYN_TR=data/synthetic/ca_pool_30k/ca_synth_4152_train_bl.arrow
  SYN_VAL=data/synthetic/ca_pool_30k/ca_synth_30k_validation_bl.arrow
else
  echo "unknown language: $LANG"; exit 1
fi
[ -n "$SYN_TR_OVR" ] && SYN_TR=$SYN_TR_OVR
[ -n "$SYN_VAL_OVR" ] && SYN_VAL=$SYN_VAL_OVR

OUT=models/blockA/${LANG}_${PROT}${OUT_SUFFIX}
mkdir -p "$OUT"
mkdir -p /tmp/man
echo "$REAL_VAL" > /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt
echo "$SYN_VAL"  > /tmp/man/${LANG}${OUT_SUFFIX}_syn_val.txt
echo "date=$(date -Iseconds) lang=$LANG protocol=$PROT base=$BASE seed=${SEED:-none} syn=$SYN_TR gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt

case $PROT in
  baseline)
    echo "=== baseline: eval base model, no training ==="
    python $KS -d $GPU test -m $BASE -f binary -u NFD $TEST 2>&1 | tee "$OUT"/eval.log
    ;;
  synth-only)
    python $KS $SEEDOPT -d $GPU train -i $BASE -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$OUT"/m -e /tmp/man/${LANG}${OUT_SUFFIX}_syn_val.txt $SYN_TR 2>&1 | tee "$OUT"/train.log
    ;;
  real-only)
    python $KS $SEEDOPT -d $GPU train -i $BASE -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$OUT"/m -e /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt $REAL_TR 2>&1 | tee "$OUT"/train.log
    ;;
  mix)
    python $KS $SEEDOPT -d $GPU train -i $BASE -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$OUT"/m -e /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt $REAL_TR $SYN_TR 2>&1 | tee "$OUT"/train.log
    ;;
  curriculum|curriculum-LR)
    P2LR=0.0001; [ "$PROT" = "curriculum-LR" ] && P2LR=0.00001
    mkdir -p "$OUT"/phase1 "$OUT"/phase2
    P1M=${PHASE1_DIR:-$OUT/phase1/m}
    ls "$P1M"/checkpoint_*.ckpt > /dev/null 2>&1 || { echo "PHASE2ONLY: no phase-1 checkpoints in $P1M"; exit 1; }
    echo "=== phase1 reused from $P1M (phase-2-only run) augment=${AUGMENT:-0} ===" | tee -a "$OUT"/run_info.txt
    rm -rf "$OUT"/phase2 "$OUT"/phase2.log; mkdir -p "$OUT"/phase2
    B1=$(ls "$P1M"/checkpoint_*.ckpt | sed 's/.*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2)
    [ -z "$B1" ] && { echo "PHASE1 FAIL"; exit 1; }
    echo "=== phase2: real, lr $P2LR ==="
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize add -r $P2LR --lag 5 --min-epochs 3 -u NFD ${AUGMENT:+--augment} \
      -o "$OUT"/phase2/m -e /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt $REAL_TR 2>&1 | tee "$OUT"/phase2.log
    ;;
esac
echo "=== DONE ${LANG} ${PROT} ==="
