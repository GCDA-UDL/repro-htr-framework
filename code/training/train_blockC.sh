#!/bin/bash
# Block C (real-data size sweep). Usage: bash train_blockC.sh <GPU> <fr|en|ca> <500|1k|2k|4k> <real-only|curriculum>
#   real-only : fine-tune the base model on the real subset only (same hyperparameters as Block A)
#   curriculum: phase 2 only, on the real subset, starting from the Block A curriculum phase-1 checkpoint
#               (synthetic 1:1 with the full real set), which does not depend on the real size; lr 1e-4
# Optional environment variables:
#   SEED=<int>        passed to ketos as --seed; for ca the phase-1 checkpoint of the same seed is used
#   OUT_SUFFIX=<str>  appended to the output folder name, e.g. _seed2
#   PHASE1_DIR=<dir>  override the phase-1 checkpoint folder (default derived from LANG and SEED)
#   BASE_CA=<path>    base model for ca (default models/base/FoNDUE-GD_v2_fr.mlmodel), e.g. the generic FoNDUE-GD_v2
set -e
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=$1; LANG=$2; SIZE=$3; PROT=$4
SEEDOPT=""; [ -n "$SEED" ] && SEEDOPT="--seed $SEED"
best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_[0-9]*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }

if [ "$LANG" = "fr" ]; then
  BASE=models/base/FoNDUE-GD_v2_fr.mlmodel
  REAL_TR=data/real/rimes/rimes_${SIZE}_train.arrow
  REAL_VAL=data/real/rimes/rimes_only_val.arrow
  P1=${PHASE1_DIR:-models/blockA/fr_curriculum/phase1/m}
elif [ "$LANG" = "en" ]; then
  BASE=models/base/FoNDUE-GD_v2.mlmodel
  REAL_TR=data/real/iam/iam_${SIZE}_train.arrow
  REAL_VAL=data/real/iam/iam_val.arrow
  P1=${PHASE1_DIR:-models/blockA/en_curriculum/phase1/m}
elif [ "$LANG" = "ca" ]; then
  BASE=${BASE_CA:-models/base/FoNDUE-GD_v2_fr.mlmodel}
  REAL_TR=data/real/catk12hw/catk12hw_${SIZE}_train.arrow
  REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
  P1=${PHASE1_DIR:-models/blockA/ca_curriculum_seed${SEED:-1}/phase1/m}
else
  echo "unknown language: $LANG"; exit 1
fi
[ -f "$REAL_TR" ] || { echo "missing real subset: $REAL_TR"; exit 1; }

OUT=models/blockC/${LANG}_${PROT}_${SIZE}${OUT_SUFFIX}
mkdir -p "$OUT" /tmp/man
echo "$REAL_VAL" > /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt

case $PROT in
  real-only)
    echo "date=$(date -Iseconds) lang=$LANG protocol=real-only size=$SIZE base=$BASE seed=${SEED:-none} gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i $BASE -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD ${AUGMENT:+--augment} \
      -o "$OUT"/m -e /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt $REAL_TR 2>&1 | tee "$OUT"/train.log
    ;;
  curriculum)
    B1=$(best "$P1")
    [ -z "$B1" ] && { echo "no phase-1 checkpoint in $P1"; exit 1; }
    echo "date=$(date -Iseconds) lang=$LANG protocol=curriculum size=$SIZE phase1=$B1 phase2_lr=0.0001 seed=${SEED:-none} gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD ${AUGMENT:+--augment} \
      -o "$OUT"/m -e /tmp/man/${LANG}${OUT_SUFFIX}_real_val.txt $REAL_TR 2>&1 | tee "$OUT"/train.log
    ;;
  *)
    echo "unknown protocol: $PROT"; exit 1
    ;;
esac
echo "=== DONE ${LANG} ${PROT} ${SIZE} ==="
