#!/bin/bash
# Braç des de zero (sense model base FoNDUE), només català.
# Ús: bash train_scratch.sh <GPU> <real-only|curriculum> <full|500> <seed>
# Passos des de zero (real-only i fase 1): LR per defecte de Kraken, --lag 10, --min-epochs 20.
# Fase 2 del curriculum: igual que la resta d'experiments (-r 0.0001 --lag 5 --min-epochs 3).
set -eo pipefail
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=$1; PROT=$2; SIZE=$3; SEED=$4
[ -n "$SEED" ] || { echo "usage: $0 GPU real-only|curriculum full|500 SEED"; exit 1; }
SEEDOPT="--seed $SEED"
SCR="--lag 10 --min-epochs 20"
REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
if [ "$SIZE" = full ]; then REAL_TR=data/real/catk12hw/catk12hw_train.arrow; BLK=blockA; TAG=""
else REAL_TR=data/real/catk12hw/catk12hw_${SIZE}_train.arrow; BLK=blockC; TAG="_${SIZE}"; fi
SYN_TR=data/synthetic/ca_kidstext_white_6k/ca_kidstext_white_4152_train_bl.arrow
SYN_VAL=data/synthetic/ca_kidstext_white_6k/ca_kidstext_white_6k_validation_bl.arrow
for f in "$REAL_TR" "$REAL_VAL" "$SYN_TR" "$SYN_VAL"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
best(){ ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }
mkdir -p /tmp/man
RV=/tmp/man/ca_scratch_s${SEED}${TAG}_real_val.txt; echo "$REAL_VAL" > "$RV"
SV=/tmp/man/ca_scratch_s${SEED}_syn_val.txt;        echo "$SYN_VAL"  > "$SV"
A_CUR=models/blockA/ca_curriculum_scratch-kidstext_seed${SEED}
P1=$A_CUR/phase1
case $PROT in
  real-only)
    OUT=models/$BLK/ca_real-only${TAG}_scratch_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=real-only size=$SIZE base=none(scratch) lr=kraken-default lag=10 min_epochs=20 seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -f binary $SCR -u NFD \
      -o "$OUT"/m -e "$RV" "$REAL_TR" 2>&1 | tee "$OUT"/train.log
    ;;
  curriculum)
    if [ ! -f "$P1"/DONE ]; then
      [ "$SIZE" = full ] || { echo "phase1 not finished: $P1"; exit 1; }
      mkdir -p "$P1"
      echo "date=$(date -Iseconds) lang=ca protocol=curriculum phase1 base=none(scratch) syn=$SYN_TR lr=kraken-default lag=10 min_epochs=20 seed=$SEED gpu=$GPU" | tee -a "$A_CUR"/run_info.txt
      python $KS $SEEDOPT -d $GPU train -f binary $SCR -u NFD \
        -o "$P1"/m -e "$SV" "$SYN_TR" 2>&1 | tee "$A_CUR"/phase1.log
      [ -n "$(best "$P1"/m)" ] || { echo "PHASE1 FAIL"; exit 1; }
      touch "$P1"/DONE
    fi
    B1=$(best "$P1"/m)
    if [ "$SIZE" = full ]; then OUT=$A_CUR; P2=$OUT/phase2; LOG=$OUT/phase2.log
    else OUT=models/blockC/ca_curriculum${TAG}_scratch-kidstext_seed${SEED}; P2=$OUT; LOG=$OUT/train.log; fi
    mkdir -p "$P2"
    echo "date=$(date -Iseconds) lang=ca protocol=curriculum phase2 size=$SIZE phase1=$B1 phase2_lr=0.0001 seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize union -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$P2"/m -e "$RV" "$REAL_TR" 2>&1 | tee "$LOG"
    ;;
  *) echo "unknown protocol: $PROT"; exit 1 ;;
esac
echo "=== DONE ca $PROT $SIZE scratch seed$SEED ==="
