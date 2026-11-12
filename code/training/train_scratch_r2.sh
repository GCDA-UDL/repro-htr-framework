#!/bin/bash
# Braç des de zero, RECEPTA 2 (pilot), només català i tot el real.
# Ús: bash train_scratch_r2.sh <GPU> <real-only|curriculum|real-only-2stage|real-only-500|curriculum-500|real-only-500-2stage> <seed>
# Passos des de zero (real-only i fase 1): -B 16, lr 1e-3, warmup 250, reduceonplateau (patience 5, gamma 0.1),
#   --augment, --lag 20, --min-epochs 50. Fase 1 = ca_white_30k complet.
# Fase 2 del curriculum: igual que la resta (-r 0.0001 --lag 5 --min-epochs 3, lot 1), --resize union.
set -eo pipefail
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=$1; PROT=$2; SEED=$3
[ -n "$SEED" ] || { echo "usage: $0 GPU real-only|curriculum SEED"; exit 1; }
SEEDOPT="--seed $SEED"
R2="-B 16 -r 0.001 --warmup 250 --schedule reduceonplateau --sched-patience 5 -g 0.1 --augment --lag 20 --min-epochs 50"
REAL_TR=data/real/catk12hw/catk12hw_train.arrow
REAL_VAL=data/real/catk12hw/catk12hw_val.arrow
REAL500=data/real/catk12hw/catk12hw_500_train.arrow   # Block C subset (seed 42), for the *-500 protocols
SCR_SYN=${SCR_SYN:-white30k}   # phase-1 pool: white30k (generic, clean) | whitepert30k (generic, --perturbations)
case $SCR_SYN in
  white30k)     SYN_TR=data/synthetic/ca_white_30k/ca_white_30k_train_bl.arrow;         SYN_VAL=data/synthetic/ca_white_30k/ca_white_30k_validation_bl.arrow ;;
  whitepert30k) SYN_TR=data/synthetic/ca_whitepert_30k/ca_whitepert_30k_train_bl.arrow; SYN_VAL=data/synthetic/ca_whitepert_30k/ca_whitepert_30k_validation_bl.arrow ;;
  *) echo "unknown SCR_SYN: $SCR_SYN"; exit 1 ;;
esac
for f in "$REAL_TR" "$REAL_VAL" "$SYN_TR" "$SYN_VAL"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
best(){ ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }
mkdir -p /tmp/man
RV=/tmp/man/ca_scratchr2_s${SEED}_real_val.txt; echo "$REAL_VAL" > "$RV"
SV=/tmp/man/ca_scratchr2-${SCR_SYN}_s${SEED}_syn_val.txt;  echo "$SYN_VAL"  > "$SV"
case $PROT in
  real-only)
    OUT=models/blockA/ca_real-only_scratchr2_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=real-only base=none(scratch) recipe=r2 [$R2] seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -f binary $R2 -u NFD -o "$OUT"/m -e "$RV" "$REAL_TR" 2>&1 | tee "$OUT"/train.log
    ;;
  curriculum)
    OUT=models/blockA/ca_curriculum_scratchr2-${SCR_SYN}_seed${SEED}; P1=$OUT/phase1; mkdir -p "$P1" "$OUT"/phase2
    if [ ! -f "$P1"/DONE ]; then
      echo "date=$(date -Iseconds) lang=ca protocol=curriculum phase1 base=none(scratch) recipe=r2 [$R2] syn=$SCR_SYN:$SYN_TR seed=$SEED gpu=$GPU" | tee -a "$OUT"/run_info.txt
      python $KS $SEEDOPT -d $GPU train -f binary $R2 -u NFD -o "$P1"/m -e "$SV" "$SYN_TR" 2>&1 | tee "$OUT"/phase1.log
      [ -n "$(best "$P1"/m)" ] || { echo "PHASE1 FAIL"; exit 1; }
      touch "$P1"/DONE
    fi
    B1=$(best "$P1"/m)
    echo "date=$(date -Iseconds) phase2 from=$B1 lr=0.0001 lag=5 min_epochs=3 resize=union seed=$SEED gpu=$GPU" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize union -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$OUT"/phase2/m -e "$RV" "$REAL_TR" 2>&1 | tee "$OUT"/phase2.log
    ;;
  real-only-2stage)
    # control for the curriculum arm: the real-only r2 model gets a second stage with the phase-2 recipe on the same real data
    SRC=models/blockA/ca_real-only_scratchr2_seed${SEED}; B1=$(best "$SRC"/m)
    [ -n "$B1" ] || { echo "no real-only r2 checkpoint for seed $SEED"; exit 1; }
    OUT=models/blockA/ca_real-only_scratchr2-2stage_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=real-only-2stage (control) stage1=$B1 stage2=phase-2 recipe (lr 1e-4, lag 5, min 3, B1, resize union) on the same real train seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize union -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
      -o "$OUT"/m -e "$RV" "$REAL_TR" 2>&1 | tee "$OUT"/train.log
    ;;
  real-only-500)
    [ -f "$REAL500" ] || { echo "missing $REAL500"; exit 1; }
    OUT=models/blockC/ca_real-only_500_scratchr2_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=real-only size=500 base=none(scratch) recipe=r2 [$R2] seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -f binary $R2 -u NFD -o "$OUT"/m -e "$RV" "$REAL500" 2>&1 | tee "$OUT"/train.log
    ;;
  curriculum-500)
    P1=models/blockA/ca_curriculum_scratchr2-${SCR_SYN}_seed${SEED}/phase1; B1=$(best "$P1"/m)
    [ -n "$B1" ] || { echo "no phase-1 checkpoint in $P1/m"; exit 1; }
    OUT=models/blockC/ca_curriculum_500_scratchr2-${SCR_SYN}_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=curriculum size=500 phase1=$B1 phase2_lr=0.0001 lag=5 min_epochs=3 resize=union seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize union -r 0.0001 --lag 5 --min-epochs 3 -u NFD -o "$OUT"/m -e "$RV" "$REAL500" 2>&1 | tee "$OUT"/train.log
    ;;
  real-only-500-2stage)
    SRC=models/blockC/ca_real-only_500_scratchr2_seed${SEED}; B1=$(best "$SRC"/m)
    [ -n "$B1" ] || { echo "no real-only-500 r2 checkpoint for seed $SEED"; exit 1; }
    OUT=models/blockC/ca_real-only_500_scratchr2-2stage_seed${SEED}; mkdir -p "$OUT"
    echo "date=$(date -Iseconds) lang=ca protocol=real-only-2stage (control) size=500 stage1=$B1 stage2=phase-2 recipe seed=$SEED gpu=$GPU out=$OUT" | tee -a "$OUT"/run_info.txt
    python $KS $SEEDOPT -d $GPU train -i "$B1" -f binary --resize union -r 0.0001 --lag 5 --min-epochs 3 -u NFD -o "$OUT"/m -e "$RV" "$REAL500" 2>&1 | tee "$OUT"/train.log
    ;;
  *) echo "unknown protocol: $PROT"; exit 1 ;;
esac
echo "=== DONE ca $PROT scratch r2 seed$SEED ==="
