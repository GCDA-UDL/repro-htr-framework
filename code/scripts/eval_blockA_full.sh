#!/bin/bash
# Re-run ketos test on all Block A models, saving FULL reports (confusion tables) to eval_full.log
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_[0-9]*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }
for L in fr en; do
  if [ "$L" = fr ]; then GPU=cuda:0; BASE=models/base/FoNDUE-GD_v2_fr.mlmodel; TEST=data/real/rimes/rimes_test.arrow
  else GPU=cuda:1; BASE=models/base/FoNDUE-GD_v2.mlmodel; TEST=data/real/iam/iam_test.arrow; fi
  for P in baseline synth-only real-only mix curriculum curriculum-LR; do
    D=models/blockA/${L}_${P}; mkdir -p "$D"
    case $P in
      baseline) M=$BASE ;;
      curriculum|curriculum-LR) M=$(best "$D/phase2/m") ;;
      *) M=$(best "$D/m") ;;
    esac
    [ -z "$M" ] && { echo "SKIP $L $P (no model found)"; continue; }
    echo "=== $L $P -> $M ==="
    python $KS -d $GPU test -m "$M" -f binary -u NFD "$TEST" > "$D/eval_full.log" 2>&1
    grep -A5 "=== report" "$D/eval_full.log" | grep -E "Errors|Character Accuracy$|Word Accuracy"
  done
done
echo "=== ALL DONE ==="
