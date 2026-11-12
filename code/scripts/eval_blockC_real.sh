#!/bin/bash
# Evaluate Block C real-only sweep models (500/1k/2k/4k, FR+EN) with ketos test, saving full reports
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
best() { ls "$1"/checkpoint_*.ckpt 2>/dev/null | sed 's/.*checkpoint_[0-9]*-\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2; }
for L in fr en; do
  if [ "$L" = fr ]; then GPU=cuda:0; TEST=data/real/rimes/rimes_test.arrow; else GPU=cuda:1; TEST=data/real/iam/iam_test.arrow; fi
  for N in 500 1k 2k 4k; do
    D=models/blockC/${L}_real-only_$N; M=$(best "$D/m")
    [ -z "$M" ] && { echo "SKIP $L $N (no checkpoint)"; continue; }
    echo "=== $L real-only $N -> $M ==="
    python $KS -d $GPU test -m "$M" -f binary -u NFD "$TEST" > "$D/eval_full.log" 2>&1
    echo -n "tracebacks: "; grep -c "Traceback" "$D/eval_full.log"
    grep -E "Character Accuracy|Word Accuracy" "$D/eval_full.log"
  done
done
echo "=== ALL DONE ==="
