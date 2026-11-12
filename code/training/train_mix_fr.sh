#!/bin/bash
# Mixt FR - synthetic + real in a SINGLE phase, standard LR.
# The natural alternative to curriculum. vs real-only 4.48, curriculum 4.70.
set -e
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=${1:-cuda:0}
SYN=data/synthetic/fr_page_v1_unique
OUT=models/experiments/E03_fr_mix
mkdir -p "$OUT" results/predictions/E03_fr_mix

mkdir -p /tmp/manifests
# Validation: both real and synthetic val, matching the mixed training.
printf '%s\n%s\n' \
  "data/real/rimes/rimes_only_val.arrow" \
  "$SYN/fr_synth_10k_val.arrow" > /tmp/manifests/mix_val.txt

echo "=== MIX FR: real + synthetic, single phase, lr 1e-4 ==="
python $KS -d $GPU train \
  -i models/base/FoNDUE-GD_v2_fr.mlmodel \
  -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
  -o "$OUT"/E03 \
  -e /tmp/manifests/mix_val.txt \
  data/real/rimes/rimes_only_train.arrow \
  "$SYN"/fr_synth_10k_train.arrow \
  2>&1 | tee "$OUT"/train.log

echo "=== DONE mix ==="
