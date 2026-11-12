#!/bin/bash
# Mix EN (E09) - IAM real + synthetic EN, single phase, lr 1e-4. Analog of E03.
set -e
cd ~/workrepo-htr-synthetic-framework
KS=code/scripts/ketos_safe.py
GPU=${1:-cuda:0}
SYN=data/synthetic/en_page
OUT=models/experiments/E09_en_mix
mkdir -p "$OUT" results/predictions/E09_en_mix

mkdir -p /tmp/manifests
printf '%s\n%s\n' \
  "data/real/iam/iam_val.arrow" \
  "$SYN/en_synth_9k_val.arrow" > /tmp/manifests/mix_en_val.txt

echo "=== MIX EN: IAM real + synth, single phase, lr 1e-4 ==="
python $KS -d $GPU train \
  -i models/base/FoNDUE-GD_v2.mlmodel \
  -f binary --resize add -r 0.0001 --lag 5 --min-epochs 3 -u NFD \
  -o "$OUT"/E09 \
  -e /tmp/manifests/mix_en_val.txt \
  data/real/iam/iam_train.arrow \
  "$SYN"/en_synth_9k_train.arrow \
  2>&1 | tee "$OUT"/train.log
echo "=== DONE mix EN ==="
