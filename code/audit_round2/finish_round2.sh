#!/bin/bash
# Audit round 2: closing steps, launched in its own tmux window and safe to relaunch.
#   1. wait until the campaign-2 predictions are complete (213 json) -> kraken_lines_analysis.py --tag c2
#   2. wait until both B1 chains have finished -> line-by-line predictions of the B1 models (cuda:1)
#      -> kraken_lines_analysis.py --tag final -> audit_round2/results/ROUND2_DONE
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
cd "$ROOT" || exit 1
exec > >(tee -a audit_round2/logs/finish_round2.log) 2>&1
echo "$(date -Iseconds) finish_round2 start"
N=$(grep -vc '^#' audit_round2/results/predictions_runs.tsv)
until [ "$(ls audit_round2/results/predictions/c2_*.json 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 120; done
echo "$(date -Iseconds) campaign-2 predictions complete ($N)"
[ -f audit_round2/results/kraken_lines_c2.md ] || $PY -I -B audit_round2/scripts/kraken_lines_analysis.py --tag c2 > /dev/null
echo "$(date -Iseconds) analysis c2 written"
until grep -q 'chain on cuda:0 finished' audit_round2/logs/chain_b1_cuda0.log 2>/dev/null && grep -q 'chain on cuda:1 finished' audit_round2/logs/chain_b1_cuda1.log 2>/dev/null; do sleep 300; done
echo "$(date -Iseconds) both B1 chains finished"
flock audit_round2/results/.eval.lock $PY audit_round2/scripts/eval_r2.py --device cuda:1
L=audit_round2/results/predictions_runs_r2.tsv
{ echo -e "# name\tcheckpoint dir (audit round 2, B1 models)"
  for d in audit_round2/models/ca_curriculum_kidstextpert500_seed*/phase2_*; do
    [ -f "$d/DONE" ] || continue; s=${d%/phase2_*}; s=${s##*_seed}; sz=${d##*phase2_}
    echo -e "r2_ca_curriculum_${sz}_kidstextpert500_seed${s}\t$d/m"
  done
  for d in audit_round2/models/ca_curriculum_kidstextpert500_seed*/phase1; do
    [ -f "$d/DONE" ] || continue; s=${d%/phase1}; s=${s##*_seed}
    echo -e "r2_ca_synth-only_kidstextpert500_seed${s}\t$d/m"
  done; } > $L
$PY audit_round2/scripts/predict_lines_r2.py --device cuda:1 --list $L
$PY -I -B audit_round2/scripts/kraken_lines_analysis.py --tag final > /dev/null
date -Iseconds > audit_round2/results/ROUND2_DONE
echo "$(date -Iseconds) finish_round2 done"
sleep 86400
