#!/bin/bash
# Audit round 2 orchestrator. Launch detached so that it survives the editor session:
#   tmux new-session -d -s audit2 -c /home/joc13/workrepo-htr-synthetic-framework 'bash audit_round2/scripts/run_round2.sh'
# Windows of the tmux session "audit2":
#   0 (this)  : pool B1 generation (CPU), then B1 chain seeds 1 3 5 on cuda:0
#   predict   : line-by-line predictions of the existing Catalan campaign-2 checkpoints on cuda:1 (inference only)
#   g1        : B1 chain seeds 2 4 6 on cuda:1 (started once the pool exists)
# Every step is idempotent (DONE markers, existing arrows / json skipped), so the script can simply be relaunched.
ROOT=/home/joc13/workrepo-htr-synthetic-framework
PY=/home/joc13/miniconda3/envs/kraken7/bin/python
cd "$ROOT" || exit 1
mkdir -p audit_round2/logs
exec > >(tee -a audit_round2/logs/run_round2.log) 2>&1
echo "$(date -Iseconds) run_round2 start (pid $$, tmux session: ${TMUX:-none})"
tmux new-window -d -t audit2 -n predict "bash -c '$PY audit_round2/scripts/predict_lines_r2.py --device cuda:1 --list audit_round2/results/predictions_runs.tsv 2>&1 | tee -a audit_round2/logs/predict.log; sleep 86400'"
bash audit_round2/scripts/gen_pool_b1.sh 2>&1 | tee -a audit_round2/logs/pool_b1.log
[ -f audit_round2/data/synthetic/ca_kidstextpert500_white_6k/GENERATION.md ] || { echo "pool B1 missing - stop"; sleep 86400; exit 1; }
tmux new-window -d -t audit2 -n g1 "bash -c 'bash audit_round2/scripts/chain_b1.sh cuda:1 2 4 6; sleep 86400'"
bash audit_round2/scripts/chain_b1.sh cuda:0 1 3 5
echo "$(date -Iseconds) run_round2: cuda:0 chain finished"
sleep 86400
