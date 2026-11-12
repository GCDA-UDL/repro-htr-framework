#!/bin/bash
# Audit round 3 orchestrator (controls B1' and B9). Launch detached so that it survives the editor session:
#   tmux new-session -d -s audit3 -c /home/joc13/workrepo-htr-synthetic-framework 'bash audit_round3/scripts/run_round3.sh'
# Windows of the tmux session "audit3":
#   0 (this) : the two B9 pools on CPU (gen_pool.sh), one after the other
#   gpu0     : chain on cuda:0, odd seeds   } each seed block runs s1, s2, s2x415, s2xall, so that an interruption
#   gpu1     : chain on cuda:1, even seeds  } leaves every arm with the same number of seeds; B9 jobs wait for their pool
# 4 arms x 6 seeds = 24 runs (phase 1 fixed 20 epochs + phase 2 on 500 real lines), about 1 h each, about 12 h in total.
# Every step is idempotent (DONE markers, registered rows and existing predictions are skipped): relaunching resumes.
ROOT=/home/joc13/workrepo-htr-synthetic-framework
cd "$ROOT" || exit 1
mkdir -p audit_round3/logs
exec > >(tee -a audit_round3/logs/run_round3.log) 2>&1
echo "$(date -Iseconds) run_round3 start (pid $$, tmux session: ${TMUX:-none})"
block() { for s in "$@"; do printf ' s1:%s s2:%s s2x415:%s s2xall:%s' "$s" "$s" "$s" "$s"; done; }
tmux new-window -d -t audit3 -n gpu0 "bash -c 'bash audit_round3/scripts/chain_r3.sh cuda:0$(block 1 3 5); sleep 86400'"
tmux new-window -d -t audit3 -n gpu1 "bash -c 'bash audit_round3/scripts/chain_r3.sh cuda:1$(block 2 4 6); sleep 86400'"
D=audit_round3/data/synthetic
bash audit_round3/scripts/gen_pool.sh ca_kidstextpert415ext_white_6k audit_round3/data/text/catk12hw_415ext_x12 \
  "Control B9a (S2-415ext): 415 training transcriptions NOT in the 500-line subset (case-insensitive), with the frequency profile of the S1 corpus (x12 -> 6000 lines; make_corpus_415ext.py, seed 42). Same textual diversity as S1, new in-domain text." \
  2>&1 | tee -a audit_round3/logs/pool_415ext.log
[ -f "$D/ca_kidstextpert415ext_white_6k/GENERATION.md" ] || touch "$D/ca_kidstextpert415ext_white_6k.FAILED"
bash audit_round3/scripts/gen_pool.sh ca_kidstextpertextall_white_6k audit_round3/data/text/catk12hw_extall_x5 \
  "Control B9b (S2-extall): every training line whose text (case-insensitive) is NOT in the 500-line subset, natural frequencies (1919 lines x5 -> 9595 lines; make_corpus_415ext.py, seed 42). High diversity, no text of the 500 labelled lines." \
  2>&1 | tee -a audit_round3/logs/pool_extall.log
[ -f "$D/ca_kidstextpertextall_white_6k/GENERATION.md" ] || touch "$D/ca_kidstextpertextall_white_6k.FAILED"
echo "$(date -Iseconds) run_round3: pools done; chains keep running in windows gpu0 / gpu1"
sleep 86400
