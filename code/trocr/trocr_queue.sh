#!/bin/bash
# Sequential TrOCR runner for one GPU. Usage: bash trocr_queue.sh <cuda:N> [jobs file] [root dir]
# Jobs: lines "<name> ::: <command with __GPU__>". Skips jobs with "END rc=0 <name>" in <root>/done_trocr.txt;
# output of each job -> <root>/logs/<name>.log.
cd ~/workrepo-htr-synthetic-framework || exit 1
GPU=$1; Q=${2:-models/queue/jobs_trocr.txt}; ROOT=${3:-models/trocr}
[ -n "$GPU" ] || { echo "usage: $0 cuda:N [jobs] [root]"; exit 1; }
DONE=$ROOT/done_trocr.txt; mkdir -p $ROOT/logs; touch $DONE
while IFS= read -r line; do
  { [ -z "$line" ] || [ "${line:0:1}" = "#" ]; } && continue
  name=${line%% ::: *}; cmd=${line#* ::: }; cmd=${cmd//__GPU__/$GPU}
  grep -q "END rc=0 $name\$" $DONE && { echo "skip $name (done)"; continue; }
  echo "$(date -Iseconds) START $name" | tee -a $DONE
  bash -c "$cmd" > $ROOT/logs/$name.log 2>&1; rc=$?
  echo "$(date -Iseconds) END rc=$rc $name" | tee -a $DONE
done < $Q
echo "$(date -Iseconds) queue finished"
