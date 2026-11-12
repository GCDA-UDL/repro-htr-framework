#!/bin/bash
# Campaign-2 job queue worker (06/10/2026).
# Usage: bash code/training/queue_worker.sh <cuda:N> <worker-id> [<chain-log> <n-done>]
#   Takes jobs from models/queue/jobs_c2.txt, one line per job:  <name> ::: <requires-glob or -> ::: <command>
#   (__GPU__ in the command is replaced by <cuda:N>), runs them one after another and appends START / END lines to
#   models/queue/done_c2.txt; the full output of each job goes to models/queue/logs/<name>.log. A job is taken only
#   when its requires-glob matches an existing file (e.g. the best_* file of the finished phase 1 it starts from);
#   the first ready job in file order wins; several workers share the file under flock. With <chain-log> <n-done>
#   the worker first waits until that log holds at least n-done "=== DONE" lines (the GPU slot of a running chain
#   becomes free). Two consecutive failed jobs stop the worker (the remaining jobs stay in the queue).
#   Never edit this file while a worker is running.
cd ~/workrepo-htr-synthetic-framework || exit 1
GPU=$1; ID=$2; WAITLOG=$3; WAITN=$4
[ -n "$GPU" ] && [ -n "$ID" ] || { echo "usage: bash code/training/queue_worker.sh <cuda:N> <worker-id> [<chain-log> <n-done>]"; exit 1; }
Q=models/queue/jobs_c2.txt; LOCK=models/queue/.jobs.lock; DONE=models/queue/done_c2.txt; LOG=models/queue/worker_${ID}.log
mkdir -p models/queue/logs
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Iseconds) worker $ID on $GPU started (pid $$, python: $(command -v python))"
python -c "import kraken" 2>/dev/null || { echo "$(date -Iseconds) kraken is not importable with $(command -v python) - worker $ID stops"; exit 1; }
if [ -n "$WAITLOG" ]; then
  echo "$(date -Iseconds) waiting until $WAITLOG has $WAITN '=== DONE' lines"
  until [ -f "$WAITLOG" ] && [ "$(grep -c '=== DONE' "$WAITLOG")" -ge "$WAITN" ]; do sleep 300; done
  echo "$(date -Iseconds) $WAITLOG has $WAITN DONE - slot free"
fi
fails=0
while :; do
  JOB=$( ( flock 9
    jobs=0; pick=""; n=0
    while IFS= read -r line || [ -n "$line" ]; do
      n=$((n+1))
      case "$line" in ''|'#'*) continue;; esac
      jobs=$((jobs+1))
      [ -n "$pick" ] && continue
      req=$(printf '%s\n' "$line" | awk -F' ::: ' '{print $2}')
      if [ "$req" = "-" ] || compgen -G "$req" > /dev/null; then pick=$n; fi
    done < "$Q"
    if [ "$jobs" -eq 0 ]; then echo "__EMPTY__"
    elif [ -z "$pick" ]; then echo "__WAIT__"
    else sed -n "${pick}p" "$Q"; sed -i "${pick}d" "$Q"; fi
  ) 9>"$LOCK" )
  case "$JOB" in
    __EMPTY__) echo "$(date -Iseconds) queue empty - worker $ID finished"; break;;
    __WAIT__)  sleep 600; continue;;
  esac
  NAME=$(printf '%s\n' "$JOB" | awk -F' ::: ' '{print $1}')
  CMD=$(printf '%s\n' "$JOB" | awk -F' ::: ' '{print $3}' | sed "s|__GPU__|$GPU|g")
  echo "$(date -Iseconds) worker=$ID gpu=$GPU START $NAME" | tee -a "$DONE"
  echo "  $CMD  (output: models/queue/logs/$NAME.log)"
  bash -c "$CMD" > "models/queue/logs/$NAME.log" 2>&1; rc=$?
  tail -n 2 "models/queue/logs/$NAME.log" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 2
  echo "$(date -Iseconds) worker=$ID gpu=$GPU END rc=$rc $NAME" | tee -a "$DONE"
  if [ "$rc" -ne 0 ]; then
    fails=$((fails+1)); echo "$(date -Iseconds) FAILED $NAME (rc=$rc, worker $ID)" | tee -a models/queue/FAILED_c2.txt
    [ "$fails" -ge 2 ] && { echo "$(date -Iseconds) two consecutive failures - worker $ID stops"; break; }
    sleep 600
  else
    fails=0
  fi
done
