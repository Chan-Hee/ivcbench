#!/usr/bin/env bash
# Keep the GPUs busy without a human in the loop.
#
# Reads runs/queue.txt (one job per line: "<job-id>\t<gpu>\t<command>[\t<needs>]"; '#' comments
# ignored). <needs> is an optional comma-separated list of job ids that must have finished DONE
# before this one may launch -- the GPU-memory check alone is not enough, because a job that is
# still loading data holds almost no GPU memory and would be treated as absent.
# and every POLL seconds launches the next queued job whose target GPU is free. A job is
# considered launched once runs/<id>.status exists, so restarting the dispatcher never
# double-launches. Exits when the queue is exhausted and nothing is RUNNING.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL="${POLL:-60}"
. "$DIR/env.sh" 2>/dev/null || true   # so IVCBENCH_JOBS_PER_GPU reaches preflight
FREE_MB="${FREE_MB:-20000}"   # a GPU counts as free below this many MiB in use

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 99999; }

while true; do
  pending=0
  while IFS=$'\t' read -r id gpu cmd needs; do
    [ -z "${id:-}" ] && continue
    case "$id" in \#*) continue;; esac
    # A queue rewritten while this loop is reading it yields a TRUNCATED line, whose first field
    # then looks like a job id -- that happened, and preflight refused a job called
    # "kic_timing_shard5.json". Require the three fields a real entry has.
    [ -z "${gpu:-}" ] || [ -z "${cmd:-}" ] && continue
    [ -f "$DIR/$id.status" ] && continue          # already launched (or finished)
    pending=$((pending+1))
    blocked=0
    if [ -n "${needs:-}" ]; then
      IFS=',' read -ra deps <<< "$needs"
      for d in "${deps[@]}"; do
        [ -z "$d" ] && continue
        grep -q '^DONE:0' "$DIR/$d.status" 2>/dev/null || blocked=1
      done
    fi
    [ "$blocked" -eq 1 ] && continue
    # The queue names a GPU, but a card that is full while another sits idle should not hold a job
    # back. Pick the emptiest card that is under the memory bar and rewrite the command to match --
    # the per-GPU job limit in preflight.py still decides whether it may start.
    best=""; best_used=999999
    for g in 0 1 2 3; do
      case " ${IVCBENCH_RESERVED_GPUS:-} " in *" $g "*) continue;; esac
      u=$(gpu_used "$g")
      n=$(grep -l RUNNING "$DIR"/*.status 2>/dev/null | while read -r f; do
            b=$(basename "$f" .status)
            [ -f "$DIR/$b.cmd" ] && grep -qE -- "--gpu $g|CUDA_VISIBLE_DEVICES=$g|--gpus $g" "$DIR/$b.cmd" && echo x
          done | wc -l)
      # rank by job count first, memory second
      rank=$(( n * 100000 + u ))
      if [ "$rank" -lt "$best_used" ]; then best_used=$rank; best=$g; fi
    done
    if [ -n "$best" ] && [ "$best" != "$gpu" ]; then
      cmd=$(printf '%s' "$cmd" | sed -E "s/--gpu [0-9]+/--gpu $best/; s/CUDA_VISIBLE_DEVICES=[0-9]+/CUDA_VISIBLE_DEVICES=$best/; s/--gpus [0-9]+/--gpus $best/")
      echo "$(date -u +%FT%TZ) rebalance $id: gpu $gpu -> $best"
      gpu="$best"
    fi
    case " ${IVCBENCH_RESERVED_GPUS:-} " in
      *" $gpu "*) continue;;          # reserved card: this job waits for an unreserved one
    esac
    used=$(gpu_used "$gpu")
    if [ "$used" -lt "$FREE_MB" ]; then
      echo "$(date -u +%FT%TZ) dispatch $id -> gpu $gpu (used ${used}MiB)"
      "$DIR/launch.sh" "$id" "$cmd"
      sleep 20
      # Restart the pass from the top of the queue after every launch. Without this the queue is
      # not a priority list: an entry refused early in a pass (every card at its limit) stays
      # refused while a card frees LATER in the same pass, and whichever entry happens to be next
      # takes it. Three CellOT lineages -- 8 h each, the critical path -- sat at the head of the
      # queue while shorter jobs below them were dispatched.
      break
    fi
  done < "$DIR/queue.txt"
  running=$(grep -l RUNNING "$DIR"/*.status 2>/dev/null | wc -l)
  if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) queue drained and nothing running"
    break
  fi
  sleep "$POLL"
done
