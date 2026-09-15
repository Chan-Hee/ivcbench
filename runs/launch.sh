#!/usr/bin/env bash
# Launch a long job in a detached screen session so it survives VSCode/SSH/network loss.
#   runs/launch.sh <job-id> <command...>
# Writes runs/<id>.log and runs/<id>.status (RUNNING -> DONE:<code> | FAILED:<code>).
set -u
ID="$1"; shift
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/$ID.log"; ST="$DIR/$ID.status"; CMD="$DIR/$ID.cmd"
printf '%s\n' "$*" > "$CMD"

# No job takes a GPU without passing the gate. Every check in preflight.py is a mistake that
# already cost GPU time -- roster, plan disposition, entry point, model assets, bundle deposit,
# overwrite, GPU collision, duplicate launch, code version.
"$(cd "$DIR/.." && pwd)/.venv/bin/python" "$DIR/preflight.py" "$ID" $*
PF=$?
if [ $PF -eq 8 ]; then
  # transient (a GPU is busy, or this id is already running): leave NO status file so the
  # dispatcher offers the job again on its next pass. Writing one would drop the cell for good.
  rm -f "$ST"
  echo "launch deferred for $ID -- will retry" >&2
  exit 0
elif [ $PF -ne 0 ]; then
  echo "REFUSED:$PF $(date -u +%FT%TZ) preflight" > "$ST"
  echo "launch refused for $ID -- see the reason above" >&2
  exit $PF
fi
cat > "$DIR/$ID.run.sh" <<INNER
#!/usr/bin/env bash
cd "$(cd "$DIR/.." && pwd)"
echo "RUNNING \$(date -u +%FT%TZ) pid \$\$" > "$ST"
# Model-asset env, asserted. A missing checkpoint path must fail the JOB here, loudly, rather
# than reach the runner and come back as ran=False -- which reads like a model limitation.
. "$DIR/env.sh" || { echo "FAILED:90 \$(date -u +%FT%TZ) runs/env.sh assertion" > "$ST"; exit 90; }
mkdir -p "\$IVCBENCH_PRED_DUMP"; date +%s > "\$IVCBENCH_PRED_DUMP/.$ID.t0"
$*
code=\$?
if [ \$code -eq 0 ]; then
  # rc=0 is not evidence of a filled cell: jobs have exited 0 with ran=False everywhere, and
  # with no bundle deposited at all. Check the output before calling this DONE.
  # the job's own result CSV, if its command names one with --out
  OUTCSV=\$(printf '%s\\n' "$*" | grep -oP '(?<=--out )\\S+' | head -1)
  if "$(cd "$DIR/.." && pwd)/.venv/bin/python" "$DIR/postflight.py" "$ID" "\$IVCBENCH_PRED_DUMP" \$OUTCSV; then
    echo "DONE:\$code \$(date -u +%FT%TZ)" > "$ST"
  else
    echo "UNUSABLE:4 \$(date -u +%FT%TZ) postflight" > "$ST"
  fi
else
  echo "FAILED:\$code \$(date -u +%FT%TZ)" > "$ST"
fi
INNER
chmod +x "$DIR/$ID.run.sh"
screen -dmS "ivc_$ID" bash -c "'$DIR/$ID.run.sh' > '$LOG' 2>&1"
sleep 1
echo "launched $ID  (screen: ivc_$ID)  log: runs/$ID.log"
