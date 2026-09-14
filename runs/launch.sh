#!/usr/bin/env bash
# Launch a long job in a detached screen session so it survives VSCode/SSH/network loss.
#   runs/launch.sh <job-id> <command...>
# Writes runs/<id>.log and runs/<id>.status (RUNNING -> DONE:<code> | FAILED:<code>).
set -u
ID="$1"; shift
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/$ID.log"; ST="$DIR/$ID.status"; CMD="$DIR/$ID.cmd"
printf '%s\n' "$*" > "$CMD"
cat > "$DIR/$ID.run.sh" <<INNER
#!/usr/bin/env bash
cd "$(cd "$DIR/.." && pwd)"
echo "RUNNING \$(date -u +%FT%TZ) pid \$\$" > "$ST"
$*
code=\$?
if [ \$code -eq 0 ]; then echo "DONE:\$code \$(date -u +%FT%TZ)" > "$ST"; else echo "FAILED:\$code \$(date -u +%FT%TZ)" > "$ST"; fi
INNER
chmod +x "$DIR/$ID.run.sh"
screen -dmS "ivc_$ID" bash -c "'$DIR/$ID.run.sh' > '$LOG' 2>&1"
sleep 1
echo "launched $ID  (screen: ivc_$ID)  log: runs/$ID.log"
