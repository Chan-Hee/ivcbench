#!/usr/bin/env bash
# Watch a pid this dispatcher did not launch and write its .status when it exits, so
# dependent queue entries unblock. Exit code is unknowable for a non-child, so success is
# inferred from a sentinel path the job is expected to produce.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
id="$1"; pid="$2"; sentinel="${3:-}"
tail --pid="$pid" -f /dev/null
if [ -z "$sentinel" ] || [ -e "$sentinel" ]; then
  echo "DONE:0 $(date -u +%FT%TZ) adopted pid=$pid" > "$DIR/$id.status"
else
  echo "FAILED:1 $(date -u +%FT%TZ) adopted pid=$pid; sentinel missing: $sentinel" > "$DIR/$id.status"
fi
