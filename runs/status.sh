#!/usr/bin/env bash
# One line per job: id, status, log size, last line.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf "%-22s %-26s %8s  %s\n" JOB STATUS LOG LAST
for st in "$DIR"/*.status; do
  [ -e "$st" ] || continue
  id=$(basename "$st" .status)
  printf "%-22s %-26s %8s  %s\n" "$id" "$(cat "$st" | cut -c1-26)" \
    "$( [ -f "$DIR/$id.log" ] && wc -c < "$DIR/$id.log" || echo 0 )" \
    "$( [ -f "$DIR/$id.log" ] && tail -1 "$DIR/$id.log" | cut -c1-70 )"
done
echo; echo "live screens:"; screen -ls 2>/dev/null | grep ivc_ || echo "  none"
