#!/usr/bin/env bash
# Append (unix time, runner pid, epoch, total) so per-epoch time can be measured rather than
# extrapolated from epoch 1, which carries the one-off data load and graph build.
out="${1:-runs/.epoch_probe.tsv}"
now=$(date +%s)
for f in logs/runners/*.log; do
  [ -e "$f" ] || continue
  pid=$(basename "$f" | grep -oP '_\K\d+(?=\.log)')
  last=$(grep -aoE "\[train\] epoch=[0-9]+/[0-9]+" "$f" | tail -1)
  [ -z "$last" ] && continue
  e=${last#*epoch=}; n=${e%/*}; t=${e#*/}
  printf '%s\t%s\t%s\t%s\n' "$now" "$pid" "$n" "$t" >> "$out"
done
