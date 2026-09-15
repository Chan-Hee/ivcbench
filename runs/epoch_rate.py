#!/usr/bin/env python
"""Per-epoch cost from the probe log, measured between the FIRST sighting of each epoch.

Taking the last sighting instead makes the estimate drift upward while an epoch is still running:
the numerator keeps growing as you sample, the denominator stays at one epoch, and the reported
rate rose 9 -> 10 -> 15 minutes on a job whose real rate had not changed.
"""
import collections
import sys
import time

rows = [l.split("\t") for l in open("runs/.epoch_probe.tsv").read().splitlines() if l]
first = collections.defaultdict(dict)      # pid -> {epoch: earliest unix time seen}
total = {}
for t, pid, e, tot in rows:
    t, e = int(t), int(e)
    d = first[pid]
    if e not in d or t < d[e]:
        d[e] = t
    total[pid] = int(tot)

now = time.time()
out = []
for pid, d in sorted(first.items()):
    eps = sorted(d)
    if len(eps) < 2:
        continue
    # average over every consecutive pair we actually observed
    deltas = [(d[b] - d[a]) / (b - a) for a, b in zip(eps, eps[1:])]
    per = sum(deltas) / len(deltas)
    last = eps[-1]
    # time already spent inside the current epoch counts against the remainder
    elapsed_in = max(0.0, now - d[last])
    rem = max(0.0, (total[pid] - last) * per - elapsed_in)
    out.append((pid, last, total[pid], per / 60, rem / 3600, len(deltas)))

if not out:
    print("EPOCH-RATE 아직 구간 없음")
    sys.exit(0)
print("EPOCH-RATE " + " | ".join(
    f"{p}: {e}/{t} · {pm:.0f}분/epoch · 남은 {r:.1f}h ({n}구간)" for p, e, t, pm, r, n in out))
print(f"  최장 잔여 {max(o[4] for o in out):.1f}h → {time.strftime('%H:%M', time.localtime(now + max(o[4] for o in out) * 3600))}")
