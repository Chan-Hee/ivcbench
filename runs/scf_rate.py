#!/usr/bin/env python
"""Remaining time for the scFoundation units, counted the way the CellOT estimate was.

GEARS prints "Epoch N Step M" as it goes. Steps per epoch is fixed (graphs / batch), so once two
sightings exist the step rate gives the remainder directly. Measured between FIRST sightings, so
the estimate does not drift upward while a step count is still standing still.
"""
import glob, os, re, time, json, sys

STATE = "runs/.scf_rate.json"
EPOCHS = int(os.environ.get("IVCBENCH_SCF_GENE_EPOCHS", "15"))

prev = {}
if os.path.exists(STATE):
    try: prev = json.load(open(STATE))
    except Exception: prev = {}

now, cur, out = time.time(), {}, []
for f in glob.glob("logs/runners/scFoundation*.log"):
    pid = re.search(r"_(\d+)\.log$", f).group(1)
    if not os.path.exists(f"/proc/{pid}"):
        continue                    # finished unit: its log stops changing and would print 100% forever
    txt = open(f, errors="replace").read()
    hits = re.findall(r"Epoch (\d+) Step (\d+)", txt)
    if not hits:
        continue
    ep, st = int(hits[-1][0]), int(hits[-1][1])
    # steps per epoch = the largest step seen in any completed epoch, else unknown yet
    per_ep = max((int(s) for e, s in hits if int(e) < ep), default=0)
    done = (ep - 1) * per_ep + st if per_ep else st
    was = prev.get(pid)
    cur[pid] = {"done": done, "t": was["t"] if was and was["done"] == done else now,
                "d0": was["d0"] if was else done, "t0": was["t0"] if was else now}
    if was and done > was["d0"]:
        rate = (done - was["d0"]) / max(1.0, now - was["t0"])      # steps per second
        total = EPOCHS * per_ep if per_ep else None
        if total and rate > 0:
            rem = max(0.0, (total - done) / rate)
            out.append(f"{pid}: epoch {ep}/{EPOCHS} step {done:,}/{total:,} "
                       f"({100*done/total:.0f}%) · 남은 {rem/3600:.1f}h")
        else:
            out.append(f"{pid}: epoch {ep}/{EPOCHS} step {st:,} · {rate*60:.0f} step/분 (epoch당 스텝 미확정)")
json.dump(cur, open(STATE, "w"))
if out:
    print("SCF-RATE " + " | ".join(out))
    fin = [float(re.search(r"남은 ([\d.]+)h", o).group(1)) for o in out if "남은" in o]
    if fin:
        print(f"  최장 {max(fin):.1f}h → {time.strftime('%H:%M', time.localtime(now + max(fin)*3600))}")
else:
    print("SCF-RATE 아직 구간 없음")
