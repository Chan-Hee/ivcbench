#!/usr/bin/env python3
"""Figure 1. Method categories and comparator framework — a clean schematic (no data).
(a) six transfer-assumption categories, (b) four executable benchmark roles, (c) the decision-rule flow.
Minimal, consistent line iconography in an equal-aspect pixel space; restrained palette, generous whitespace."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon, Rectangle, Arc
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "results" / "_paper" / "figure_method_framework.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

W, H = 1360, 916
fig = plt.figure(figsize=(W / 100, H / 100), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
INK = "#222831"; GREY = "#5B6670"; FAINT = "#9AA4AE"; PANELB = "#E2E7EC"; LW = 2.3

def mix(hexc, f):  # blend colour toward white by fraction f
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(c * (1 - f) + f)
# restrained category palette
COL = {"found": "#3D6FB4", "lat": "#3C9A6B", "graph": "#7E5BB0", "ot": "#DA8038", "hyb": "#2E9A9A", "chem": "#C39126"}

def rrect(x, y, w, h, fc, ec, lw=1.3, r=12, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z, joinstyle="round"))

def badge(cx, cy, n, color, rad=14.5):
    ax.add_patch(Circle((cx, cy), rad, fc=color, ec="none", zorder=8))
    ax.text(cx, cy, str(n), ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=9)

def L(p, c, lw=LW, z=6, ls="solid"):
    ax.plot([q[0] for q in p], [q[1] for q in p], color=c, lw=lw, zorder=z, ls=ls,
            solid_capstyle="round", solid_joinstyle="round")

def disc(cx, cy, r, c, z=6):
    ax.add_patch(Circle((cx, cy), r, fc=c, ec="none", zorder=z))

def ring(cx, cy, r, c, lw=LW, z=6):
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=c, lw=lw, zorder=z))

def arrow(p0, p1, c, lw=LW, z=7, ls="solid", rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, ls=ls, zorder=z,
                                 capstyle="round", joinstyle="round"))

# ---------------- icons: minimal, single-weight, in a ~54-unit box ----------------
def i_found(x, y, c):
    for dx, dy in [(-7, -6), (-13, -2), (-9, 2), (-4, -1), (-11, -8)]:
        disc(x + dx - 6, y + dy, 2.6, c)
    cc = mix(c, 0.45)
    for dx, dy in [(7, 5), (12, 9), (9, 2), (14, 4), (6, 10)]:
        disc(x + dx + 4, y + dy, 2.6, cc)

def i_lat(x, y, c):
    ax.add_patch(Arc((x, y), 50, 34, angle=12, theta1=205, theta2=120, ec=c, lw=LW, ls=(0, (5, 4)), zorder=6))
    arrow((x + 16, y - 9), (x + 23, y + 1), c, lw=LW - 0.2, ms=10)
    for dx, dy in [(-17, 4), (2, 11)]:
        disc(x + dx, y + dy, 3.4, c)

def i_graph(x, y, c):
    P = [(-20, -3), (-3, 13), (16, 7), (18, -12), (-4, -15)]
    P = [(x + a, y + b) for a, b in P]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]:
        L([P[a], P[b]], mix(c, 0.5), lw=1.6)
    for p in P:
        disc(p[0], p[1], 4.4, c)

def i_ot(x, y, c):
    disc(x - 16, y, 9, mix(COL["ot"], 0.6))
    disc(x + 16, y, 9, mix("#3D6FB4", 0.6))
    arrow((x - 5, y), (x + 5, y), GREY, lw=LW - 0.2, ms=10)

def i_hyb(x, y, c):
    for dx, lab in [(-15, "z"), (15, "T")]:
        rrect(x + dx - 11, y - 11, 22, 22, "white", c, lw=1.8, r=5, z=6)
        ax.text(x + dx, y, lab, ha="center", va="center", fontsize=11, fontstyle="italic", color=c, zorder=7)
    arrow((x - 4, y), (x + 4, y), INK, lw=LW - 0.2, ms=10)

def i_chem(x, y, c):
    r = 15; a = np.deg2rad(np.arange(6) * 60 + 30)
    hx, hy = x - 4 + r * np.cos(a), y + r * np.sin(a)
    ax.add_patch(Polygon(np.column_stack([hx, hy]), closed=True, fill=False, ec=c, lw=LW, joinstyle="round", zorder=6))
    ax.add_patch(Circle((x - 4, y), 6.5, fill=False, ec=mix(c, 0.4), lw=1.4, zorder=6))
    L([(hx[0], hy[0]), (hx[0] + 15, hy[0] + 6)], c)

def i_net(x, y, c):
    pts = [(x + dx, y + dy) for dy in (15, 0, -15) for dx in (-15, 0, 15)]
    for i, p in enumerate(pts):
        for qp in pts[i + 1:]:
            if abs(p[0] - qp[0]) + abs(p[1] - qp[1]) <= 16:
                L([p, qp], mix(c, 0.5), lw=1.5)
    for p in pts:
        disc(p[0], p[1], 3.8, c)

def i_target(x, y, c):
    ring(x, y, 18, c, lw=LW); ring(x, y, 10.5, mix(c, 0.35), lw=LW); disc(x, y, 3.4, c)

def i_scales(x, y, c):
    L([(x, y - 16), (x, y + 15)], c); L([(x - 21, y + 15), (x + 21, y + 15)], c)
    L([(x - 12, y - 16), (x + 12, y - 16)], c)
    for sx in (-21, 21):
        L([(x + sx, y + 15), (x + sx, y + 4)], c, lw=1.6)
        ax.add_patch(Arc((x + sx, y + 4), 18, 12, theta1=200, theta2=340, ec=c, lw=LW, zorder=6))

def i_check(x, y, c):
    ring(x, y, 17, c, lw=LW)
    L([(x - 8, y), (x - 2, y - 7), (x + 9, y + 8)], c, lw=LW + 0.4)

def i_people(x, y, c, s=1.0):
    for dx in (-16 * s, 0, 16 * s):
        disc(x + dx, y + 6 * s, 4.6 * s, c)
        ax.add_patch(FancyBboxPatch((x + dx - 6 * s, y - 9 * s), 12 * s, 11 * s, boxstyle="round,pad=0,rounding_size=3.5", fc=c, ec="none", zorder=6))

def i_bars(x, y, c):
    for dx, h in [(-11, 11), (0, 20), (11, 15)]:
        ax.add_patch(FancyBboxPatch((x + dx - 4, y - 13), 8, h, boxstyle="round,pad=0,rounding_size=1.6", fc=c, ec="none", zorder=6))

def i_line(x, y, c):
    L([(x - 15, y + 12), (x - 15, y - 13), (x + 16, y - 13)], FAINT, lw=1.5)
    L([(x - 12, y - 8), (x + 13, y + 10)], c); disc(x - 12, y - 8, 2.6, c); disc(x + 13, y + 10, 2.6, c)

def i_square(x, y, c):
    ax.add_patch(FancyBboxPatch((x - 10, y - 10), 20, 20, boxstyle="round,pad=0,rounding_size=3.5", fc=c, ec="none", zorder=6))

# ===================== title =====================
ax.text(W / 2, H - 28, "Method categories and comparator framework", ha="center", va="center",
        fontsize=21, fontweight="bold", color=INK)
GX = [30, 485, 915]; CW = [428, 402, 415]; TOP = H - 72

def column(x, w, letter, title):
    rrect(x, 26, w, TOP - 26, "white", PANELB, lw=1.3, r=18, z=1)
    ax.text(x + 24, TOP - 30, letter, ha="left", va="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(x + w / 2 + 14, TOP - 30, title, ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)

# ===================== (a) Transfer assumptions =====================
column(GX[0], CW[0], "a", "Transfer assumptions")
cats = [(1, "found", "Foundation models", "Reusable cell-state representation", "scGPT, scFoundation, scBERT", i_found),
        (2, "lat", "Latent / compositional\nmodels", "Latent perturbation effect", "scGen, CPA", i_lat),
        (3, "graph", "Prior-graph models", "Regulatory wiring", "GEARS, AttentionPert", i_graph),
        (4, "ot", "Optimal-transport models", "Distributional intervention", "CellOT, scPRAM", i_ot),
        (5, "hyb", "Hybrid models", "Representation +\ntransition operator", "STATE, PertAdapt", i_hyb),
        (6, "chem", "Chemistry-aware models", "Compound-structure signal", "chemCPA, FP-ridge", i_chem)]
ax0 = GX[0] + 18; aw = CW[0] - 36; ch = 116; gap = 12; ytop = TOP - 66
for i, (n, k, title, desc, ex, icon) in enumerate(cats):
    c = COL[k]; yt = ytop - i * (ch + gap); yb = yt - ch
    rrect(ax0, yb, aw, ch, mix(c, 0.93), mix(c, 0.35), lw=1.2, r=12, z=2)
    badge(ax0 + 30, yt - 26, n, c)
    icon(ax0 + 92, yt - ch / 2, c)
    tx = ax0 + 140
    nl = title.count("\n") + 1
    ax.text(tx, yt - (24 if nl == 1 else 19), title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=c, zorder=7, linespacing=1.06)
    ax.text(tx, yt - (24 if nl == 1 else 19) - nl * 16 - 7, desc, ha="left", va="top", fontsize=10, color=INK, zorder=7, linespacing=1.12)
    ax.text(tx, yb + 16, ex, ha="left", va="center", fontsize=9.4, color=GREY, zorder=7)

# ===================== (b) Benchmark roles =====================
column(GX[1], CW[1], "b", "Benchmark roles")
bx0 = GX[1] + 18; bw = CW[1] - 36
roles = [(1, "found", i_net, "Conditioned predictors", "General predictive models\nevaluated on held-out tasks", None),
         (2, "lat", i_target, "Task-specific\ndeterministic predictors", "FP-ridge, linear-shift-KOemb", None),
         (3, "graph", i_scales, "Distributional comparator", "CINEMA-OT", "reference only; never counted\nas a conditioned win")]
ybt = TOP - 66; rh = 156; rgap = 15
for i, (n, k, icon, title, desc, ital) in enumerate(roles):
    c = COL[k]; yt = ybt - i * (rh + rgap); yb = yt - rh
    rrect(bx0, yb, bw, rh, mix(c, 0.93), mix(c, 0.35), lw=1.2, r=12, z=2)
    badge(bx0 + 30, yt - 26, n, c)
    icon(bx0 + 84, yt - rh / 2, c)
    tx = bx0 + 128
    nl = title.count("\n") + 1
    ax.text(tx, yt - 26, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=c, zorder=7, linespacing=1.06)
    ax.text(tx, yt - 26 - nl * 16 - 8, desc, ha="left", va="top", fontsize=10, color=INK, zorder=7, linespacing=1.14)
    if ital:
        ax.text(tx, yb + 16, ital, ha="left", va="bottom", fontsize=9.1, fontstyle="italic", color=GREY, zorder=7, linespacing=1.14)

# Baselines card (4)
c = COL["chem"]; yt = ybt - 3 * (rh + rgap); base_h = 214; yb = yt - base_h
rrect(bx0, yb, bw, base_h, mix(c, 0.93), mix(c, 0.35), lw=1.2, r=12, z=2)
badge(bx0 + 30, yt - 25, 4, c)
ax.text(bx0 + bw / 2 + 14, yt - 25, "Baselines", ha="center", va="center", fontsize=12, fontweight="bold", color=c, zorder=7)
sub_y = yb + 16; sub_h = base_h - 62; sw = (bw - 40) / 2
for j, (lab, items, ic) in enumerate([("Universal floor", ["cell-mean shift", "linear-PCA shift"], i_square),
                                       ("Context baselines", ["control-as-prediction", "donor shift", "training-mean shift"], i_people)]):
    sx = bx0 + 14 + j * (sw + 12)
    rrect(sx, sub_y, sw, sub_h, "white", mix(c, 0.45), lw=1.0, r=9, z=3)
    ax.text(sx + sw / 2, sub_y + sub_h - 17, lab, ha="center", va="center", fontsize=10, fontweight="bold", color=mix(c, 0.15), zorder=7)
    ic(sx + sw / 2, sub_y + sub_h - 47, FAINT)
    for kk, it in enumerate(items):
        ax.text(sx + 15, sub_y + sub_h - 80 - kk * 19, it, ha="left", va="center", fontsize=9.2, color=INK, zorder=7)

# ===================== (c) Decision rule =====================
column(GX[2], CW[2], "c", "Decision rule")
cx0 = GX[2] + CW[2] / 2; cw = CW[2] - 48

def cbox(yc, h, lines, fc, ec, fs, r=12, ls="solid", icon=None, ic_c=None):
    rrect(cx0 - cw / 2, yc - h / 2, cw, h, fc, ec, lw=1.4, r=r, ls=ls, z=3)
    if icon:
        icon(cx0 - cw / 2 + 36, yc, ic_c)
        xt = cx0 - cw / 2 + 64; al = "left"
    else:
        xt = cx0; al = "center"
    if isinstance(lines, str):
        ax.text(xt, yc, lines, ha=al, va="center", fontsize=fs[0], fontweight="bold", color=fs[1], zorder=7)
    else:
        ax.text(xt, yc + lines[0][2], lines[0][0], ha=al, va="center", fontsize=lines[0][1][0], fontweight="bold", color=lines[0][1][1], zorder=7)
        ax.text(xt, yc + lines[1][2], lines[1][0], ha=al, va="center", fontsize=lines[1][1][0], color=lines[1][1][1], zorder=7)

def varrow(y1, y2):
    arrow((cx0, y1), (cx0, y2), INK, lw=2.0, ms=15)

y = TOP - 98
cbox(y, 56, "Conditioned prediction", mix(COL["found"], 0.93), mix(COL["found"], 0.3), (12, COL["found"]))
varrow(y - 28, y - 70)
cmp_h = 184; ccy = y - 70 - cmp_h / 2
rrect(cx0 - cw / 2, ccy - cmp_h / 2, cw, cmp_h, "#FBFCFD", FAINT, lw=1.3, r=12, ls=(0, (5, 4)), z=3)
ax.text(cx0, ccy + cmp_h / 2 - 21, "Compare against", ha="center", va="center", fontsize=11, color=INK, zorder=7)
for row, (lab, ic) in enumerate([("cell-mean shift", i_bars), ("linear-PCA shift", i_line)]):
    ry = ccy + 31 - row * 70
    rrect(cx0 - cw / 2 + 18, ry - 26, cw - 36, 52, "white", "#CBD3DB", lw=1.0, r=9, z=4)
    ic(cx0 - cw / 2 + 50, ry, GREY); ax.text(cx0 - cw / 2 + 86, ry, lab, ha="left", va="center", fontsize=10.5, color=INK, zorder=7)
    if row == 0:
        ax.text(cx0, ccy - 4, "AND", ha="center", va="center", fontsize=9.5, fontweight="bold", color=GREY, zorder=7)
varrow(ccy - cmp_h / 2, ccy - cmp_h / 2 - 44)
gy = ccy - cmp_h / 2 - 44 - 41
cbox(gy, 82, [("Counted as working", (11.5, COL["lat"]), 13), ("only if it beats both floor members", (9.7, INK), -13)],
     mix(COL["lat"], 0.92), COL["lat"], None, r=12, icon=i_check, ic_c=COL["lat"])
ny = gy - 41 - 26 - 33
rrect(cx0 - cw / 2, ny - 33, cw, 66, "#FAFBFC", "#B2BBC4", lw=1.1, r=11, ls=(0, (4, 3)), z=3)
i_people(cx0 - cw / 2 + 36, ny, FAINT, 0.92)
ax.text(cx0 - cw / 2 + 66, ny, "Context baselines and task-specific\ncomparators are reported for interpretation",
        ha="left", va="center", fontsize=9.2, color=GREY, zorder=7, linespacing=1.18)
dvy = ny - 33 - 24
L([(cx0 - cw / 2, dvy), (cx0 + cw / 2, dvy)], COL["graph"], lw=1.2, ls=(0, (5, 4)), z=3)
py = dvy - 24 - 35
rrect(cx0 - cw / 2, py - 35, cw, 70, mix(COL["graph"], 0.92), COL["graph"], lw=1.4, r=12, z=3)
i_scales(cx0 - cw / 2 + 38, py, COL["graph"])
ax.text(cx0 - cw / 2 + 70, py + 11, "CINEMA-OT", ha="left", va="center", fontsize=11.5, fontweight="bold", color=COL["graph"], zorder=7)
ax.text(cx0 - cw / 2 + 70, py - 12, "distributional reference only", ha="left", va="center", fontsize=9.5, color=INK, zorder=7)
arrow((cx0 + cw / 2 - 8, py + 28), (cx0 + cw / 2 - 8, ccy), COL["graph"], lw=1.3, ls=(0, (4, 3)), rad=-0.34, ms=11)

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
plt.close(fig)
print("wrote", OUT, Image.open(OUT).size)
