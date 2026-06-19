#!/usr/bin/env python3
"""make_figure2_method_comparator_framework.py
Publication-ready vector schematic: "Method categories and comparator framework"
(manuscript Figure 1). Drawn programmatically with matplotlib vector primitives only —
no external icon images. Physical size 180 x 120 mm; coordinates are in millimetres so
font sizes are true print points. Exports editable-text SVG + PDF and a 600-dpi PNG."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon, Arc, Rectangle, Wedge, Ellipse

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "none",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
OUTDIR = Path(__file__).resolve().parents[1] / "results" / "_paper"
OUTDIR.mkdir(parents=True, exist_ok=True)
STEM = "figure2_method_comparator_framework"

# ---- palette (exact) ----
BLUE, GREEN, PURPLE, ORANGE, TEAL, GOLD = "#2B6CB0", "#2F855A", "#6B46C1", "#DD6B20", "#319795", "#B7791F"
GREYTX, OUTLINE, PANELBG, INK = "#4A5568", "#CBD5E0", "#F8FAFC", "#1A202C"
def tint(hexc, f=0.90):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c * (1 - f) + f)
def mid(hexc, f=0.45):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c * (1 - f) + f)

# ---- font points ----
F_TITLE, F_PANEL, F_LET, F_BOX, F_BODY, F_EX, F_NOTE = 15, 10.5, 11, 8.6, 7.4, 7.0, 6.6
ILW = 0.85  # icon line weight (pt)

W, H = 190.0, 122.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

def rrect(x, y, w, h, fc, ec, lw=0.7, r=1.6, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z, joinstyle="round"))
def badge(cx, cy, n, c, rad=1.9):
    ax.add_patch(Circle((cx, cy), rad, fc=c, ec="none", zorder=8))
    ax.text(cx, cy, str(n), ha="center", va="center", fontsize=F_BOX - 1.4, fontweight="bold", color="white", zorder=9)
def line(p, c, lw=ILW, z=6, ls="solid"):
    ax.plot([q[0] for q in p], [q[1] for q in p], color=c, lw=lw, zorder=z, ls=ls,
            solid_capstyle="round", solid_joinstyle="round")
def disc(x, y, r, c, z=6): ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=z))
def ring(x, y, r, c, lw=ILW, z=6): ax.add_patch(Circle((x, y), r, fill=False, ec=c, lw=lw, zorder=z))
def arr(p0, p1, c, lw=ILW, z=7, ls="solid", rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, ls=ls, zorder=z, capstyle="round"))

# ---------------- icons (millimetre boxes ~7 mm) ----------------
def ic_found(x, y, c):
    axc = "#D5DBE1"
    x0, y0 = x - 2.95, y - 2.95
    line([(x0, y + 2.95), (x0, y0), (x + 2.95, y0)], axc, 0.9, z=4)
    rng = np.random.default_rng(1)
    light = mid(c, 0.5)
    def cloud(cx, cy, col):
        for _ in range(18):
            a = rng.uniform(0, 2 * np.pi)
            rr = 1.0 * np.sqrt(rng.uniform(0, 1))
            disc(cx + rr * np.cos(a), cy + rr * np.sin(a), 0.31, col, z=6)
    cloud(x - 1.3, y - 1.2, c)
    cloud(x + 1.4, y + 1.4, light)
def ic_lat(x, y, c):
    # latent-space boundary: dashed light-green ring
    lc = mid(c, 0.5)
    ax.add_patch(Circle((x, y), 3.0, fill=False, ec=lc, lw=0.95, ls=(0, (3, 2.2)), zorder=4))
    # 3 dots along a gentle trajectory (relative to center)
    pts = [(-1.95, -1.05), (-0.35, 0.95), (1.95, -0.35)]
    P = [(x + dx, y + dy) for dx, dy in pts]
    def shorten(a, b, ra, rb):
        dx_, dy_ = b[0] - a[0], b[1] - a[1]
        L = np.hypot(dx_, dy_); ux, uy = dx_ / L, dy_ / L
        return (a[0] + ux * ra, a[1] + uy * ra), (b[0] - ux * rb, b[1] - uy * rb)
    for a, b in [(P[0], P[1]), (P[1], P[2])]:
        s, e = shorten(a, b, 0.64, 0.82)
        arr(s, e, c, lw=1.0, z=6, rad=-0.13, ms=8)
    for p in P:
        disc(p[0], p[1], 0.5, c, z=7)
def ic_graph(x, y, c):
    ec = mid(c, 0.5)
    # 6 nodes at irregular positions inside ~6 mm (within +/-3.3 of center)
    P = [(-2.5, 1.3), (0.4, 2.6), (2.6, 0.9), (1.9, -2.0), (-1.0, -2.5), (-2.3, -0.4)]
    P = [(x + px, y + py) for px, py in P]
    # ~4 solid + ~2 dashed; node 1 (top) is a degree-3 hub = regulator
    solid = [(0, 1), (1, 2), (2, 3), (1, 5)]
    dashed = [(3, 4), (4, 5)]
    for i, j in solid:
        line([P[i], P[j]], ec, lw=0.9, z=5)
    for i, j in dashed:
        line([P[i], P[j]], ec, lw=0.9, z=5, ls=(0, (2.0, 1.6)))
    for px, py in P:
        ax.add_patch(Circle((px, py), 0.6, fc=c, ec="white", lw=0.7, zorder=7))
def ic_ot(x, y, c):
    cb = "#2B6CB0"
    R = 1.55
    dx = 2.0
    rdot = 0.28
    yc = y - 0.18  # center the composite bbox (arrow rides above the blobs) on (x,y)
    # two pale population blobs
    ax.add_patch(Circle((x-dx, yc), R, fc=mid(c,0.78),  ec=mid(c,0.40),  lw=0.7, zorder=4))
    ax.add_patch(Circle((x+dx, yc), R, fc=mid(cb,0.78), ec=mid(cb,0.40), lw=0.7, zorder=4))
    # cells inside each blob
    loff = [(-0.5,0.45),(0.45,0.55),(0.65,-0.30),(-0.45,-0.50),(0.05,-0.05),(-0.05,0.85)]
    roff = [( 0.5,0.45),(-0.45,0.50),(-0.60,-0.30),(0.45,-0.50),(0.00,0.00),(0.10,0.85)]
    for ox,oy in loff: disc(x-dx+ox, yc+oy, rdot, c,  z=6)
    for ox,oy in roff: disc(x+dx+ox, yc+oy, rdot, cb, z=6)
    # curved dashed transport arrow arcing over the top
    arr((x-1.6, yc+1.2), (x+1.6, yc+1.2), "#7A8794", lw=0.95, ls=(0,(3,2)), rad=-0.45, ms=8, z=8)
def ic_hyb(x, y, c):
    # geometry in mm, centered at (x,y), within +/-3.6mm
    s = 2.7                      # box side
    h = s / 2.0                  # half side
    cxL, cxR = x - 2.0, x + 2.0  # box centers
    # left box (italic z) and right box (italic T)
    for cx, lab in ((cxL, "z"), (cxR, "T")):
        ax.add_patch(FancyBboxPatch((cx - h, y - h), s, s,
                     boxstyle="round,pad=0,rounding_size=0.55",
                     fc="white", ec=c, lw=0.9, zorder=6))
        ax.text(cx, y - 0.02, lab, ha="center", va="center",
                fontsize=13, color=c, fontstyle="italic", zorder=7)
    # arrow from right edge of z box to left edge of T box (even clearance both ends)
    x0 = cxL + h + 0.12
    x1 = cxR - h - 0.12
    arr((x0, y), (x1, y), c, lw=1.0, ms=9)
def ic_chem(x, y, c):
    lw = 0.85
    dy = -0.30                      # nudge composition to balance the high O label
    R = 1.55                        # hexagon "radius" (vertex distance)
    cx, cy = x - 0.55, y + dy       # ring center, shifted left to leave room on the right

    # --- benzene ring: regular hexagon, pointy left/right ---
    angs = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = [(cx + R*np.cos(a), cy + R*np.sin(a)) for a in angs]
    ax.add_patch(Polygon(verts, closed=True, fill=False, ec=c, lw=lw, zorder=6,
                         joinstyle="round"))
    ring(cx, cy, R*0.58, c, lw=lw*0.85)             # inner aromatic circle

    Vr = verts[0]                                    # right vertex (0 deg)
    Vl = verts[3]                                    # far-left vertex (180 deg)

    # --- left: methylamino  ring-N(-CH3) ---
    N = (Vl[0] - 0.62, Vl[1] + 0.70)                 # N atom (up-left of ring)
    Me = (N[0] - 0.62, N[1] - 0.70)                  # methyl terminal (down-left)
    line([Vl, (N[0] + 0.10, N[1] - 0.12)], c, lw)    # bond ring->N (stops short of glyph)
    line([(N[0] - 0.08, N[1] - 0.10), Me], c, lw)    # N->methyl stub
    ax.text(N[0], N[1] + 0.02, "N", ha="center", va="center",
            fontsize=5, color=c, zorder=8)

    # --- right: benzamide  ring-C(=O)-NH2 ---
    C = (Vr[0] + 0.72, Vr[1] + 0.66)                 # carbonyl carbon (up-right)
    line([Vr, C], c, lw)                             # ring -> carbonyl
    O = (C[0], C[1] + 1.05)                          # carbonyl O above
    line([(C[0] - 0.11, C[1] + 0.12), (C[0] - 0.11, O[1] - 0.30)], c, lw)  # C=O line 1
    line([(C[0] + 0.11, C[1] + 0.12), (C[0] + 0.11, O[1] - 0.30)], c, lw)  # C=O line 2
    ax.text(O[0], O[1] + 0.02, "O", ha="center", va="center",
            fontsize=5, color=c, zorder=8)
    Nh = (C[0] + 0.72, C[1] - 0.62)                  # amide N (down-right)
    line([C, (Nh[0] - 0.18, Nh[1] + 0.14)], c, lw)   # C -> NH2 (stops short of glyph)
    ax.text(Nh[0] + 0.04, Nh[1], "NH2", ha="left", va="center",
            fontsize=5, color=c, zorder=8)
def ic_net(x, y, c):
    s = 2.0            # node spacing (mm)
    r = 0.5            # node radius (mm)
    ec = mid(c, 0.4)   # edge colour: c blended 40% toward white
    coords = [-s, 0.0, s]
    pos = {(i, j): (x + coords[i], y + coords[j]) for i in range(3) for j in range(3)}
    # edges: horizontal, vertical AND diagonal neighbours (dense lattice)
    seen = set()
    for (i, j), (px, py) in pos.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    key = frozenset(((i, j), (ni, nj)))
                    if key in seen:
                        continue
                    seen.add(key)
                    line([(px, py), pos[(ni, nj)]], ec, lw=0.7, z=5)
    # nodes: filled colour-c discs with a thin white edge, drawn on top
    for (px, py) in pos.values():
        ax.add_patch(Circle((px, py), r, fc=c, ec="white", lw=0.8, zorder=7))
def ic_target(x, y, c):
    cl = mid(c, 0.34)            # slightly lighter tone for the outer rings
    # bullseye: two outline rings + filled center dot
    ring(x, y, 2.5, cl, lw=0.95)
    ring(x, y, 1.5, cl, lw=0.95)
    disc(x, y, 0.5, c)
    # dart shaft + head, striking from upper-right toward the center
    T = (x + 2.5, y + 2.5)       # tail (upper right)
    H = (x + 0.27, y + 0.27)     # head tip, just at the center dot
    arr(T, H, c, lw=1.1, ms=9)
    # tiny V fletching at the tail (vertex seated onto the visible shaft)
    V0 = (T[0] - 0.13*np.cos(np.deg2rad(45)), T[1] - 0.13*np.sin(np.deg2rad(45)))
    L = 0.78
    a = np.deg2rad(45)
    da = np.deg2rad(33)
    f1 = (V0[0] + L*np.cos(a + da), V0[1] + L*np.sin(a + da))
    f2 = (V0[0] + L*np.cos(a - da), V0[1] + L*np.sin(a - da))
    line([f1, V0, f2], c, lw=1.0)
def ic_scales(x, y, c):
    lw = 0.9
    oy = 0.1             # nudge up to vertically centre the content
    bx = 2.35            # beam half-width  -> ~4.7 mm beam
    top = y + 2.45 + oy  # beam height
    base_y = y - 2.35 + oy
    # central post
    line([(x, top), (x, base_y)], c, lw)
    # horizontal beam
    line([(x - bx, top), (x + bx, top)], c, lw)
    # pivot finial at top centre
    disc(x, top, 0.24, c)
    # trapezoid foot (wider at bottom) on a thin base plate
    ax.add_patch(Polygon([(x - 0.5, base_y), (x + 0.5, base_y),
                          (x + 0.95, base_y - 0.5), (x - 0.95, base_y - 0.5)],
                         closed=True, fill=True, fc=c, ec=c, lw=lw))
    line([(x - 1.25, base_y - 0.5), (x + 1.25, base_y - 0.5)], c, lw)
    # two pans: narrow triangle of cords down to a shallow wide bowl
    rim_y = y + 0.95 + oy
    hw = 1.0             # bowl half-width
    for s in (-1, 1):
        ax_ = x + s * bx
        line([(ax_, top), (ax_ - hw, rim_y)], c, lw)
        line([(ax_, top), (ax_ + hw, rim_y)], c, lw)
        disc(ax_, top, 0.15, c)
        ax.add_patch(Arc((ax_, rim_y), 2 * hw, 0.85, angle=0,
                         theta1=180, theta2=360, ec=c, lw=lw))
def ic_trophy(x, y, c):
    yo = 0.1                      # small upward shift to balance the base
    cl = mid(c, 0.5)             # lighter shade for the cup opening
    ec = c
    # --- handles (open arcs); centers tucked inward so inner ends hide behind bowl ---
    for sgn in (-1, 1):
        ax.add_patch(Arc((x + sgn*1.60, y + 1.50 + yo), 1.25, 1.65, angle=0,
                         theta1=(90 if sgn < 0 else -90),
                         theta2=(270 if sgn < 0 else 90),
                         ec=ec, lw=1.0, zorder=5))
    # --- cup bowl (tapers to meet the knob smoothly) ---
    bowl = [(x-2.00, y+2.40+yo), (x-1.85, y+1.50+yo), (x-1.50, y+0.60+yo),
            (x-0.90, y+0.00+yo), (x-0.50, y-0.55+yo),
            (x+0.50, y-0.55+yo), (x+0.90, y+0.00+yo),
            (x+1.50, y+0.60+yo), (x+1.85, y+1.50+yo), (x+2.00, y+2.40+yo)]
    ax.add_patch(Polygon(bowl, closed=True, fill=True, fc=c, ec=ec, lw=0.95, zorder=6))
    # rim opening
    ax.add_patch(Ellipse((x, y+2.40+yo), 4.0, 0.7, fc=cl, ec=ec, lw=0.95, zorder=7))
    # --- knob + stem ---
    ax.add_patch(Ellipse((x, y-0.55+yo), 1.00, 0.42, fc=c, ec=ec, lw=0.95, zorder=6))
    ax.add_patch(Polygon([(x-0.30,y-0.70+yo),(x+0.30,y-0.70+yo),
                          (x+0.42,y-1.65+yo),(x-0.42,y-1.65+yo)],
                 closed=True, fill=True, fc=c, ec=ec, lw=0.95, zorder=5))
    # --- base (trapezoid + rounded plinth) ---
    ax.add_patch(Polygon([(x-0.85,y-1.65+yo),(x+0.85,y-1.65+yo),
                          (x+1.40,y-2.45+yo),(x-1.40,y-2.45+yo)],
                 closed=True, fill=True, fc=c, ec=ec, lw=0.95, zorder=5))
    ax.add_patch(FancyBboxPatch((x-1.70,y-2.90+yo),3.4,0.50,
                 boxstyle="round,pad=0,rounding_size=0.12",
                 fc=c, ec=ec, lw=0.95, zorder=5))
def ic_people(x, y, c):
    # Three person silhouettes side by side, all color c.
    dx = 2.1                 # horizontal spacing between people
    hr = 0.68                # head radius
    hw = 0.92                # body half-width (~1.8 mm wide shoulders)
    bh = 1.2                 # body (shoulders) height
    base = y - 1.4           # bottom of the shoulders shape
    head_cy = y + 0.6        # head center
    t = np.linspace(0.0, np.pi, 80)
    for i in (-1, 0, 1):
        cx = x + i * dx
        # rounded shoulders / body: filled half-ellipse dome
        bx = cx + hw * np.cos(t)
        by = base + bh * np.sin(t)
        pts = list(zip(bx, by))
        ax.add_patch(Polygon(pts, closed=True, fill=True, fc=c, ec="none", zorder=6))
        # head, set just above the shoulders with a small even neck gap
        disc(cx, head_cy, hr, c, z=7)
def ic_bars(x, y, c):
    w = 0.9                       # bar width
    pitch = 1.35                  # center-to-center spacing
    heights = [1.5, 2.8, 2.1]     # bar heights (mm)
    y0 = y - 1.4                  # common baseline (chart vertically centered)
    cx = [x - pitch, x, x + pitch]
    r = 0.16                      # top-corner rounding radius
    # faint baseline, extends slightly beyond the outer bars
    line([(cx[0] - w/2 - 0.35, y0), (cx[2] + w/2 + 0.35, y0)],
         mid(c, 0.55), lw=0.9, z=5)
    # bars: square body flush on baseline + rounded-top cap.
    # the cap's rounded bottom is hidden under the full-width body,
    # so there is no white mask and no background-colour dependency.
    for bx, h in zip(cx, heights):
        top = y0 + h
        ax.add_patch(Rectangle((bx - w/2, y0), w, h - r,
                     fc=c, ec="none", zorder=6))
        ax.add_patch(FancyBboxPatch((bx - w/2, top - 2*r), w, 2*r,
                     boxstyle="round,pad=0,rounding_size=%s" % r,
                     fc=c, ec="none", zorder=6, mutation_aspect=1.0))
def ic_line(x, y, c):
    base = y - 2.45
    # faint dashed horizontal baseline
    line([(x-2.9, base), (x+2.9, base)], mid(c, 0.62), lw=0.8, z=5, ls=(0,(3,2)))
    # solid rising diagonal trend line, lifting off the baseline (lower-left to upper-right)
    p0 = (x-2.6, base + 0.15)
    p1 = (x+2.6, y + 2.5)
    line([p0, p1], c, lw=1.0, z=6)
    # 3 small dots along the line, each on a thin white halo so the line reads cleanly
    for t in (0.1, 0.5, 0.9):
        dx = p0[0] + t*(p1[0]-p0[0])
        dy = p0[1] + t*(p1[1]-p0[1])
        disc(dx, dy, 0.37, "white", z=7)
        disc(dx, dy, 0.27, c, z=8)
def ic_square(x, y, c):
    # dashed baseline "floor" reference line, ~3mm wide, ABOVE the square
    yline = y + 1.9
    line([(x-1.55, yline),(x+1.55, yline)], mid(c,0.18), lw=1.05, z=6, ls=(0,(2.8,2.0)))
    # filled rounded color-c square (~2.4mm), low-centre
    s = 2.4
    cy = y - 0.45
    ax.add_patch(FancyBboxPatch((x-s/2, cy-s/2), s, s,
        boxstyle="round,pad=0,rounding_size=0.42",
        fc=c, ec="none", zorder=7))

# ---------------- title + columns ----------------
ax.text(W / 2, H - 4.5, "Method categories and comparator framework", ha="center", va="center",
        fontsize=F_TITLE, fontweight="bold", color=INK)
GX = [2.5, 75.5, 136.5]; CW = [71.0, 59.0, 52.0]; TOP = H - 11.0
def column(x, w, letter, title):
    rrect(x, 3.0, w, TOP - 3.0, PANELBG, OUTLINE, lw=0.8, r=2.4, z=1)
    ax.text(x + 3.4, TOP - 4.4, letter, ha="left", va="center", fontsize=F_LET, fontweight="bold", color=INK)
    ax.text(x + w / 2 + 2.4, TOP - 4.4, title, ha="center", va="center", fontsize=F_PANEL, fontweight="bold", color=INK)

# ===================== (a) Transfer assumptions =====================
column(GX[0], CW[0], "a", "Transfer assumptions")
cats = [(1, BLUE, "Foundation models", "Reusable cell-state representation", "scGPT, scFoundation, scBERT", ic_found),
        (2, GREEN, "Latent / compositional models", "Latent perturbation effect", "scGen, CPA", ic_lat),
        (3, PURPLE, "Prior-graph models", "Regulatory wiring", "GEARS, AttentionPert", ic_graph),
        (4, ORANGE, "Optimal-transport models", "Distributional intervention", "CellOT, scPRAM", ic_ot),
        (5, TEAL, "Hybrid models", "Representation + transition operator", "STATE, PertAdapt", ic_hyb),
        (6, GOLD, "Chemistry-aware models", "Compound-structure signal", "chemCPA, FP-ridge", ic_chem)]
ax0 = GX[0] + 2.6; aw = CW[0] - 5.2; ch = 14.7; gap = 1.55; ytop = TOP - 9.0
for i, (n, c, title, desc, ex, icon) in enumerate(cats):
    yt = ytop - i * (ch + gap); yb = yt - ch
    rrect(ax0, yb, aw, ch, tint(c, 0.92), mid(c, 0.30), lw=0.6, r=1.5, z=2)
    badge(ax0 + 3.8, yt - 3.3, n, c)
    icon(ax0 + 10.6, yt - ch / 2, c)
    tx = ax0 + 16.8
    ax.text(tx, yt - 3.3, title, ha="left", va="top", fontsize=F_BOX, fontweight="bold", color=c, zorder=7)
    ax.text(tx, yt - 8.0, desc, ha="left", va="top", fontsize=F_BODY, color=INK, zorder=7)
    ax.text(tx, yb + 2.5, ex, ha="left", va="center", fontsize=F_EX, color=GREYTX, zorder=7)

# ===================== (b) Benchmark roles =====================
column(GX[1], CW[1], "b", "Benchmark roles")
bx0 = GX[1] + 2.6; bw = CW[1] - 5.2
roles = [(1, BLUE, ic_net, "Conditioned predictors", "General predictive models\nevaluated on held-out tasks", None),
         (2, GREEN, ic_target, "Task-specific\ndeterministic predictors", "FP-ridge, linear-shift-KOemb", None),
         (3, PURPLE, ic_scales, "Distributional comparator", "CINEMA-OT", "reference only; never counted\nas a conditioned win")]
ybt = TOP - 9.0; rh = 20.4; rgap = 1.9
for i, (n, c, icon, title, desc, ital) in enumerate(roles):
    yt = ybt - i * (rh + rgap); yb = yt - rh
    rrect(bx0, yb, bw, rh, tint(c, 0.92), mid(c, 0.30), lw=0.6, r=1.5, z=2)
    badge(bx0 + 3.6, yt - 3.3, n, c)
    icon(bx0 + 8.8, yt - rh / 2, c)
    tx = bx0 + 13.0
    nl = title.count("\n") + 1
    ax.text(tx, yt - 3.2, title, ha="left", va="top", fontsize=F_BOX, fontweight="bold", color=c, zorder=7, linespacing=1.04)
    ax.text(tx, yt - 3.4 - nl * 3.55 - 1.0, desc, ha="left", va="top", fontsize=F_BODY, color=INK, zorder=7, linespacing=1.18)
    if ital:
        ax.text(tx, yb + 2.0, ital, ha="left", va="bottom", fontsize=F_NOTE, fontstyle="italic", color=GREYTX, zorder=7, linespacing=1.12)

# Baselines (4)
c = GOLD; yt = ybt - 3 * (rh + rgap); base_h = 27.6; yb = yt - base_h
rrect(bx0, yb, bw, base_h, tint(c, 0.92), mid(c, 0.30), lw=0.6, r=1.5, z=2)
badge(bx0 + 3.6, yt - 3.2, 4, c)
ax.text(bx0 + bw / 2 + 1.8, yt - 3.2, "Baselines", ha="center", va="center", fontsize=F_BOX + 0.4, fontweight="bold", color=c, zorder=7)
sub_y = yb + 2.1; sub_h = base_h - 8.0
SUBS = [("Universal floor", ["cell-mean shift", "linear-PCA shift"], ic_square, bx0 + 1.6, 22.0),
        ("Context baselines", ["control-as-prediction", "donor shift", "training-mean shift"], ic_people, bx0 + 25.2, bw - 25.2)]
for lab, items, ic, sx, sww in SUBS:
    rrect(sx, sub_y, sww, sub_h, "white", mid(c, 0.4), lw=0.55, r=1.1, z=3)
    ax.text(sx + sww / 2, sub_y + sub_h - 2.2, lab, ha="center", va="center", fontsize=F_NOTE, fontweight="bold", color=mid(c, 0.05), zorder=7)
    ic(sx + sww / 2, sub_y + sub_h - 6.0, GREYTX)
    for kk, it in enumerate(items):
        ax.text(sx + 1.9, sub_y + sub_h - 10.2 - kk * 2.5, it, ha="left", va="center", fontsize=F_NOTE - 0.3, color=INK, zorder=7)

# ===================== (c) Decision rule =====================
column(GX[2], CW[2], "c", "Decision rule")
cx0 = GX[2] + CW[2] / 2; cw = CW[2] - 6.4
def varr(y1, y2): arr((cx0, y1), (cx0, y2), INK, lw=1.0, ms=7)
y = TOP - 13.0
rrect(cx0 - cw / 2, y - 3.7, cw, 7.4, tint(BLUE, 0.92), mid(BLUE, 0.28), lw=0.8, r=1.6, z=3)
ax.text(cx0, y, "Conditioned prediction", ha="center", va="center", fontsize=F_BOX, fontweight="bold", color=BLUE, zorder=7)
varr(y - 3.7, y - 9.0)
cmp_h = 24.0; ccy = y - 9.0 - cmp_h / 2
rrect(cx0 - cw / 2, ccy - cmp_h / 2, cw, cmp_h, "#FCFDFE", mid(GREYTX, 0.55), lw=0.7, r=1.6, ls=(0, (3, 2.2)), z=3)
ax.text(cx0, ccy + cmp_h / 2 - 2.7, "Compare against", ha="center", va="center", fontsize=F_BODY + 0.3, color=INK, zorder=7)
for row, (lab, ic) in enumerate([("cell-mean shift", ic_bars), ("linear-PCA shift", ic_line)]):
    ry = ccy + 4.1 - row * 9.2
    rrect(cx0 - cw / 2 + 2.4, ry - 3.4, cw - 4.8, 6.8, "white", "#CBD3DB", lw=0.55, r=1.2, z=4)
    ic(cx0 - cw / 2 + 6.2, ry, GREYTX); ax.text(cx0 - cw / 2 + 10.6, ry, lab, ha="left", va="center", fontsize=F_BODY + 0.2, color=INK, zorder=7)
    if row == 0:
        ax.text(cx0, ccy - 0.5, "AND", ha="center", va="center", fontsize=F_NOTE, fontweight="bold", color=GREYTX, zorder=7)
varr(ccy - cmp_h / 2, ccy - cmp_h / 2 - 5.8)
gy = ccy - cmp_h / 2 - 5.8 - 5.4
rrect(cx0 - cw / 2, gy - 5.4, cw, 10.8, tint(GREEN, 0.91), GREEN, lw=0.8, r=1.6, z=3)
ic_trophy(cx0 - cw / 2 + 5.0, gy, GREEN)
ax.text(cx0 - cw / 2 + 9.0, gy + 1.7, "Counted as working", ha="left", va="center", fontsize=F_BOX, fontweight="bold", color=GREEN, zorder=7)
ax.text(cx0 - cw / 2 + 9.0, gy - 1.8, "only if it beats both\nfloor members", ha="left", va="center", fontsize=F_NOTE, color=INK, zorder=7, linespacing=1.12)
ny = gy - 5.4 - 3.4 - 4.6
rrect(cx0 - cw / 2, ny - 4.6, cw, 9.2, "#FAFBFC", mid(GREYTX, 0.6), lw=0.6, r=1.5, ls=(0, (2.6, 2.0)), z=3)
ic_people(cx0 - cw / 2 + 5.0, ny, GREYTX)
ax.text(cx0 - cw / 2 + 9.2, ny, "Context baselines and\ntask-specific comparators are\nreported for interpretation",
        ha="left", va="center", fontsize=F_NOTE, color=GREYTX, zorder=7, linespacing=1.16)
dvy = ny - 4.6 - 3.3
line([(cx0 - cw / 2, dvy), (cx0 + cw / 2, dvy)], PURPLE, lw=0.7, ls=(0, (3, 2.2)), z=3)
py = dvy - 3.3 - 4.7
rrect(cx0 - cw / 2, py - 4.7, cw, 9.4, tint(PURPLE, 0.91), PURPLE, lw=0.8, r=1.6, z=3)
ic_scales(cx0 - cw / 2 + 5.2, py, PURPLE)
ax.text(cx0 - cw / 2 + 9.4, py + 1.6, "CINEMA-OT", ha="left", va="center", fontsize=F_BOX, fontweight="bold", color=PURPLE, zorder=7)
ax.text(cx0 - cw / 2 + 9.4, py - 1.9, "distributional reference only", ha="left", va="center", fontsize=F_NOTE, color=INK, zorder=7)
arr((cx0 + cw / 2 - 1.1, py + 3.8), (cx0 + cw / 2 - 1.1, ccy), PURPLE, lw=0.75, ls=(0, (2.6, 2.0)), rad=-0.32, ms=6)

for ext in ("svg", "pdf"):
    fig.savefig(OUTDIR / f"{STEM}.{ext}", facecolor="white")
fig.savefig(OUTDIR / f"{STEM}.png", dpi=600, facecolor="white")
plt.close(fig)
print("wrote", STEM, "svg/pdf/png to", OUTDIR)
