#!/usr/bin/env python3
"""make_figure1_benchmark_process.py

Manuscript Figure 1 — a single holistic benchmark-process schematic that makes the whole
pipeline visible at a glance as one calm left-to-right flow. It REPLACES the former Figure 1
(benchmark framework) by absorbing its split / axis / task / method content into one
comprehensive process figure.

Five stages, read as one argument, connected by flow arrows:
  1 Curated tasks         — five tasks T1-T5 across three perturbation classes
  2 Leak-safe splits      — four held-out settings, fit on training-fold cells only
  3 Methods + floor       — six families by assumed transferable signal + the universal floor
  4 Immune-aware metrics  — three complementary fidelity axes
  5 Floor-adjudicated     — a model "works" only if it beats BOTH floor members

Drawn with matplotlib vector primitives only (no raster icon assets). Physical size in
millimetres so font sizes are true print points; editable-text SVG + PDF (svg.fonttype none,
pdf.fonttype 42) and a 600-dpi PNG. Palette and family icons match the neighbouring vector
schematic so the whole plate reads as one polished paper. Unicode minus throughout.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, FancyArrowPatch, Polygon,
                                Arc, Rectangle, Ellipse)

plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": True})

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "_paper"
OUTDIR.mkdir(parents=True, exist_ok=True)
STEM = "figure1_benchmark_process"
DEPOT = ROOT.parent / "BiB_submission" / "figures"
VECT = DEPOT / "vector_pdf"

# ---- palette (exact; cohesive with the neighbouring method-comparator schematic) ----
BLUE, GREEN, PURPLE, ORANGE, TEAL, GOLD = "#2B6CB0", "#2F855A", "#6B46C1", "#DD6B20", "#319795", "#B7791F"
GREYTX, OUTLINE, PANELBG, INK = "#4A5568", "#CBD5E0", "#F8FAFC", "#1A202C"
NAVY = "#1F4E79"          # structural pipeline ink (spine, arrows, stage badges)
CLAY = "#C2410C"          # below-floor counter-pole (used only as a small accent in stage 5)
# muted axis-identity hues (the four splits) — the established cross-figure axis key
AXIS = {"cell-context": "#6E94C9", "perturbation": "#E0876B",
        "modality": "#5BB3A9", "donor": "#A98BCB"}


def tint(hexc, f=0.90):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c * (1 - f) + f)


def mid(hexc, f=0.45):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c * (1 - f) + f)


def text_on(hexc):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return "white" if lum < 0.62 else INK


# ---- font points ----
F_TITLE = 14.5
F_STAGE = 10.0     # short stage name (Tasks / Splits / ...)
F_NUM = 7.6        # number in the stage badge
F_SUB = 6.8        # stage subtitle
F_CARD = 7.4       # task / family / metric names
F_BODY = 6.6       # secondary body
F_SMALL = 6.2      # sub-lines, task tags
F_NOTE = 6.0       # rule / interpretation notes
F_CLASS = 6.0      # perturbation-class group caps
ILW = 0.8          # icon line weight

# ---- canvas (mm) ----
W, H = 186.0, 112.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# registry of (artist, panel-inner-box) for the containment self-verify
PANELS: dict = {}
TXT: list = []


def T(pid, x, y, s, **kw):
    t = ax.text(x, y, s, **kw)
    TXT.append((t, pid))
    return t


# ---- primitives ----
def rrect(x, y, w, h, fc, ec, lw=0.7, r=1.6, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z, joinstyle="round"))


def line(p, c, lw=ILW, z=6, ls="solid"):
    ax.plot([q[0] for q in p], [q[1] for q in p], color=c, lw=lw, zorder=z, ls=ls,
            solid_capstyle="round", solid_joinstyle="round")


def disc(x, y, r, c, z=6):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=z))


def ring(x, y, r, c, lw=ILW, z=6):
    ax.add_patch(Circle((x, y), r, fill=False, ec=c, lw=lw, zorder=z))


def arr(p0, p1, c, lw=ILW, z=7, ls="solid", rad=0.0, ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, ls=ls, zorder=z,
                                 capstyle="round"))


# ---- family / role icons (authored in mm, ~6 mm footprint; match the sibling schematic) ----
def ic_found(x, y, c):
    # BIG-THEN-SHRINK: author the embedding glyph in a local unit frame (~ -5..+5),
    # then map to mm with ONE uniform scale S so it fills the ~5.4 mm icon box.
    S = 0.52

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    def darken(hexc, f):
        cc = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
        return tuple(cc * (1 - f))

    dark = darken(c, 0.30)      # lower-center cluster: darkened navy-blue
    light = mid(c, 0.38)        # upper-right cluster: lighter / medium blue
    axc = "#A3AEBC"             # subtle grey L-frame

    # thin grey L-shaped axis frame (origin bottom-left; no arrowheads)
    ox, oy = -4.4, -4.4
    line([L(ox, 4.7), L(ox, oy), L(4.7, oy)], axc, 1.55 * S, z=4)

    rng = np.random.default_rng(7)

    def cloud(cx, cy, col, n=18, spread=1.18, rad=0.40):
        k = 0
        while k < n:
            px, py = rng.normal(0.0, spread, 2)
            if px * px + py * py <= (2.0 * spread) ** 2:
                lx, ly = L(cx + px, cy + py)
                disc(lx, ly, rad * S, col, z=6)
                k += 1
    cloud(-0.3, -1.7, dark)     # lower-center, dark navy
    cloud(2.5, 2.2, light)      # upper-right, lighter blue


def ic_lat(x, y, c):
    # BIG-THEN-SHRINK: author the manifold-orbit glyph in a local unit frame (~ -5..+5),
    # then map to mm with ONE uniform scale S so it fills the ~5.4 mm icon box.
    # Reference (ref icon 2): a dashed green orbit; three filled green nodes (left-center,
    # top, lower-right); a SOLID curved arrow top -> lower-right along the right of the orbit
    # and a DASHED curved arrow lower-right -> left along the bottom -> a cyclic compositional
    # trajectory among latent states on a manifold.
    from matplotlib.path import Path
    S = 0.52

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    R = 4.0                       # orbit radius (local units)
    orbit_c = mid(c, 0.42)        # lighter green for the manifold ring

    # dashed orbit (manifold)
    ax.add_patch(Circle((x, y), R * S, fill=False, ec=orbit_c, lw=1.7 * S,
                        ls=(0, (3, 2.2)), zorder=5))

    # curved orbit-following arrow (polyline shaft so it hugs the ring; head at the end)
    def orbit_arrow(a0, a1, dashed):
        a = np.deg2rad(np.linspace(a0, a1, 30))
        verts = [L(R * np.cos(t), R * np.sin(t)) for t in a]
        ax.add_patch(FancyArrowPatch(path=Path(verts), arrowstyle="-|>", mutation_scale=4.0,
                                     color=c, lw=1.85 * S, zorder=7,
                                     ls=((0, (2.6, 2.0)) if dashed else "solid"),
                                     capstyle="round", joinstyle="round"))

    orbit_arrow(90 - 13, -40 + 14, dashed=False)    # solid: top -> lower-right (right side)
    orbit_arrow(-40 - 13, -180 + 13, dashed=True)   # dashed: lower-right -> left (bottom)

    # three filled nodes evenly placed around the orbit
    for ang in (90.0, 180.0, -40.0):                # top, left-center, lower-right
        t = np.deg2rad(ang)
        nx, ny = L(R * np.cos(t), R * np.sin(t))
        ax.add_patch(Circle((nx, ny), 0.60 * S, fc=c, ec="white", lw=0.5 * S, zorder=8))


def ic_graph(x, y, c):
    # BIG-THEN-SHRINK: author the regulatory-wiring graph in a local unit frame (~ -5..+5),
    # then map to mm with ONE uniform scale S so it fills the ~5.4 mm icon box.
    # Reference (ref icon 3): six purple filled nodes in an irregular hexagonal spread,
    # joined by a MIX of solid (known) and dashed (inferred) purple edges into one connected
    # network; each node carries a faint darker rim.
    S = 0.52

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    def darken(hexc, f):
        cc = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
        return tuple(cc * (1 - f))

    edge = mid(c, 0.24)          # purple edges, a touch lighter than the node ink
    rim = darken(c, 0.34)        # faint darker node rim

    # six nodes in an irregular hexagonal spread (local units)
    N = [(-0.6, 3.90), (3.90, 2.00), (2.40, -2.20),
         (-1.30, -3.80), (-3.90, -0.60), (-2.70, 2.40)]
    P = [L(nx, ny) for nx, ny in N]

    # connected wiring: four solid (known) + two dashed (inferred) edges, hexagon perimeter
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        line([P[i], P[j]], edge, lw=1.62 * S, z=5)
    for i, j in [(2, 3), (5, 0)]:
        line([P[i], P[j]], edge, lw=1.62 * S, z=5, ls=(0, (6.2 * S, 4.2 * S)))

    # filled nodes with a faint darker rim
    for px, py in P:
        ax.add_patch(Circle((px, py), 0.82 * S, fc=c, ec=rim, lw=0.95 * S, zorder=7))


def ic_ot(x, y, c):
    # BIG-THEN-SHRINK: author the optimal-transport glyph in a local unit frame (~ -5..+5),
    # then map to mm with ONE uniform scale S so it fills the ~5.4 mm icon box.
    # Reference (ref icon 4): two SOFT round dot-clouds side by side -- LEFT = orange dots
    # (control), RIGHT = blue dots (perturbed) -- with a dashed navy curved arrow arching from
    # the top of the left cloud to the top of the right cloud (the distributional transport map).
    # No hard outlined circle borders.
    S = 0.52
    cb = BLUE

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    rng = np.random.default_rng(11)

    def cloud(cx, cy, col, n=13, spread=0.86, rad=0.42):
        k = 0
        while k < n:
            px, py = rng.normal(0.0, spread, 2)
            if px * px + py * py <= (1.8 * spread) ** 2:
                disc(x + S * (cx + px), y + S * (cy + py), rad * S, col, z=6)
                k += 1

    cloud(-2.65, -0.5, c)      # left soft cloud: orange (control)
    cloud(2.65, -0.5, cb)      # right soft cloud: blue (perturbed)

    # dashed navy curved transport arrow arching over the top, left cloud -> right cloud.
    # Shaft and head are decoupled so both stay crisp at 6 mm: the shaft is a dashed arc
    # polyline (clean, even dash cadence) and the head is a separate solid triangle that lands
    # on the blue (perturbed) cloud.
    def arcy(xx):
        return 2.3 - 0.176 * xx * xx          # shallow parabolic arch over both clouds
    xs = np.linspace(-2.5, 2.2, 44)
    shaft = [L(xx, arcy(xx)) for xx in xs]
    ax.plot([p[0] for p in shaft], [p[1] for p in shaft], color=NAVY, lw=0.9 * S,
            ls=(0, (2.2, 1.6)), dash_capstyle="butt", zorder=8)
    ex = 2.2                                   # arrowhead anchored at the shaft end
    hx, hy = 1.0, -0.352 * ex                  # tangent of the arch at ex
    n = (hx * hx + hy * hy) ** 0.5
    hx, hy = hx / n, hy / n
    px_, py_ = -hy, hx                         # in-plane normal
    hl, hw = 0.95, 0.46                        # head length / half-width (local units)
    E = (ex, arcy(ex))
    head = [L(E[0] + hl * hx, E[1] + hl * hy),
            L(E[0] + hw * px_, E[1] + hw * py_),
            L(E[0] - hw * px_, E[1] - hw * py_)]
    ax.add_patch(Polygon(head, closed=True, fc=NAVY, ec="none", zorder=8))


def ic_hyb(x, y, c):
    # BIG-THEN-SHRINK: author the hybrid glyph in a local unit frame (~ -5..+5), then map to mm
    # with ONE uniform scale S so it fills the ~5.4 mm icon box.
    # Reference (ref icon 5): two equal rounded-square teal-outlined white boxes; LEFT holds an
    # italic "z" (a learned representation), RIGHT a bold italic "T" (a transition operator); a
    # short solid teal arrow points z -> T across the gap between them. Clean and balanced.
    S = 0.52

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    hs = 1.95                     # box half-side (local units) -> equal rounded squares
    cxL, cxR = -2.62, 2.62        # box centers (local)
    rnd = 0.62 * S                # corner rounding (mm)

    def box(cx):
        ax.add_patch(FancyBboxPatch((x + S * (cx - hs), y + S * (-hs)),
                     2 * hs * S, 2 * hs * S,
                     boxstyle=f"round,pad=0,rounding_size={rnd}",
                     fc="white", ec=c, lw=1.6 * S, zorder=6, joinstyle="round"))

    box(cxL)
    box(cxR)

    # short solid teal arrow z -> T across the inner gap (positive stub, head bumped so it reads
    # clearly as a directional arrowhead at the ~6 mm print size)
    arr(L(cxL + hs - 0.32, 0.0), L(cxR - hs + 0.04, 0.0), c, lw=1.7 * S, ms=6.0)

    # letters: italic z (representation) and bold italic T (transition operator); z carries a
    # larger point size so its x-height optically matches the full cap height of the bold T.
    ax.text(*L(cxL, 0.04), "z", ha="center", va="center", fontsize=13.8 * S, color=c,
            fontstyle="italic", zorder=7)
    ax.text(*L(cxR, 0.0), "T", ha="center", va="center", fontsize=11.0 * S, color=c,
            fontstyle="italic", fontweight="bold", zorder=7)


def ic_chem(x, y, c):
    # BIG-THEN-SHRINK: author the skeletal molecule in a local unit frame (~ -5..+5), then map to
    # mm with ONE uniform scale S so it fills the ~5.4 mm icon box centered on (x, y).
    # Reference (ref icon 6): a realistic SKELETAL structure -- 4-(methylamino)phenyl-C(=O)-CH2-NH2.
    # A benzene ring (aromatic marks) bears a para methylamino group ("N" + methyl stub) on the left
    # and, at the opposite vertex, a carbonyl ("O" up) continuing through a CH2 kink to a terminal
    # "NH2". Per spec the molecule is drawn in DARK INK (the gold family colour `c` stays the family
    # text colour, NOT the structure). The amide is routed vertically so the wide "NH2" label
    # terminates near the box centre and the glyph reads de-crowded at 6 mm; the hexagon is the hero.
    ink = INK
    S = 0.52
    base_pt, nh2_pt = 9.0, 8.0     # atom labels bumped for clear legibility at the ~6 mm print size

    def L(lx, ly):
        return (x + S * lx, y + S * ly)

    def seg(a, b, w):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ink, lw=w, solid_capstyle="round",
                solid_joinstyle="round", zorder=6)

    def bond(a, b, w, g0=0.0, g1=0.0):
        # straight bond a -> b (local units), trimmed by g0 at the a-end / g1 at the b-end so a
        # bonded atom label sits in a clean gap (the gaps scale with the bumped label glyphs)
        a = np.array(a, float); b = np.array(b, float)
        u = (b - a) / np.hypot(*(b - a))
        seg(L(*(a + u * g0)), L(*(b - u * g1)), w)

    lw = 0.95 * S
    # benzene ring (para vertices left/right), pushed LEFT; the visual hero
    rcx, rcy, rr = -1.68, 0.15, 1.50
    ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    V = [(rcx + rr * np.cos(a), rcy + rr * np.sin(a)) for a in ang]
    ax.add_patch(Polygon([L(*v) for v in V], closed=True, fill=False, ec=ink, lw=lw,
                         zorder=6, joinstyle="round"))
    # aromatic cue: two short, thin inner lines (top + bottom edges) -- reads as benzene, stays crisp
    cen = np.array([rcx, rcy])
    for i, j in [(1, 2), (4, 5)]:
        A, B = np.array(V[i]), np.array(V[j])
        n = cen - (A + B) / 2; n = n / np.hypot(*n)
        u = B - A; ulen = np.hypot(*u); u = u / ulen
        seg(L(*(A + n * 0.36 + u * 0.26 * ulen)), L(*(B + n * 0.36 - u * 0.26 * ulen)), lw * 0.82)

    V0, V3 = V[0], V[3]            # right (chain), left (methylamino)
    # methylamino on V3 (left): ring bond -> "N" label -> short methyl stub up; gaps clear the glyph
    N = (V3[0] - 0.92, V3[1] + 0.64)
    Me = (N[0] - 0.30, N[1] + 1.30)
    bond(V3, N, lw, g1=0.74)                     # ring -> N (stop before the N glyph)
    bond(N, Me, lw, g0=0.74)                     # N -> methyl stub (start after the N glyph)
    ax.text(*L(*N), "N", ha="center", va="center", fontsize=base_pt * S, color=ink, zorder=8)
    # carbonyl on V0 (right): a proper C=O drawn as a short DOUBLE BOND -- two clearly separated
    # parallel lines from the carbonyl carbon up to an "O" letter label (NOT a hollow circle);
    # then a CH2 kink down to a terminal "NH2" label.
    Cc = (V0[0] + 0.76, V0[1] + 0.52)            # carbonyl carbon
    bond(V0, Cc, lw)                             # ring -> carbonyl C
    O = (Cc[0] + 0.03, Cc[1] + 2.42)             # "O" letter label, well above the double bond
    for dx in (-0.28, 0.28):                     # C=O double bond: two parallel short lines
        seg(L(Cc[0] + dx, Cc[1] + 0.16), L(Cc[0] + dx, Cc[1] + 1.24), lw)
    ax.text(*L(*O), "O", ha="center", va="center", fontsize=base_pt * S, color=ink, zorder=8)
    M = (Cc[0] + 0.40, Cc[1] - 1.16)             # CH2 vertex (down, slight right -> kink)
    bond(Cc, M, lw)
    Nv = (M[0] + 0.04, M[1] - 1.46)              # terminal amine vertex (label centre)
    seg(L(*M), L(Nv[0], Nv[1] + 0.96), lw)       # CH2 -> NH2, stopping above the label glyph
    ax.text(*L(Nv[0] - 0.12, Nv[1]), "NH$_2$", ha="left", va="center", fontsize=nh2_pt * S,
            color=ink, zorder=8)


def ic_square(x, y, c):
    yline = y + 1.85
    line([(x - 1.5, yline), (x + 1.5, yline)], mid(c, 0.18), lw=1.0, z=6, ls=(0, (2.8, 2.0)))
    s = 2.3
    ax.add_patch(FancyBboxPatch((x - s / 2, y - 0.45 - s / 2), s, s,
                 boxstyle="round,pad=0,rounding_size=0.4", fc=c, ec="none", zorder=7))


def ic_trophy(x, y, c):
    yo = 0.1
    cl = mid(c, 0.5)
    for sgn in (-1, 1):
        ax.add_patch(Arc((x + sgn * 1.5, y + 1.4 + yo), 1.2, 1.55, angle=0,
                     theta1=(90 if sgn < 0 else -90), theta2=(270 if sgn < 0 else 90),
                     ec=c, lw=0.95, zorder=5))
    bowl = [(x - 1.85, y + 2.25 + yo), (x - 1.72, y + 1.4 + yo), (x - 1.4, y + 0.55 + yo),
            (x - 0.84, y + 0.0 + yo), (x - 0.46, y - 0.52 + yo),
            (x + 0.46, y - 0.52 + yo), (x + 0.84, y + 0.0 + yo),
            (x + 1.4, y + 0.55 + yo), (x + 1.72, y + 1.4 + yo), (x + 1.85, y + 2.25 + yo)]
    ax.add_patch(Polygon(bowl, closed=True, fill=True, fc=c, ec=c, lw=0.9, zorder=6))
    ax.add_patch(Ellipse((x, y + 2.25 + yo), 3.7, 0.65, fc=cl, ec=c, lw=0.9, zorder=7))
    ax.add_patch(Ellipse((x, y - 0.52 + yo), 0.94, 0.4, fc=c, ec=c, lw=0.9, zorder=6))
    ax.add_patch(Polygon([(x - 0.28, y - 0.66 + yo), (x + 0.28, y - 0.66 + yo),
                          (x + 0.4, y - 1.55 + yo), (x - 0.4, y - 1.55 + yo)],
                 closed=True, fill=True, fc=c, ec=c, lw=0.9, zorder=5))
    ax.add_patch(Polygon([(x - 0.8, y - 1.55 + yo), (x + 0.8, y - 1.55 + yo),
                          (x + 1.3, y - 2.3 + yo), (x - 1.3, y - 2.3 + yo)],
                 closed=True, fill=True, fc=c, ec=c, lw=0.9, zorder=5))
    ax.add_patch(FancyBboxPatch((x - 1.6, y - 2.72 + yo), 3.2, 0.46,
                 boxstyle="round,pad=0,rounding_size=0.11", fc=c, ec=c, lw=0.9, zorder=5))


def ic_shield(x, y, c):
    # rounded shield outline for the leak-safe rule
    top = y + 2.0
    bot = y - 2.4
    w = 1.9
    pts = [(x - w, top), (x + w, top), (x + w, y - 0.4),
           (x, bot), (x - w, y - 0.4)]
    ax.add_patch(Polygon(pts, closed=True, fill=True, fc=tint(c, 0.9), ec=c, lw=0.95,
                         zorder=6, joinstyle="round"))
    # check-mark inside
    line([(x - 0.85, y - 0.1), (x - 0.2, y - 0.85), (x + 0.95, y + 0.75)], c, lw=1.1, z=7)


def ic_scales(x, y, c):
    lw = 0.85
    oy = 0.1
    bx = 2.1
    top = y + 2.25 + oy
    base_y = y - 2.15 + oy
    line([(x, top), (x, base_y)], c, lw)
    line([(x - bx, top), (x + bx, top)], c, lw)
    disc(x, top, 0.22, c)
    ax.add_patch(Polygon([(x - 0.45, base_y), (x + 0.45, base_y),
                          (x + 0.85, base_y - 0.45), (x - 0.85, base_y - 0.45)],
                 closed=True, fill=True, fc=c, ec=c, lw=lw))
    line([(x - 1.1, base_y - 0.45), (x + 1.1, base_y - 0.45)], c, lw)
    rim_y = y + 0.85 + oy
    hw = 0.9
    for s in (-1, 1):
        ax_ = x + s * bx
        line([(ax_, top), (ax_ - hw, rim_y)], c, lw)
        line([(ax_, top), (ax_ + hw, rim_y)], c, lw)
        disc(ax_, top, 0.14, c)
        ax.add_patch(Arc((ax_, rim_y), 2 * hw, 0.78, angle=0, theta1=180, theta2=360, ec=c, lw=lw))


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------
ax.text(W / 2, H - 5.0, "An immune-aware benchmark of perturbation-prediction generalization",
        ha="center", va="center", fontsize=F_TITLE, fontweight="bold", color=INK)

OM, G, PAD = 2.5, 5.0, 2.7
WID = [33.0, 31.0, 42.0, 26.0, 29.0]
PB = 3.0
TOP = H - 11.5
XS = []
xc = OM
for w in WID:
    XS.append(xc)
    xc += w + G

STAGE = [("1", "Tasks", "5 curated, 3 classes"),
         ("2", "Splits", "held-out, leak-safe"),
         ("3", "Methods", "6 families + floor"),
         ("4", "Metrics", "3 immune-aware"),
         ("5", "Verdict", "floor-adjudicated")]


def panel(i):
    """draw panel shell + header; register inner box; return content-top y (cursor)."""
    x, w = XS[i], WID[i]
    il, ir = x + PAD, x + w - PAD
    PANELS[i] = (il, PB + 1.6, ir, TOP - 1.4)
    rrect(x, PB, w, TOP - PB, PANELBG, OUTLINE, lw=0.8, r=2.4, z=1)
    num, name, sub = STAGE[i]
    hy = TOP - 5.0
    ax.add_patch(Circle((il + 2.3, hy), 2.3, fc=NAVY, ec="none", zorder=6))
    T(i, il + 2.3, hy, num, ha="center", va="center", fontsize=F_NUM, fontweight="bold",
      color="white", zorder=7)
    T(i, il + 5.8, hy, name, ha="left", va="center", fontsize=F_STAGE, fontweight="bold",
      color=INK, zorder=7)
    dy = TOP - 9.6
    ax.plot([il, ir], [dy, dy], color=OUTLINE, lw=0.8, zorder=3)
    T(i, il, dy - 2.7, sub, ha="left", va="center", fontsize=F_SUB, color=GREYTX, zorder=7)
    return dy - 6.0


# ===================== (1) Curated tasks =====================
c0 = panel(0)
il = XS[0] + PAD
ir = XS[0] + WID[0] - PAD
classes = [("IMMUNE STIMULATION", TEAL,
            [("T1", "Kang IFN-β", "PBMC · RNA"),
             ("T2", "Soskic 2022", "CD4 T activation")]),
           ("CRISPR GENE", GREEN,
            [("T3", "Primary-T", "CRISPR pool · RNA"),
             ("T4", "Frangieh", "Perturb-CITE-seq")]),
           ("COMPOUND", GOLD,
            [("T5", "OP3", "small molecules")])]
y = c0
for cname, ccol, tasks in classes:
    T(0, il, y, cname, ha="left", va="center", fontsize=F_CLASS, fontweight="bold",
      color=mid(ccol, 0.05), zorder=7)
    line([(il, y - 2.1), (ir, y - 2.1)], mid(ccol, 0.35), lw=0.7, z=4)
    y -= 4.8
    for tid, name, sub in tasks:
        ch = 9.0
        rrect(il, y - ch, ir - il, ch, "white", mid(ccol, 0.4), lw=0.6, r=1.3, z=3)
        ax.add_patch(FancyBboxPatch((il + 0.6, y - ch + 0.8), 0.9, ch - 1.6,
                     boxstyle="round,pad=0,rounding_size=0.25", fc=ccol, ec="none", zorder=4))
        T(0, il + 2.6, y - 3.0, tid, ha="left", va="center", fontsize=F_CARD + 0.4,
          fontweight="bold", color=mid(ccol, 0.0), zorder=7)
        T(0, il + 6.8, y - 3.0, name, ha="left", va="center", fontsize=F_CARD,
          fontweight="bold", color=INK, zorder=7)
        T(0, il + 2.6, y - 6.4, sub, ha="left", va="center", fontsize=F_SMALL, color=GREYTX,
          zorder=7)
        y -= ch + 2.2
    y -= 2.0
END0 = y

# ===================== (2) Leak-safe splits =====================
c1 = panel(1)
il = XS[1] + PAD
ir = XS[1] + WID[1] - PAD
splits = [("Cell-context", "cell-context", "held-out lineage", "T1, T5"),
          ("Perturbation", "perturbation", "unseen target", "T3, T4, T5"),
          ("Modality", "modality", "held-out readout", "T4"),
          ("Donor", "donor", "held-out donor", "T1, T2")]
y = c1
ph = 9.0
pw_pill = (ir - il) - 2.8
for label, key, desc, tags in splits:
    col = AXIS[key]
    rrect(il, y - ph, ir - il, ph, tint(col, 0.9), mid(col, 0.45), lw=0.7, r=1.4, z=3)
    rrect(il + 1.4, y - 3.3, pw_pill, 3.0, col, "none", lw=0, r=1.0, z=4)
    T(1, il + 1.4 + pw_pill / 2, y - 1.8, label, ha="center", va="center", fontsize=F_BODY,
      fontweight="bold", color=text_on(col), zorder=7)
    T(1, il + 1.7, y - 5.5, desc, ha="left", va="center", fontsize=F_SMALL, color=INK, zorder=7)
    T(1, il + 1.7, y - 7.6, tags, ha="left", va="center", fontsize=F_SMALL, fontweight="bold",
      color=mid(col, 0.0), zorder=7)
    y -= ph + 1.7
# leak-safe rule note
y -= 0.9
rh = 23.0
rrect(il, y - rh, ir - il, rh, "white", NAVY, lw=0.75, r=1.4, z=3)
ic_shield(il + 3.4, y - 4.6, NAVY)
T(1, il + 7.0, y - 4.6, "Leak-safe", ha="left", va="center", fontsize=F_BODY,
  fontweight="bold", color=NAVY, zorder=7)
T(1, il + 1.9, y - 9.0,
  "Held-out cells appear\nin the test fold only.\nResponse-gene panel,\nPCA basis and floor are\nfit on train cells only.",
  ha="left", va="top", fontsize=5.8, color=INK, zorder=7, linespacing=1.36)
END1 = y - rh

# ===================== (3) Methods + universal floor =====================
c2 = panel(2)
il = XS[2] + PAD
ir = XS[2] + WID[2] - PAD
fams = [("Foundation", "reusable cell-state", ic_found, BLUE),
        ("Latent", "compositional effect", ic_lat, GREEN),
        ("Prior-graph", "regulatory wiring", ic_graph, PURPLE),
        ("Optimal-transport", "distributional map", ic_ot, ORANGE),
        ("Hybrid", "repr. + transition op.", ic_hyb, TEAL),
        ("Chemistry-aware", "compound structure", ic_chem, GOLD)]
y = c2
fh = 6.6
for name, desc, icon, col in fams:
    yc = y - fh / 2
    icon(il + 3.4, yc, col)
    T(2, il + 6.9, yc + 1.45, name, ha="left", va="center", fontsize=F_CARD,
      fontweight="bold", color=col, zorder=7)
    T(2, il + 6.9, yc - 1.55, desc, ha="left", va="center", fontsize=F_SMALL, color=GREYTX,
      zorder=7)
    y -= fh
# roles strip
y -= 1.4
T(2, il, y, "ROLES", ha="left", va="center", fontsize=F_CLASS, fontweight="bold",
  color=GREYTX, zorder=7)
line([(il + 8.0, y), (ir, y)], OUTLINE, lw=0.7, z=4)
y -= 3.2
roles = [("Conditioned predictors", "evaluated on held-out tasks"),
         ("Deterministic predictors", "FP-ridge, linear-shift-KOemb"),
         ("Comparator: CINEMA-OT", "never counted as a win")]
for rlead, rdet in roles:
    disc(il + 1.0, y + 0.9, 0.40, NAVY, z=6)
    T(2, il + 2.6, y + 0.9, rlead, ha="left", va="center", fontsize=F_NOTE,
      fontweight="bold", color=INK, zorder=7)
    T(2, il + 2.6, y - 1.7, rdet, ha="left", va="center", fontsize=F_NOTE,
      color=GREYTX, zorder=7)
    y -= 5.2
# universal floor box
y -= 1.3
flh = 14.6
rrect(il, y - flh, ir - il, flh, tint(NAVY, 0.9), NAVY, lw=0.95, r=1.5, z=3)
ic_square(il + 3.8, y - 3.6, NAVY)
T(2, il + 8.2, y - 3.4, "Universal floor", ha="left", va="center", fontsize=F_CARD,
  fontweight="bold", color=NAVY, zorder=7)
T(2, il + 2.4, y - 8.4, "cell-mean  +  linear-PCA shift", ha="left", va="center",
  fontsize=F_SMALL, color=INK, zorder=7)
T(2, il + 2.4, y - flh + 2.6, "works only if it beats both members",
  ha="left", va="center", fontsize=F_NOTE, fontstyle="italic", color=GREYTX, zorder=7)
END2 = y - flh

# ===================== (4) Immune-aware metrics =====================
c3 = panel(3)
il = XS[3] + PAD
ir = XS[3] + WID[3] - PAD
metrics = [("Response\ndirection", "Pearson-Δ", "downstream Δ", "higher"),
           ("Program /\nreadout", "AUCell-Δ", "per-marker", "higher"),
           ("Distribution", "E-distance", "state-space", "lower")]
y = c3
mh = 18.5
for title, stat, desc, better in metrics:
    nl = title.count("\n") + 1
    rrect(il, y - mh, ir - il, mh, "white", OUTLINE, lw=0.75, r=1.4, z=3)
    ax.add_patch(FancyBboxPatch((il + 1.2, y - mh + 1.2), 0.9, mh - 2.4,
                 boxstyle="round,pad=0,rounding_size=0.3", fc=NAVY, ec="none", zorder=4))
    T(3, il + 3.4, y - 2.4, title, ha="left", va="top", fontsize=F_CARD, fontweight="bold",
      color=INK, zorder=7, linespacing=1.08)
    sy = y - 2.4 - nl * 3.3 - 2.0
    T(3, il + 3.4, sy, stat, ha="left", va="center", fontsize=F_BODY, fontweight="bold",
      color=NAVY, zorder=7)
    T(3, il + 3.4, sy - 2.9, desc, ha="left", va="center", fontsize=F_SMALL, color=GREYTX,
      zorder=7)
    glyph = r"$\uparrow$ better" if better == "higher" else r"$\downarrow$ better"
    T(3, ir - 1.6, y - mh + 1.9, glyph, ha="right", va="center", fontsize=F_SMALL,
      fontstyle="italic", color=mid(GREEN, 0.0) if better == "higher" else mid(CLAY, 0.0),
      zorder=7)
    y -= mh + 2.0
END3 = y

# ===================== (5) Floor-adjudicated verdict =====================
c4 = panel(4)
il = XS[4] + PAD
ir = XS[4] + WID[4] - PAD
cx = (il + ir) / 2
cw = ir - il

def vdown(y1, y2):
    arr((cx, y1), (cx, y2), NAVY, lw=1.0, ms=5.5)


y = c4
# conditioned prediction
b1h = 9.0
rrect(il, y - b1h, cw, b1h, tint(BLUE, 0.9), mid(BLUE, 0.3), lw=0.8, r=1.4, z=3)
T(4, cx, y - b1h / 2, "Conditioned\nprediction", ha="center", va="center", fontsize=F_BODY,
  fontweight="bold", color=mid(BLUE, 0.0), zorder=7, linespacing=1.12)
y -= b1h
vdown(y, y - 4.2)
y -= 4.2
# decision
b2h = 16.5
rrect(il, y - b2h, cw, b2h, "#FCFDFE", GREYTX, lw=0.8, r=1.4, ls=(0, (3, 2.2)), z=3)
T(4, cx, y - 3.2, "Beats BOTH floor", ha="center", va="center", fontsize=F_BODY,
  fontweight="bold", color=INK, zorder=7)
T(4, cx, y - 6.2, "members on the", ha="center", va="center", fontsize=F_BODY,
  fontweight="bold", color=INK, zorder=7)
T(4, cx, y - 9.2, "axes it supports?", ha="center", va="center", fontsize=F_BODY,
  fontweight="bold", color=INK, zorder=7)
T(4, cx, y - b2h + 2.6, "cell-mean · linear-PCA", ha="center", va="center",
  fontsize=F_NOTE, fontstyle="italic", color=GREYTX, zorder=7)
y -= b2h
vdown(y, y - 4.2)
T(4, cx + 1.0, y - 2.1, "yes", ha="left", va="center", fontsize=F_NOTE, fontstyle="italic",
  color=GREEN, zorder=8)
y -= 4.2
# works
b3h = 11.0
rrect(il, y - b3h, cw, b3h, tint(GREEN, 0.88), GREEN, lw=0.9, r=1.4, z=3)
ic_trophy(il + 4.4, y - b3h / 2, GREEN)
T(4, il + 8.4, y - b3h / 2 + 1.6, "Works on", ha="left", va="center", fontsize=F_CARD,
  fontweight="bold", color=mid(GREEN, 0.0), zorder=7)
T(4, il + 8.4, y - b3h / 2 - 1.7, "the task", ha="left", va="center", fontsize=F_CARD,
  fontweight="bold", color=mid(GREEN, 0.0), zorder=7)
y -= b3h
# reliability-ceiling note
y -= 2.0
nh = 17.0
rrect(il, y - nh, cw, nh, "white", mid(CLAY, 0.5), lw=0.7, r=1.3, ls=(0, (2.6, 2.0)), z=3)
T(4, cx, y - 2.8, "below floor", ha="center", va="center", fontsize=F_NOTE, fontweight="bold",
  color=mid(CLAY, 0.0), zorder=7)
T(4, il + 1.9, y - 6.0,
  "A split-half\nreliability ceiling\nbounds below-floor\nresults.",
  ha="left", va="top", fontsize=F_NOTE, color=INK, zorder=7, linespacing=1.32)
END4 = y - nh

# ===================== flow arrows between stages =====================
yflow = (TOP + PB) / 2 + 6.0
for i in range(4):
    x_from = XS[i] + WID[i] + 0.5
    x_to = XS[i + 1] - 0.5
    arr((x_from, yflow), (x_to, yflow), NAVY, lw=1.6, ms=7, z=9)

# ----------------------------------------------------------------------------------------------
# SELF-VERIFY: cursor floors + per-text containment in its panel (measured, not eyeballed)
# ----------------------------------------------------------------------------------------------
ENDS = {0: END0, 1: END1, 2: END2, 3: END3, 4: END4}
for i, e in ENDS.items():
    assert e >= PB + 1.0, f"STAGE {i+1} content overruns panel floor: cursor {e:.2f} < {PB+1.0}"

fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def extent_mm(t):
    e = t.get_window_extent(rend)
    x0, y0 = inv.transform((e.x0, e.y0))
    x1, y1 = inv.transform((e.x1, e.y1))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


TOL = 0.45
worst = 0.0
for t, pid in TXT:
    il, ib, ir, it = PANELS[pid]
    x0, y0, x1, y1 = extent_mm(t)
    over = max(il - x0, x0 + 0 - x0 - 0, x1 - ir, ib - y0, y1 - it)
    # explicit four-side checks for clarity
    msgs = []
    if x0 < il - TOL:
        msgs.append(f"left {il - x0:.2f}")
    if x1 > ir + TOL:
        msgs.append(f"right {x1 - ir:.2f}")
    if y0 < ib - TOL:
        msgs.append(f"bottom {ib - y0:.2f}")
    if y1 > it + TOL:
        msgs.append(f"top {y1 - it:.2f}")
    worst = max(worst, x1 - ir, il - x0)
    assert not msgs, f"OVERFLOW stage {pid+1} '{t.get_text()[:28]}': " + ", ".join(msgs)
print(f"self-verify OK  (worst horizontal margin slack ~{worst:.2f} mm; all text inside panels)")

# ----------------------------------------------------------------------------------------------
# render + deposit
# ----------------------------------------------------------------------------------------------
for ext in ("svg", "pdf"):
    fig.savefig(OUTDIR / f"{STEM}.{ext}", facecolor="white")
fig.savefig(OUTDIR / f"{STEM}.png", dpi=600, facecolor="white")
plt.close(fig)
print("wrote", STEM, "svg/pdf/png to", OUTDIR)

# deposit as manuscript Figure 1 (overwriting the former framework figure)
DEPOT.mkdir(parents=True, exist_ok=True)
VECT.mkdir(parents=True, exist_ok=True)
from PIL import Image
# flatten the rendered RGBA raster onto white -> RGB (matches the other deposited figures)
_raw = Image.open(OUTDIR / f"{STEM}.png").convert("RGBA")
_rgb = Image.new("RGB", _raw.size, "white")
_rgb.paste(_raw, mask=_raw.split()[-1])
_rgb.save(OUTDIR / f"{STEM}.png", dpi=(600, 600))
_rgb.save(DEPOT / "Figure1.png", dpi=(600, 600))
_rgb.save(DEPOT / "Figure1.tiff", compression="tiff_lzw", dpi=(600, 600))
shutil.copyfile(OUTDIR / f"{STEM}.pdf", VECT / "Figure1.pdf")
shutil.copyfile(OUTDIR / f"{STEM}.svg", VECT / "Figure1.svg")
print("deposited Figure1.png/.tiff +", "vector_pdf/Figure1.pdf/.svg")
