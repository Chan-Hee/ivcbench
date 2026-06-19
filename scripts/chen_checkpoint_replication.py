#!/usr/bin/env python
"""Chen Perturb-icCITE-seq (E-GEAD-648) — SECOND-DATASET replication of the Frangieh checkpoint finding.

FLAGSHIP CLAIM under test (n=2 with Frangieh):
  Under CRISPR perturbation, the checkpoint RECEPTOR (PD-1 / CD279) is strongly modulated and its
  across-KO direction is recovered by the simple training-mean (cell-mean) shift, whereas the
  checkpoint LIGAND (PD-L1 / CD274) is near-zero at the SURFACE, direction-unrecoverable — and this
  asymmetry tracks the assay-floor confound (effect-size -> sign-match).

Chen panel: the surface ADT (CITE) panel carries 277 surface antibodies including
  surface_A0007_PDL1  (PD-L1 = CD274, SURFACE)   and   surface_A0088_PD1 (PD-1 = CD279, SURFACE).
So this is a SURFACE-vs-SURFACE replication (NOT intracellular) — directly on-claim.

METHOD (identical to the Frangieh c4_per_marker / c4_pdl1_assay_power):
  - Build the multimodal cells per library: GEX-called barcodes carry a confident guide call (argmax
    over the 907-sgRNA CROP-seq matrix -> target gene by stripping the _<n>); we join the surface ADT
    counts onto exactly those barcodes (ADT and GEX share the cell barcode within a library).
  - Subsample 200 cells / (library, target) to bound memory (matches the C3 chen.py loader).
  - Library-log-normalize the 277-marker ADT panel (target_sum 1e4, log1p) — the SAME preprocessing
    the Frangieh protein modality got (apples-to-apples; not a bespoke transform).
  - Leave-KO-out: held KO genes' cells removed from "train". The cell-mean shift predicts ONE global
    Δ (= treated_mean_train - control_mean) for every held KO. Per held-KO stratum the observed
    Δ = mean(KO cells) - control_mean. Per marker:
        obsΔ_mean / sd over held-KO strata; predΔ (constant); sign_match = mean over held KOs of
        [sign(obsΔ_KO) == sign(predΔ)]; effect_sd_units = |obsΔ_mean|/sd; CI = mean ± 1.96*sd/sqrt(n).
  - Two holdout fractions (25%, 50%) and the pooled effect-size-vs-sign-match relationship, exactly
    as Frangieh.

ABSOLUTE: every number is computed from the real E-GEAD-648 data. No fabrication.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr

sys.path.insert(0, "src")
from ivcbench.data.crispr import is_control_guide, read_10x_mtx, strip_trailing_index
from ivcbench.data.preprocess import library_log_normalize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "C3" / "chen"
OUT_PAPER = ROOT / "results" / "_paper"
OUT_NEW = ROOT / "results" / "newdata"
OUT_PAPER.mkdir(parents=True, exist_ok=True)
OUT_NEW.mkdir(parents=True, exist_ok=True)

Z = 1.959963984540054
TARGET_SUM = 1e4
SUBSAMPLE_PER_GENE = 200
SEED = 0

PDL1_COL = "surface_A0007_PDL1"   # CD274
PD1_COL = "surface_A0088_PD1"     # CD279

# surface ADT run is one per library (DRR500327..341); guide DRR500342..356; GEX DRR500361..374(+combos)
ADT_GLOB = "icCITE_ADT_kallisto_output_matrix_*_ADT.txt"


def _strip_bc(bc: str) -> str:
    return re.sub(r"-\d+$", "", str(bc))


def _orient(counts, barcodes, names):
    if counts.shape == (len(names), len(barcodes)) and len(names) != len(barcodes):
        return counts.T.tocsr()
    return counts


def _classify(dirpath: Path) -> str:
    feats = dirpath / "features.tsv.gz"
    names, gex = [], 0
    with gzip.open(feats, "rt") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            names.append(parts[0])
            if len(parts) >= 3 and parts[2] == "Gene Expression":
                gex += 1
    if gex > 10000:
        return "gex"
    if 500 <= len(names) <= 2000 and sum(bool(re.search(r"_\d+$", n)) for n in names) > 0.6 * len(names):
        return "guide"
    return "other"


def _runs_for_library(lib: str) -> list[str]:
    import csv
    rows = list(csv.reader(open(DATA / "E-GEAD-648.sdrf.txt"), delimiter="\t"))
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    run_i, lib_i = ci["Comment[SRA_RUN]"], ci["Factor Value[library]"]
    return [r[run_i] for r in rows[1:] if r[lib_i] == lib]


def _libraries() -> dict[str, list[str]]:
    import csv
    rows = list(csv.reader(open(DATA / "E-GEAD-648.sdrf.txt"), delimiter="\t"))
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    run_i, lib_i = ci["Comment[SRA_RUN]"], ci["Factor Value[library]"]
    libs: dict[str, list[str]] = {}
    for r in rows[1:]:
        libs.setdefault(r[lib_i], []).append(r[run_i])
    return libs


def _find_dir(run: str, kind_files: list[str]) -> Path | None:
    """Find an already-extracted run dir under data/.../extracted/* whose run id matches and that
    contains the requested files (matrix.mtx.gz for 10x, the ADT txt for protein)."""
    ex = DATA / "extracted"
    for marker in ex.glob("*"):
        # the run dir tree is .../<run>/GEA/transfer_*/<run>/...
        for sub in marker.rglob("*"):
            if not sub.is_dir():
                continue
            if sub.name != run and run not in sub.name:
                continue
            if all(any(sub.glob(p)) for p in kind_files):
                return sub
    return None


def _adt_file_for_run(run: str) -> Path | None:
    d = _find_dir(run, [ADT_GLOB])
    if d is None:
        return None
    hits = list(d.glob(ADT_GLOB))
    return hits[0] if hits else None


def read_adt(path: Path):
    """Read the surface ADT txt (CellBarcode + 277 surface markers). Returns (csr counts cells x
    markers, barcodes, marker_names)."""
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    markers = hdr[1:]
    df = pd.read_csv(path, sep="\t", header=0)
    bc = df.iloc[:, 0].astype(str).to_numpy()
    X = sp.csr_matrix(df.iloc[:, 1:].to_numpy(dtype=np.float32))
    return X, bc, markers


# ---------------------------------------------------------------------------------------------------
# Assemble the multimodal Chen surface-CITE cells
# ---------------------------------------------------------------------------------------------------
def assemble():
    libs = _libraries()
    rng = np.random.default_rng(SEED)
    blocks_X = []          # ADT counts per kept cell (cells x markers)
    blocks_pert = []
    blocks_ctrl = []
    blocks_lib = []
    marker_ref = None

    for lib in sorted(libs, key=lambda s: (len(s), s)):
        runs = libs[lib]
        gex_dir = guide_dir = adt_path = None
        for run in runs:
            if adt_path is None:
                ap = _adt_file_for_run(run)
                if ap is not None:
                    adt_path = ap
                    continue
            d = _find_dir(run, ["matrix.mtx.gz"])
            if d is None:
                continue
            kind = _classify(d)
            if kind == "gex" and gex_dir is None:
                gex_dir = d
            elif kind == "guide" and guide_dir is None:
                guide_dir = d
        if gex_dir is None or guide_dir is None or adt_path is None:
            print(f"  [skip] {lib}: gex={gex_dir is not None} guide={guide_dir is not None} adt={adt_path is not None}")
            continue

        # GEX barcodes (the real cells)
        gex_counts, gex_bc, genes, _ = read_10x_mtx(gex_dir)
        gex_counts = _orient(gex_counts, gex_bc, genes)
        gex_bc = [_strip_bc(b) for b in gex_bc]

        # guide assignment by argmax
        g_counts, g_bc, g_names, _ = read_10x_mtx(guide_dir)
        g_counts = _orient(g_counts, g_bc, g_names)
        g_bc = [_strip_bc(b) for b in g_bc]
        g_arg = np.asarray(g_counts.argmax(axis=1)).ravel()
        g_tot = np.asarray(g_counts.sum(axis=1)).ravel()
        guide_gene = {b: (strip_trailing_index(g_names[a]) if t > 0 else None)
                      for b, a, t in zip(g_bc, g_arg, g_tot)}

        # ADT counts
        adt_X, adt_bc, markers = read_adt(adt_path)
        adt_bc = np.array([_strip_bc(b) for b in adt_bc])
        if marker_ref is None:
            marker_ref = markers
        else:
            assert markers == marker_ref, f"marker panel differs in {lib}"
        adt_index = {b: i for i, b in enumerate(adt_bc)}

        # keep GEX cells that have a confident guide call AND an ADT row
        keep_adt_rows, perts, is_ctrl = [], [], []
        for b in gex_bc:
            gene = guide_gene.get(b)
            if gene is None:
                continue
            ai = adt_index.get(b)
            if ai is None:
                continue
            ctrl = is_control_guide(gene)
            keep_adt_rows.append(ai)
            perts.append("__ctrl__" if ctrl else gene)
            is_ctrl.append(ctrl)
        if not keep_adt_rows:
            continue
        keep_adt_rows = np.array(keep_adt_rows)
        perts = np.array(perts, dtype=object)
        is_ctrl = np.array(is_ctrl, dtype=bool)

        # cap cells per target within the library
        sel = []
        for lab in pd.unique(perts):
            idx = np.where(perts == lab)[0]
            sel.append(idx if len(idx) <= SUBSAMPLE_PER_GENE
                       else rng.choice(idx, SUBSAMPLE_PER_GENE, replace=False))
        sel = np.sort(np.concatenate(sel))
        rows = keep_adt_rows[sel]
        blocks_X.append(adt_X[rows])
        blocks_pert.append(perts[sel])
        blocks_ctrl.append(is_ctrl[sel])
        blocks_lib.append(np.array([lib] * len(sel), dtype=object))
        print(f"  [ok] {lib}: {len(sel)} cells ({int(is_ctrl[sel].sum())} ctrl), "
              f"{len(pd.unique(perts[~is_ctrl]))} KO targets")

    X = sp.vstack(blocks_X).tocsr()
    pert = np.concatenate(blocks_pert)
    is_ctrl = np.concatenate(blocks_ctrl)
    lib = np.concatenate(blocks_lib)
    return X, pert, is_ctrl, lib, marker_ref


print("=== Assembling Chen surface-CITE multimodal cells ===")
X_counts, pert, is_ctrl, lib_arr, markers = assemble()
print(f"\nTotal cells: {X_counts.shape[0]}  markers: {X_counts.shape[1]}  "
      f"controls: {int(is_ctrl.sum())}  KO targets: {len(set(pert[~is_ctrl]))}")
assert PDL1_COL in markers and PD1_COL in markers
jpdl1 = markers.index(PDL1_COL)
jpd1 = markers.index(PD1_COL)

# Library-log-normalize the 277-marker ADT panel (SAME preprocessing as Frangieh protein)
Xn = library_log_normalize(X_counts, TARGET_SUM, log1p=True)
Xn = np.asarray(Xn.todense(), dtype=np.float32)

# CLR (centered-log-ratio) — the canonical ADT normalization — for a robustness cross-check.
# CLR per cell over the 277-marker panel: log1p(count) - mean_over_markers(log1p(count)).
_l = np.log1p(np.asarray(X_counts.todense(), dtype=np.float64))
Xclr = (_l - _l.mean(axis=1, keepdims=True)).astype(np.float32)

genes_perturbed = sorted(set(pert[~is_ctrl]))
print(f"n KO target genes available: {len(genes_perturbed)}")

# raw-count diagnostics for the two checkpoint markers (sanity: PD-1 high on T cells, PD-L1 low)
raw = np.asarray(X_counts.todense())
print(f"\nRAW surface ADT diagnostics (the cells used):")
for nm, j in [("PD-L1/CD274", jpdl1), ("PD-1/CD279", jpd1)]:
    col = raw[:, j]
    print(f"  {nm:12s}: total counts={col.sum():.0f}  cells>0={int((col > 0).sum())}/{len(col)} "
          f"({100*(col > 0).mean():.1f}%)  mean/cell={col.mean():.3f}  median nonzero={np.median(col[col>0]) if (col>0).any() else 0:.1f}")


# ---------------------------------------------------------------------------------------------------
# Held-KO recovery, identical to Frangieh c4_per_marker
# ---------------------------------------------------------------------------------------------------
def held_ko_fraction(genes, frac, seed=0):
    rng = np.random.default_rng(seed)
    g = sorted(genes)
    k = max(1, int(round(frac * len(g))))
    return sorted(rng.choice(g, size=k, replace=False).tolist())


rows_rec = []
MARKER_ALIAS = {PDL1_COL: "PD-L1 (CD274)", PD1_COL: "PD-1 (CD279)"}

for frac, lbl in [(0.25, "25"), (0.50, "50")]:
    held = set(held_ko_fraction(genes_perturbed, frac, seed=SEED))
    # train = all controls + all NON-held KO cells; held KO cells are the test
    is_held = np.array([(not c) and (p in held) for p, c in zip(pert, is_ctrl)])
    train_mask = ~is_held
    # cell-mean shift: training treated mean (non-held KO cells) - control mean
    tr_treated = train_mask & (~is_ctrl)
    ctrl_mask = is_ctrl  # controls are in train by construction
    treated_mean = Xn[tr_treated].mean(0)
    ctrl_mean = Xn[ctrl_mask].mean(0)
    pred_delta = treated_mean - ctrl_mean   # one global predicted Δ

    # per held-KO observed Δ
    held_genes_sorted = sorted(held)
    obs_delta = np.vstack([Xn[(pert == s) & is_held].mean(0) - ctrl_mean for s in held_genes_sorted])
    n_held = len(held_genes_sorted)

    for j, mk in enumerate(markers):
        o = obs_delta[:, j]
        pj = float(pred_delta[j])
        signmatch = float(np.mean(np.sign(o) == np.sign(pj))) if pj != 0 else np.nan
        sd = float(o.std())
        sem = sd / np.sqrt(n_held)
        mean = float(o.mean())
        rows_rec.append({
            "marker": mk,
            "alias": MARKER_ALIAS.get(mk, mk),
            "held_frac_pct": int(lbl),
            "n_held_KO": int(n_held),
            "predDelta": pj,
            "obsDelta_mean": mean,
            "obsDelta_sd": sd,
            "abs_err": abs(mean - pj),
            "sign_match_frac": signmatch,
            "sem": sem,
            "ci_lo": mean - Z * sem,
            "ci_hi": mean + Z * sem,
            "straddles_zero": (mean - Z * sem <= 0 <= mean + Z * sem),
            "effect_sd_units": abs(mean) / sd if sd > 0 else np.nan,
            "z_vs_zero": mean / sem if sem > 0 else np.nan,
        })

rec = pd.DataFrame(rows_rec)
rec.to_csv(OUT_NEW / "chen_cite_marker_recovery.csv", index=False)
rec.to_csv(OUT_PAPER / "chen_surface_marker_CIs.csv", index=False)


def grab(marker, frac):
    return rec[(rec["marker"] == marker) & (rec["held_frac_pct"] == frac)].iloc[0]


summary = {"dataset": "Chen E-GEAD-648 (Perturb-icCITE-seq, primary human CD4+ Treg, CRISPR-KO)",
           "modality": "SURFACE ADT (CITE) — PD-L1 and PD-1 BOTH on the surface panel",
           "n_cells": int(X_counts.shape[0]), "n_surface_markers": int(X_counts.shape[1]),
           "n_KO_targets": len(genes_perturbed)}
print("\n=== Surface-marker held-KO recovery (Chen replication) ===")
for frac in (25, 50):
    pdl1 = grab(PDL1_COL, frac)
    pd1 = grab(PD1_COL, frac)
    summary[f"frac{frac}"] = {
        "PD-L1_CD274": {
            "obs_mean": float(pdl1["obsDelta_mean"]), "sd": float(pdl1["obsDelta_sd"]),
            "sem": float(pdl1["sem"]), "ci": [float(pdl1["ci_lo"]), float(pdl1["ci_hi"])],
            "straddles_zero": bool(pdl1["straddles_zero"]),
            "effect_sd_units": float(pdl1["effect_sd_units"]), "z_vs_zero": float(pdl1["z_vs_zero"]),
            "sign_match_frac": float(pdl1["sign_match_frac"]), "n_held_KO": int(pdl1["n_held_KO"]),
        },
        "PD-1_CD279": {
            "obs_mean": float(pd1["obsDelta_mean"]), "sd": float(pd1["obsDelta_sd"]),
            "sem": float(pd1["sem"]), "ci": [float(pd1["ci_lo"]), float(pd1["ci_hi"])],
            "straddles_zero": bool(pd1["straddles_zero"]),
            "effect_sd_units": float(pd1["effect_sd_units"]), "z_vs_zero": float(pd1["z_vs_zero"]),
            "sign_match_frac": float(pd1["sign_match_frac"]), "n_held_KO": int(pd1["n_held_KO"]),
        },
    }
    s = summary[f"frac{frac}"]
    print(f"\n-- holdout {frac}% (n_held_KO={s['PD-L1_CD274']['n_held_KO']}) --")
    for nm in ("PD-L1_CD274", "PD-1_CD279"):
        d = s[nm]
        print(f"  {nm:12s} Δ={d['obs_mean']:+.4f}  95%CI=[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]"
              f"  straddles0={d['straddles_zero']}  |eff|={d['effect_sd_units']:.3f}sd"
              f"  z={d['z_vs_zero']:+.2f}  signmatch={d['sign_match_frac']:.3f}")

# ---------------------------------------------------------------------------------------------------
# ROBUSTNESS: same recovery for the 2 checkpoints under CLR (canonical ADT transform) — is the
# PD-1 > PD-L1 surface asymmetry a normalization artifact? (It should survive both.)
# ---------------------------------------------------------------------------------------------------
def checkpoint_recovery(Xmat, frac):
    held = set(held_ko_fraction(genes_perturbed, frac, seed=SEED))
    is_held_ = np.array([(not c) and (p in held) for p, c in zip(pert, is_ctrl)])
    ctrl_mean_ = Xmat[is_ctrl].mean(0)
    held_sorted = sorted(held)
    obs = np.vstack([Xmat[(pert == s) & is_held_].mean(0) - ctrl_mean_ for s in held_sorted])
    treated_mean_ = Xmat[(~is_held_) & (~is_ctrl)].mean(0)
    pred = treated_mean_ - ctrl_mean_
    out = {}
    for nm, j in [("PD-1_CD279", jpd1), ("PD-L1_CD274", jpdl1)]:
        o = obs[:, j]; pj = float(pred[j]); sd = float(o.std()); mean = float(o.mean())
        sem = sd / np.sqrt(len(o))
        out[nm] = {
            "obs_mean": mean, "ci": [mean - Z * sem, mean + Z * sem],
            "straddles_zero": bool(mean - Z * sem <= 0 <= mean + Z * sem),
            "effect_sd_units": abs(mean) / sd if sd > 0 else float("nan"),
            "sign_match_frac": float(np.mean(np.sign(o) == np.sign(pj))) if pj != 0 else float("nan"),
        }
    return out


summary["robustness_CLR"] = {f"frac{int(l)}": checkpoint_recovery(Xclr, f) for f, l in [(0.25, "25"), (0.50, "50")]}
print("\n=== ROBUSTNESS: CLR (canonical ADT) transform, checkpoint recovery ===")
for frac in (25, 50):
    cr = summary["robustness_CLR"][f"frac{frac}"]
    print(f"  {frac}% CLR: PD-1 |eff|={cr['PD-1_CD279']['effect_sd_units']:.3f}sd "
          f"sm={cr['PD-1_CD279']['sign_match_frac']:.3f} straddles0={cr['PD-1_CD279']['straddles_zero']}  |  "
          f"PD-L1 |eff|={cr['PD-L1_CD274']['effect_sd_units']:.3f}sd "
          f"sm={cr['PD-L1_CD274']['sign_match_frac']:.3f} straddles0={cr['PD-L1_CD274']['straddles_zero']}")

# effect-size vs sign-match relationship (pooled across markers + both fracs), exactly as Frangieh
m = rec.dropna(subset=["sign_match_frac"])
r_eff, p_eff = pearsonr(m["effect_sd_units"], m["sign_match_frac"])
near_floor = rec[rec["straddles_zero"]]
recovered = rec[~rec["straddles_zero"]]
summary["assay_floor_confound"] = {
    "pearson_effect_vs_signmatch": float(r_eff), "p": float(p_eff), "n": int(len(m)),
    "mean_signmatch_near_floor": float(near_floor["sign_match_frac"].mean()),
    "mean_signmatch_recovered": float(recovered["sign_match_frac"].mean()),
    "mean_effect_near_floor": float(near_floor["effect_sd_units"].mean()),
    "mean_effect_recovered": float(recovered["effect_sd_units"].mean()),
}
print(f"\n  ASSAY-FLOOR CONFOUND (pooled, n={summary['assay_floor_confound']['n']} marker-frac points):")
print(f"    corr(|effect|, sign_match) = {r_eff:.3f}  (p={p_eff:.2e})")
print(f"    near-floor markers: mean sign_match={summary['assay_floor_confound']['mean_signmatch_near_floor']:.3f}"
      f" at |eff|={summary['assay_floor_confound']['mean_effect_near_floor']:.3f}sd")
print(f"    recovered markers:  mean sign_match={summary['assay_floor_confound']['mean_signmatch_recovered']:.3f}"
      f" at |eff|={summary['assay_floor_confound']['mean_effect_recovered']:.3f}sd")

# raw count diagnostics into summary
summary["raw_surface_diag"] = {}
for nm, j in [("PD-L1_CD274", jpdl1), ("PD-1_CD279", jpd1)]:
    col = raw[:, j]
    summary["raw_surface_diag"][nm] = {
        "total_counts": float(col.sum()), "frac_cells_positive": float((col > 0).mean()),
        "mean_per_cell": float(col.mean()),
    }


def _pyify(o):
    if isinstance(o, dict):
        return {k: _pyify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pyify(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


with open(OUT_NEW / "chen_checkpoint_replication_summary.json", "w") as fh:
    json.dump(_pyify(summary), fh, indent=2)
with open(OUT_PAPER / "chen_checkpoint_replication_summary.json", "w") as fh:
    json.dump(_pyify(summary), fh, indent=2)

print(f"\nWROTE {OUT_NEW/'chen_cite_marker_recovery.csv'}")
print(f"WROTE {OUT_NEW/'chen_checkpoint_replication_summary.json'}")
print(f"WROTE {OUT_PAPER/'chen_surface_marker_CIs.csv'}")
print(f"WROTE {OUT_PAPER/'chen_checkpoint_replication_summary.json'}")
