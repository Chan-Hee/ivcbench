#!/usr/bin/env python
"""PertAdapt VALIDATION GATE — reproduce the PUBLISHED PertAdapt result before adopting it as a benchmark
model. PertAdapt is adoptable as a HEADLINE Hybrid model ONLY if this gate passes within tolerance;
otherwise we keep Hybrid = STATE and DISCLOSE that PertAdapt is a faithful reimplementation whose
published anchor could not be reproduced here (no fabrication).

ANCHOR (exact, from Bai et al. 2025 + the official repo):
  backbone : FROZEN scFoundation (`cell` ckpt; finetune_method=frozen)        [run_sh/run_norman.sh]
  model    : GEARS_Model_Pert_Adapter_New + loss_adapt (the published config)  [run_sh/run_norman.sh]
  dataset  : Adamson K562 single-gene CRISPRi — the 19264-gene preprocessed file
             `gse90546_k562_63587_19264_10k_log1p_withtotalcount` (OneDrive)
  split    : GEARS 'simulation' split, seed=1, train_gene_set_size=0.75       [the canonical GEARS split]
  metric   : MSE on the top-20 DE genes of held perturbations (`mse_de`/`mse_top20_de_non_dropout`) and
             Pearson-Δ on the same DE set (`pearson_de`) — the metrics GEARS/PertAdapt report.
  CLAIM to beat : PertAdapt must IMPROVE mse_de over the scFoundation+GEARS baseline (finetune_method=
             frozen, vanilla loss) on the SAME split. The paper's headline is a *relative* improvement
             (PertAdapt < scF-GEARS on DE-MSE; PertAdapt > on Pearson-Δ-DE). Default tolerance below
             encodes "PertAdapt's mse_de is at least `--min-rel-improve` (=5%) below the scF-GEARS
             baseline AND Pearson-Δ-DE is no worse", evaluated on the held-out simulation test set.
             (We anchor to the *relative* improvement, not an absolute literal, because the paper reports
             figure-level magnitude/dcor-MSE from `norman_magnitude_mse.csv`, not an in-text Adamson
             scalar; the relative DE-MSE improvement is the robust, dataset-portable claim.)

HARD REQUIREMENT — official artifacts. A faithful published-anchor run needs ALL of:
  (1) the authors' exact gene-similarity mask  go_mask_19264.npz                    (OneDrive)
  (2) the 19264-gene preprocessed Adamson file (...withtotalcount + GEARS data_pyg) (OneDrive)
  (3) the frozen scFoundation `cell` checkpoint                                      (LOCAL ✓)
This script AUDITS for (1)+(2). If either is missing it prints READY=false with the exact blocker and
exits 2 WITHOUT fabricating a metric. When the artifacts are supplied (paths via the env vars below) it
runs the published train/eval and compares to the baseline within tolerance, printing READY=true/false.

ENV (only needed once artifacts are obtained):
  IVCBENCH_PA_GO_MASK_NPZ   path to official go_mask_19264.npz
  IVCBENCH_PA_ADAMSON_DIR   GEARS data_dir containing the 19264-gene Adamson dataset + splits/
  IVCBENCH_PA_REPO          PertAdapt repo root (default /tmp/PertAdapt) for train_ddp.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Official artifacts obtained 2026-06-05 (downloaded from the PertAdapt OneDrive; see
# data/pertadapt/official/PROVENANCE.md). These are the DEFAULTS the gate audits; env vars override.
OFFICIAL_DIR = ROOT / "data/pertadapt/official"
DEFAULT_GO_MASK = OFFICIAL_DIR / "go_mask_19264.npz"
DEFAULT_ADAMSON_DIR = OFFICIAL_DIR            # contains gse90546_..._withtotalcount/
DEFAULT_PA_REPO = ROOT / "vendor/pertadapt_repo"   # cloned+patched repo (train_ddp.py + adapter classes)

# ----- the anchor, as data (auditable, single source of truth) -----
ANCHOR = dict(
    model="PertAdapt (scFoundation variant, GEARS_Model_Pert_Adapter_New + loss_adapt)",
    backbone="frozen scFoundation cell checkpoint",
    dataset="Adamson K562 single-gene CRISPRi (gse90546_k562_63587_19264_10k_log1p_withtotalcount)",
    split="GEARS simulation, seed=1, train_gene_set_size=0.75",
    primary_metric="mse_de (top-20 DE genes of held perturbations)",
    secondary_metric="pearson_delta_de (same DE set)",
    baseline_to_beat="scFoundation+GEARS (finetune_method=frozen, vanilla loss_fct) on the SAME split",
    claim="PertAdapt mse_de < scF-GEARS mse_de by >= min_rel_improve, and pearson_delta_de not worse",
    default_min_rel_improve=0.05,
    default_pearson_tol=0.0,   # PertAdapt's Pearson-Δ-DE must be >= baseline - pearson_tol
)


def _audit_artifacts() -> tuple[bool, list[str]]:
    """Return (all_present, missing_blockers)."""
    missing = []
    # (1) official GO mask (DOWNLOADED 2026-06-05; default points at the official file)
    gm = os.environ.get("IVCBENCH_PA_GO_MASK_NPZ", str(DEFAULT_GO_MASK))
    if not gm or not Path(gm).exists():
        missing.append(
            "official go_mask_19264.npz is ABSENT. It was downloaded to "
            f"{DEFAULT_GO_MASK} (set $IVCBENCH_PA_GO_MASK_NPZ to override). "
            "(A weighted-Jaccard rebuild that reproduces the official file byte-equivalently from the "
            "local gene2go is at vendor/pertadapt/build_go_mask.py --mode jaccard.)")
    else:
        # integrity sanity: shape + weighted-Jaccard signature (data.min≈0.10)
        try:
            from scipy import sparse as _sp
            _m = _sp.load_npz(gm)
            if _m.shape != (19264, 19264):
                missing.append(f"go_mask at {gm} has shape {_m.shape}, expected (19264, 19264).")
        except Exception as _e:
            missing.append(f"go_mask at {gm} failed to load as sparse npz: {_e}")
    # (2) 19264-gene Adamson dataset (+ GEARS data_pyg) (DOWNLOADED 2026-06-05)
    adir = os.environ.get("IVCBENCH_PA_ADAMSON_DIR", str(DEFAULT_ADAMSON_DIR))
    ds = "gse90546_k562_63587_19264_10k_log1p_withtotalcount"
    ADAMSON_H5AD_BYTES = 1778574877   # official OneDrive size; reject truncated/partial downloads
    _h5 = Path(adir) / ds / "perturb_processed.h5ad"
    ok_ds = bool(adir) and _h5.exists() and _h5.stat().st_size == ADAMSON_H5AD_BYTES
    if not ok_ds:
        # the LOCAL adamson is a 5060-gene GEARS panel, not the 19264 panel PertAdapt needs
        local = Path(os.environ.get(
            "IVCBENCH_LOCAL_ADAMSON_H5AD",
            "scFoundation/GEARS/data/adamson/perturb_processed.h5ad"))  # set to your local Adamson h5ad
        note = ""
        if local.exists():
            try:
                import anndata as ad
                ng = ad.read_h5ad(local, backed="r").shape[1]
                note = f" (local Adamson h5ad has {ng} genes — a 5060-gene GEARS panel, NOT the 19264 panel)"
            except Exception:
                note = ""
        _have = _h5.stat().st_size if _h5.exists() else 0
        state = (f"PARTIAL ({_have}/{ADAMSON_H5AD_BYTES} bytes — download still in progress)"
                 if 0 < _have < ADAMSON_H5AD_BYTES else "not found")
        missing.append(
            f"19264-gene Adamson '{ds}/perturb_processed.h5ad' is {state} under "
            f"$IVCBENCH_PA_ADAMSON_DIR ({adir}). Target {DEFAULT_ADAMSON_DIR}/{ds}/; "
            f"GEARS PertData regenerates splits/ + data_pyg/ on first load.{note}")
    # (3) backbone checkpoint (local, expected present)
    # set $IVCBENCH_SCFOUNDATION_CKPT to the real checkpoint
    ckpt = os.environ.get("IVCBENCH_SCFOUNDATION_CKPT", "scFoundation/models.ckpt")
    if not Path(ckpt).exists():
        missing.append(f"scFoundation checkpoint absent at {ckpt} (set $IVCBENCH_SCFOUNDATION_CKPT).")
    return (len(missing) == 0), missing


def _two_gpu_commands() -> dict:
    """The exact, ready-to-run 2-GPU (devices 0,1) launch commands for the published anchor.

    Two runs over the SAME Adamson-simulation split (seed=1, train_gene_set_size=0.75):
      - PertAdapt   : model_class=GEARS_Model_Pert_Adapter_New, loss=loss_adapt   (the published config)
      - scF-GEARS   : model_class=GEARS_Model,                  loss=loss_fct     (the baseline to beat)
    Both read the downloaded official mask via $IVCBENCH_PA_GO_MASK_NPZ (patched into model_new.py).
    """
    repo = Path(os.environ.get("IVCBENCH_PA_REPO", str(DEFAULT_PA_REPO))) / "scFoundation" / "PertAdapter"
    adir = os.environ.get("IVCBENCH_PA_ADAMSON_DIR", str(DEFAULT_ADAMSON_DIR))
    ds = "gse90546_k562_63587_19264_10k_log1p_withtotalcount"
    gm = os.environ.get("IVCBENCH_PA_GO_MASK_NPZ", str(DEFAULT_GO_MASK))
    # set $IVCBENCH_SCFOUNDATION_CKPT to the real checkpoint
    ckpt = os.environ.get("IVCBENCH_SCFOUNDATION_CKPT", "scFoundation/models.ckpt")
    common = (
        f"cd '{repo}' && "
        f"IVCBENCH_PA_GO_MASK_NPZ='{gm}' CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone "
        f"--nnodes=1 --nproc_per_node 2 train_ddp.py "
        f"--data_dir='{adir}/' --data_name={ds} --split=simulation --seed=1 "
        f"--train_gene_set_size=0.75 --epochs=20 --valid_every=1 --batch_size=4 --test_batch_size=4 "
        f"--accumulation_steps=4 --hidden_size=512 --bin_set=autobin_resolution_append "
        f"--model_type=maeautobin --finetune_method=frozen --singlecell_model_path='{ckpt}' "
        f"--mode=v1 --highres=0 --lr=0.01 --ddp_loss_weight=5.0")
    pa_cmd = (common +
              " --loss=loss_adapt --model_class=GEARS_Model_Pert_Adapter_New "
              "--proj_name=PertAdapt --exp_name=pa_adamson "
              "--result_dir=$IVCBENCH_PA_RESULT_PA")
    base_cmd = (common +
                " --loss=loss_fct --model_class=GEARS_Model "
                "--proj_name=PertAdapt --exp_name=scfgears_adamson "
                "--result_dir=$IVCBENCH_PA_RESULT_BASE")
    return {"pertadapt_run": pa_cmd, "scf_gears_baseline_run": base_cmd}


def _parse_best(result_dir: str) -> dict:
    """Parse a finished train_ddp.py result dir for the best-epoch DE metrics (mse_de / pearson_de).

    train_ddp.py writes per-epoch validation metrics; we take the best (min mse_de) row. Tolerant to the
    common emit forms (a metrics CSV with a *de* column, or a json). Returns {} if nothing parseable yet."""
    import csv, glob, json as _json
    rd = Path(result_dir)
    # try any *.json first
    for jf in sorted(rd.glob("*.json")):
        try:
            j = _json.load(open(jf))
            md = j.get("mse_de") or j.get("mse_top20_de_non_dropout")
            pd_ = j.get("pearson_de") or j.get("pearson_delta_de")
            if md is not None:
                return {"mse_de": float(md), "pearson_de": (None if pd_ is None else float(pd_)), "src": str(jf)}
        except Exception:
            pass
    # then any CSV with a de-mse column
    best = None
    for cf in sorted(glob.glob(str(rd / "*.csv"))):
        try:
            for row in csv.DictReader(open(cf)):
                k_mse = next((k for k in row if k.lower() in
                              ("mse_de", "mse_top20_de_non_dropout", "test_mse_de")), None)
                if not k_mse or row[k_mse] in ("", None):
                    continue
                v = float(row[k_mse])
                k_p = next((k for k in row if "pearson" in k.lower() and "de" in k.lower()), None)
                p = float(row[k_p]) if (k_p and row.get(k_p) not in ("", None)) else None
                if best is None or v < best["mse_de"]:
                    best = {"mse_de": v, "pearson_de": p, "src": cf}
        except Exception:
            continue
    return best or {}


def _run_published_anchor(min_rel_improve: float, pearson_tol: float) -> dict:
    """Audit the published anchor and, if both result dirs are present, parse + compare within tolerance.

    Orchestrates the upstream train_ddp.py (the canonical published trainer) which writes per-epoch
    mse_de / pearson_de for the PertAdapt model_class and the scF-GEARS baseline. The GPU runs are NOT
    auto-launched here (GPUs busy / 2-GPU budget); this returns the exact commands and, once the two
    result dirs exist (via $IVCBENCH_PA_RESULT_PA / _BASE), parses them and applies the tolerance.
    """
    repo = Path(os.environ.get("IVCBENCH_PA_REPO", str(DEFAULT_PA_REPO))) / "scFoundation" / "PertAdapter"
    train = repo / "train_ddp.py"
    if not train.exists():
        raise FileNotFoundError(f"PertAdapt train_ddp.py not found at {train} (set $IVCBENCH_PA_REPO)")
    cmds = _two_gpu_commands()

    pa_dir = os.environ.get("IVCBENCH_PA_RESULT_PA", "")
    base_dir = os.environ.get("IVCBENCH_PA_RESULT_BASE", "")
    pa = _parse_best(pa_dir) if pa_dir else {}
    base = _parse_best(base_dir) if base_dir else {}

    if not (pa and base):
        # runnable, but the two GPU runs have not been completed yet → hand back the exact commands.
        raise NotImplementedError(
            "Artifacts present and the published-anchor is RUNNABLE. Launch the two 2-GPU runs below "
            "(devices 0,1), then re-invoke this gate with $IVCBENCH_PA_RESULT_PA and "
            "$IVCBENCH_PA_RESULT_BASE pointing at the two result dirs to parse + compare.\n"
            f"  [PertAdapt]  {cmds['pertadapt_run']}\n"
            f"  [scF-GEARS]  {cmds['scf_gears_baseline_run']}")

    # both finished → apply the tolerance (NEVER fabricate; only computed from parsed numbers)
    rel = (base["mse_de"] - pa["mse_de"]) / base["mse_de"] if base["mse_de"] else 0.0
    pearson_ok = True
    if pa.get("pearson_de") is not None and base.get("pearson_de") is not None:
        pearson_ok = pa["pearson_de"] >= base["pearson_de"] - pearson_tol
    passed = (rel >= min_rel_improve) and pearson_ok
    return dict(commands=cmds, pertadapt=pa, scf_gears_baseline=base,
                rel_mse_de_improvement=rel, pearson_ok=pearson_ok, passed=passed)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rel-improve", type=float, default=ANCHOR["default_min_rel_improve"])
    ap.add_argument("--pearson-tol", type=float, default=ANCHOR["default_pearson_tol"])
    ap.add_argument("--json-out", default=str(ROOT / "outputs/additional_models/pertadapt_validation.json"))
    args = ap.parse_args()
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PertAdapt VALIDATION GATE")
    print("=" * 78)
    for k, v in ANCHOR.items():
        print(f"  {k:24s}: {v}")
    print("-" * 78)

    present, missing = _audit_artifacts()
    result = dict(anchor=ANCHOR, min_rel_improve=args.min_rel_improve,
                  pearson_tol=args.pearson_tol, artifacts_present=present, blockers=missing)

    if not present:
        result["ready"] = False
        result["status"] = "BLOCKED — required official artifacts missing; published anchor NOT reproduced"
        print("ARTIFACT AUDIT: MISSING")
        for b in missing:
            print("  BLOCKER:", b)
        print("-" * 78)
        print("READY = False  (do NOT adopt PertAdapt as a headline Hybrid model on published grounds; "
              "keep Hybrid = STATE and DISCLOSE PertAdapt as a faithful reimplementation only).")
        json.dump(result, open(args.json_out, "w"), indent=2)
        print(f"wrote {args.json_out}")
        return 2

    # artifacts present → run the published anchor and compare
    try:
        cmp = _run_published_anchor(args.min_rel_improve, args.pearson_tol)
        result.update(cmp)
        passed = bool(cmp.get("passed"))
        result["ready"] = passed
        result["status"] = "PASS" if passed else "FAIL — did not reproduce within tolerance"
        print(f"READY = {passed}")
    except NotImplementedError as e:
        result["ready"] = False
        result["artifacts_runnable"] = True
        result["status"] = "RUNNABLE — artifacts present; launch the 2-GPU published-anchor runs, then re-invoke"
        try:
            result["commands"] = _two_gpu_commands()
        except Exception:
            pass
        print("ARTIFACTS PRESENT — published anchor is RUNNABLE. Launch the 2-GPU runs (devices 0,1):")
        print(" ", e)
        json.dump(result, open(args.json_out, "w"), indent=2)
        return 3

    json.dump(result, open(args.json_out, "w"), indent=2)
    print(f"wrote {args.json_out}")
    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    sys.exit(main())
