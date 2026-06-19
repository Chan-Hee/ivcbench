#!/usr/bin/env python3
"""Re-wrap bespoke C2 (Soskic) + C4 (Frangieh fills) into framework results_raw schema,
then assemble cross-cluster HEADLINE + within-family CONSISTENCY tables.

HARD RULES (PREREGISTRATION.md):
  - Universal floor = {cell-mean, linear-PCA} (NOT cluster-specific donor-shift/FP-ridge).
  - 3 axes: response-direction (pearson_delta, headline), distributional (e_distance), immune-program.
  - Use ONLY real already-computed results. No fabrication. Missing -> say so, skip.
  - CPU-only. No GPU jobs.

Outputs (results/_paper/, all NEW files, do NOT touch results_section.md):
  - results_raw_C2_rewrapped.csv      (framework C2 per-donor rows: CellOT/STATE/CPA + floor + scGen)
  - results_raw_C4_fills_rewrapped.csv (framework C4 RNA fills: CellOT/CPA/scPRAM/STATE/GEARS/AttnPert)
  - cross_cluster_headline.csv / .md  (task x family: response-direction delta vs universal floor)
  - within_family_consistency.csv / .md
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results", "_paper")
AM = os.path.join(ROOT, "outputs", "additional_models")
os.makedirs(OUT, exist_ok=True)

FAMILY = {  # model -> family (per PREREG headline contrasts)
    "cell-mean": "simple", "linear-PCA": "simple", "ctrl-pred": "simple", "donor-shift": "simple",
    "scGen": "latent", "CPA": "latent", "chemCPA": "chemistry", "FP-ridge": "chemistry",
    "scGPT": "foundation", "scFoundation": "foundation",
    "GEARS": "graph", "AttentionPert": "graph", "AttnPert": "graph",
    "STATE": "hybrid", "PertAdapt": "hybrid",
    "CellOT": "ot", "scPRAM": "ot", "CINEMA-OT": "ot",
}

# ----------------------------------------------------------------------------
# (1) RE-WRAP C2 SOSKIC bespoke CSVs -> framework per-donor rows
# ----------------------------------------------------------------------------
def pivot_bespoke_soskic(path, model, scorecol):
    """Long-on-metric per-donor bespoke -> wide-on-metric per-donor rows."""
    df = pd.read_csv(path)
    rows = []
    for donor, g in df.groupby("donor"):
        rec = {"donor": donor, "model": model, "family": FAMILY[model],
               "split": f"C2_soskic_LODO_{donor}"}
        for _, r in g.iterrows():
            rec[r["metric"]] = r[scorecol]
        rec["n_test"] = int(g["n_test"].iloc[0])
        rec["n_strata"] = int(g["n_strata"].iloc[0])
        rec["n_response_genes"] = int(g["n_response_genes"].iloc[0])
        rec["leak_free"] = bool(g["leak_free"].iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)

def assemble_c2():
    # universal floor + scGen come from the canonical framework-shaped axis file
    axis = pd.read_csv(os.path.join(ROOT, "results", "C2", "soskic_donor_axis.csv"))
    floor = axis[axis["model"].isin(["cell-mean", "linear-PCA", "ctrl-pred", "donor-shift", "scGen"])].copy()
    floor = floor.rename(columns={"aucell_delta_score": "aucell_program_corr"})
    keep = ["model", "family", "split", "donor", "pearson_delta", "e_distance",
            "aucell_program_corr", "n_test", "n_strata", "n_response_genes", "leak_free"]
    floor = floor[keep]

    parts = [floor]
    # CellOT (done, 3-seed seed0 point in cellot_score)
    parts.append(pivot_bespoke_soskic(os.path.join(AM, "cellot_soskic_raw_3seed.csv"),
                                      "CellOT", "cellot_score").rename(
        columns={"aucell_delta_score": "aucell_program_corr"}))
    # STATE (2 shards)
    for f in sorted(glob.glob(os.path.join(AM, "state_soskic_shard*.csv"))):
        parts.append(pivot_bespoke_soskic(f, "STATE", "state_score").rename(
            columns={"aucell_delta_score": "aucell_program_corr"}))
    # CPA (2 shards)
    for f in sorted(glob.glob(os.path.join(AM, "cpa_soskic_shard*.csv"))):
        parts.append(pivot_bespoke_soskic(f, "CPA", "cpa_score").rename(
            columns={"aucell_delta_score": "aucell_program_corr"}))

    c2 = pd.concat(parts, ignore_index=True)
    c2["cluster"] = "C2"; c2["dataset"] = "soskic_CD4_activation"; c2["modality"] = "RNA"
    c2["registry_task"] = "C2_LODO"; c2["seed"] = 0; c2["ran"] = True
    c2["headline_eligible"] = c2["family"].ne("ot")  # OT floors not "conditioned"
    # scPRAM-Soskic: PENDING (GPU job in progress, partial donor coverage). Emit a marked stub.
    pend = pd.DataFrame([{
        "model": "scPRAM", "family": "ot", "split": "C2_soskic_LODO_PENDING",
        "donor": "PENDING", "pearson_delta": np.nan, "e_distance": np.nan,
        "aucell_program_corr": np.nan, "n_test": np.nan, "n_strata": np.nan,
        "n_response_genes": np.nan, "leak_free": np.nan, "cluster": "C2",
        "dataset": "soskic_CD4_activation", "modality": "RNA", "registry_task": "C2_LODO",
        "seed": 0, "ran": False, "headline_eligible": False,
    }])
    c2 = pd.concat([c2, pend], ignore_index=True)
    c2.to_csv(os.path.join(OUT, "results_raw_C2_rewrapped.csv"), index=False)
    return c2

# ----------------------------------------------------------------------------
# (2) RE-WRAP C4 FRANGIEH night fills -> framework rows (RNA, two modality_lo_ko folds)
# ----------------------------------------------------------------------------
def assemble_c4():
    rows = []
    def base(model, family, split, frac, ran, leak, ntr, nte, nstr,
             pd_, edist, pd_on=np.nan, auc=np.nan, action="run_headline"):
        return {"baseline": model, "family": family, "split": split,
                "registry_task": "C4_Axis2", "action": action,
                "headline_eligible": family not in ("ot",),
                "seed": 0, "ran": ran, "leak_free": leak,
                "n_train": ntr, "n_test": nte, "n_test_strata": nstr,
                "pearson_delta": pd_, "pearson_delta_lo": np.nan, "pearson_delta_hi": np.nan,
                "pearson_delta_ontarget": pd_on, "e_distance": edist,
                "e_distance_lo": np.nan, "e_distance_hi": np.nan,
                "aucell_program_corr": auc, "cluster": "C4",
                "dataset": "frangieh_rna", "modality": "RNA", "frac_held": frac}

    # CellOT g2/g3 (OT floor)
    for f, frac in [("cellot_frangieh_g2.csv", 25), ("cellot_frangieh_g3.csv", 50)]:
        d = pd.read_csv(os.path.join(AM, f)).iloc[0]
        rows.append(base("CellOT", "ot", d["split"], frac, bool(d["ran"]), bool(d["leak_free"]),
                         int(d["n_train"]), int(d["n_test"]), int(d["n_test_strata"]),
                         d["pearson_delta"], d["e_distance"], action="run_floor"))
    # CPA g2/g3 (latent)
    for f, frac in [("cpa_frangieh_g2.csv", 25), ("cpa_frangieh_g3.csv", 50)]:
        d = pd.read_csv(os.path.join(AM, f)).iloc[0]
        rows.append(base("CPA", "latent", d["split"], frac, bool(d["ran"]), bool(d["leak_free"]),
                         int(d["n_train"]), int(d["n_test"]), int(d["n_test_strata"]),
                         d["pearson_delta"], d["e_distance"],
                         pd_on=d.get("pearson_delta_ontarget", np.nan),
                         auc=d.get("aucell_program_corr", np.nan)))
    # scPRAM g2/g3 (OT floor; not headline-eligible on C4 per applicability_note)
    for f, frac in [("scpram_frangieh_g2.csv", 25), ("scpram_frangieh_g3.csv", 50)]:
        d = pd.read_csv(os.path.join(AM, f)).iloc[0]
        rows.append(base("scPRAM", "ot", d["split"], frac, bool(d["ran"]), bool(d["leak_free"]),
                         int(d["n_train"]), int(d["n_test"]), int(d["n_test_strata"]),
                         d["pearson_delta"], d["e_distance"], action="run_floor"))
    # STATE g2/g3 (hybrid)
    sd = pd.read_csv(os.path.join(AM, "state_frangieh_raw.csv"))
    for _, d in sd.iterrows():
        frac = int(d["frac_held"])
        rows.append(base("STATE", "hybrid", d["split"], frac, bool(d["ran"]), bool(d["leak_free"]),
                         int(d["n_train"]), int(d["n_test"]), int(d["n_test_strata"]),
                         d["pearson_delta"], d["e_distance"],
                         pd_on=d.get("pearson_delta_ontarget", np.nan)))
    # GEARS / AttentionPert + floor rows from jsonl
    with open(os.path.join(ROOT, "results", "C4", "graph_frangieh_rows.jsonl")) as fh:
        for line in fh:
            j = json.loads(line)
            if j.get("baseline") in ("GEARS", "AttentionPert"):
                split = j["split"]; frac = 25 if split.endswith("25") else 50
                rows.append(base(j["baseline"], "graph", split, frac, j["ran"], j["leak_free"],
                                 j["n_train"], j["n_test"], j["n_test_strata"],
                                 j["pearson_delta"], j["e_distance"],
                                 pd_on=j.get("pearson_delta_ontarget", np.nan),
                                 auc=j.get("aucell_program_corr", np.nan), action="run_floor"))
    c4 = pd.DataFrame(rows)
    # E-distance scale caveat: CellOT/scPRAM bespoke e_distance is the model's own
    # (MMD-/normalized) scale, NOT the framework PCA-50 E-distance used by floor/CPA/STATE/graph.
    # pearson_delta (headline axis) IS comparable across all; e_distance is comparable only
    # within the framework-scale group.
    c4["e_distance_framework_scale"] = ~c4["baseline"].isin(["CellOT", "scPRAM"])
    c4.to_csv(os.path.join(OUT, "results_raw_C4_fills_rewrapped.csv"), index=False)
    return c4

C2 = assemble_c2()
C4 = assemble_c4()
print("C2 rewrapped rows:", len(C2), "models:", sorted(C2["model"].unique()))
print("  per-model donor counts:")
print(C2[C2.donor != "PENDING"].groupby("model")["donor"].nunique())
print("C4 fills rewrapped rows:", len(C4), "baselines:", sorted(C4["baseline"].unique()))
