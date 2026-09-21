#!/usr/bin/env python3
"""Frozen-bundle, CPU-only Figure 3 program and response reanalysis.

This is the analysis behind Figure 3b/3c, Figure S6 and Supplementary Tables S7, S12 and S22: each
immune program scored as the equal-weight mean-expression shift of its measured members, applied
identically to the predicted and the observed mean profile. It reads the deposited prediction
bundles and nothing else -- no cells, no model, no fitting -- so a clone of this archive can
recompute every value in results/_paper/immune_program_revision/ on CPU in about twenty seconds.

The code is the one that produced the published numbers, with its inputs repointed at the archive
(scripts/_program_inputs.py) and its outputs at a working directory. The protocol, membership and
support masks, NA conventions and tolerances are unchanged, and the run still verifies each
bundle's sha256 and still asserts its recomputed full-census Pearson-delta against the deposited
results/_paper/census_unit_scores.csv.

    python scripts/program_analysis.py                     # recompute and compare, writes nothing
    python scripts/program_analysis.py -o work/program     # keep the rebuilt CSVs for inspection
"""
from __future__ import annotations

import argparse
import tempfile
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _program_inputs  # noqa: E402  (resolves the frozen workspace out of this archive)

BASE = Path(__file__).resolve().parents[1]
# OUT and INPUTS are bound in main(); the frozen workspace they used to name is not in the archive
OUT = None
INPUTS = None
SCALAR_TOL = 1e-12
PROFILE_TOL = 1e-4
ALIGN_ATOL = 1e-4
ALIGN_RTOL = 1e-5
EXPECTED_UNITS = {"T1": 8, "T3": 5, "T5c": 4}
FLOORS = ("cell-mean", "linear-PCA")
SCOPES = ("shared_supported_strata", "full_census_strata")
MASKS = ("benchmark_mask", "all_measured_original_members")
METHODS = ("mean_expression_delta", "rank_top5_mean_profile")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, float_format="%.17g")
    return frame


def read_programs():
    programs, sources = {}, {}
    for task, letter in (("T3", "3"), ("T5c", "5")):
        path = BASE / "src/ivcbench/clusters" / f"c{letter}.py"
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == f"C{letter}_PROGRAMS":
                    programs[task] = ast.literal_eval(node.value)
        if task not in programs:
            raise ValueError(f"Missing frozen program definitions: {path}")
        sources[str(path.relative_to(BASE))] = sha(path)
    return programs, sources


def load_inputs():
    manifest = pd.read_csv(INPUTS / "selected_bundle_manifest.csv", keep_default_na=False)
    if manifest.duplicated(["task", "model", "unit"]).any():
        raise ValueError("Duplicate selected bundle key")
    census = pd.read_csv(BASE / "results/_paper/census_uncertainty.csv")
    statuses = {(r.task_key, r.model): r.status for r in census.itertuples()}
    data, grouped = {}, defaultdict(list)
    for row in manifest.to_dict("records"):
        path = BASE / row["snapshot_path"]
        if sha(path) != row["sha256"] or row["snapshot_sha256"] != row["sha256"]:
            raise ValueError(f"Frozen bundle hash mismatch: {path}")
        key = row["task"], row["model"], row["unit"]
        with np.load(path, allow_pickle=True) as z:
            a = {k: np.asarray(z[k], dtype=np.float64) for k in
                 ("pred_means", "obs_means", "control_mean")}
            a["genes"] = z["genes"].astype(str)
            a["strata"] = z["strata"].astype(str)
            a["excluded"] = np.asarray(z["exclude_gene_idx"], int) if "exclude_gene_idx" in z else np.array([], int)
            a["declined"] = set(z["declined_strata"].astype(str)) if "declined_strata" in z else set()
        n, g = a["pred_means"].shape
        assert a["obs_means"].shape == (n, g) and a["control_mean"].shape == (g,)
        assert len(a["genes"]) == g and len(a["strata"]) == n
        assert len(set(a["genes"])) == g and len(set(a["strata"])) == n
        assert all(np.isfinite(a[k]).all() for k in ("pred_means", "obs_means", "control_mean"))
        assert a["declined"].issubset(set(a["strata"]))
        a["row"] = row
        a["status"] = "floor" if row["model"] in FLOORS else statuses[(row["task"], row["model"])]
        data[key] = a
        grouped[(key[0], key[2])].append(key)
    for task, expected in EXPECTED_UNITS.items():
        roster = sorted({m for t, m, u in data if t == task})
        for model in roster:
            units = {u for t, m, u in data if t == task and m == model}
            if len(units) != expected:
                raise ValueError(f"Incomplete final roster: {task}/{model}: {units}")
    return manifest, data, grouped


def align_inputs(data, grouped):
    alignment, supports, canonical = [], [], {}
    for (task, unit), keys in sorted(grouped.items()):
        ref = data[(task, "cell-mean", unit)]
        genes, strata = ref["genes"], ref["strata"]
        union = set().union(*(data[k]["declined"] for k in keys))
        canonical[(task, unit)] = dict(genes=genes, strata=strata,
            observed=ref["obs_means"], control=ref["control_mean"],
            excluded=ref["excluded"], union_declined=union)
        supports.append(dict(task=task, unit=unit, n_models=len(keys),
            full_n_strata=len(strata), shared_n_strata=len(strata)-len(union),
            excluded_strata=";".join(sorted(union)),
            exclusion_models=json.dumps({k[1]: sorted(data[k]["declined"]) for k in keys if data[k]["declined"]}, sort_keys=True),
            full_strata=";".join(strata),
            shared_strata=";".join(s for s in strata if s not in union)))
        for key in keys:
            a = data[key]
            if set(a["genes"]) != set(genes) or set(a["strata"]) != set(strata):
                raise ValueError(f"Different target labels: {key}")
            gene_positions = {g: i for i, g in enumerate(a["genes"])}
            stratum_positions = {s: i for i, s in enumerate(a["strata"])}
            go = np.array([gene_positions[g] for g in genes])
            so = np.array([stratum_positions[s] for s in strata])
            pred = a["pred_means"][so][:, go]
            obs = a["obs_means"][so][:, go]
            ctrl = a["control_mean"][go]
            aexcluded = set(a["genes"][a["excluded"]])
            if aexcluded != set(genes[ref["excluded"]]):
                raise ValueError(f"Different feature mask: {key}")
            od = float(np.max(np.abs(obs-ref["obs_means"])))
            cd = float(np.max(np.abs(ctrl-ref["control_mean"])))
            if not np.allclose(obs, ref["obs_means"], atol=ALIGN_ATOL, rtol=ALIGN_RTOL):
                raise ValueError(f"Unpaired observations: {key}: {od}")
            if not np.allclose(ctrl, ref["control_mean"], atol=ALIGN_ATOL, rtol=ALIGN_RTOL):
                raise ValueError(f"Unpaired controls: {key}: {cd}")
            a["aligned_pred"] = pred
            alignment.append(dict(task=task, model=key[1], unit=unit, status=a["status"],
                snapshot_path=a["row"]["snapshot_path"], sha256=a["row"]["sha256"],
                canonical_model="cell-mean", n_strata=len(strata), n_genes=len(genes),
                n_excluded_genes=len(aexcluded), max_observed_difference=od,
                max_control_difference=cd, target_and_control_used="canonical_cell_mean_bundle",
                n_explicit_declined=len(a["declined"]),
                declined_strata=";".join(sorted(a["declined"])), aligned=True))
    return canonical, alignment, supports


def protocol_record(manifest, data, canonical, programs, sources):
    masks = []
    for (task, unit), a in sorted(canonical.items()):
        genes = a["genes"]
        excluded = set(genes[a["excluded"]])
        targets = {s.split("=", 1)[-1] for s in a["strata"]} if task == "T3" else set()
        masks.append(dict(task=task, unit=unit, canonical_model="cell-mean",
            gene_order=list(genes), full_strata=list(a["strata"]),
            excluded_feature_indices=list(map(int, a["excluded"])),
            excluded_feature_genes=sorted(excluded),
            held_targets=sorted(targets),
            shared_excluded_strata=sorted(a["union_declined"]),
            program_members={name: {"listed": members,
                "measured": [g for g in genes if g in members],
                "benchmark_scored": [g for g in genes if g in members and g not in excluded],
                "removed_by_fixed_target_union": sorted(set(members)&set(genes)&targets),
                "removed_by_benchmark_mask": sorted(set(members)&excluded)}
                for name, members in programs.get(task, {}).items()}))
    return dict(protocol_version=1,
        selected_manifest_sha256=sha(INPUTS / "selected_bundle_manifest.csv"),
        frozen_gene_definition_sources=sources,
        bundle_count=len(manifest),
        primary=dict(scope=SCOPES[0], gene_mask=MASKS[0], method=METHODS[0],
            formula="mean_g(predicted_mean_g-control_mean_g) and mean_g(observed_mean_g-control_mean_g), identical fixed measured members",
            program_summary="unweighted mean of per-unit Pearson correlations on common observed-defined units; no available-case model averages",
            unit_eligibility="at least 3 shared strata, >=1 scored member, observed score SD >1e-12, observed retained-profile maximum range >=1e-4"),
        sensitivities=dict(scopes=list(SCOPES), masks=list(MASKS), methods=list(METHODS),
            rank_universe="benchmark_mask ranks within retained benchmark features; all_measured_original_members ranks within the original full bundle feature universe (legacy operator)",
            rank_cutoff="max(1, round(0.05 * ranking_universe_size)); original NumPy argsort tie convention",
            omitted_variant="per-stratum direct-target exclusion is not used because it changes program composition across conditions"),
        constants=dict(scalar_sd_tolerance=SCALAR_TOL, profile_max_range_tolerance=PROFILE_TOL,
            alignment_atol=ALIGN_ATOL, alignment_rtol=ALIGN_RTOL),
        targets="After alignment validation all candidates use the same canonical cell-mean observed/control arrays, removing tolerated float32 target accumulation differences.",
        declined="Primary shared support removes the union of explicitly declined strata across final-roster models and floors; full-scope program correlations with explicit declines remain unavailable as model performance.",
        constant_prediction="Correlation missing, never zero; raw scores and errors retained. Numerical-constant profile guard is distinct from biological absence.",
        candidate_rule="native/adapted final-census entries only; full task bundle roster required; diagnostics and floors shown separately; best is a descriptive maximum without significance claim",
        response_metrics=dict(feature_mask="benchmark mask", norm="||pred-control||_2 / ||obs-control||_2 per stratum; median within unit then median across units",
            calibration_slope="dot(pred-control,obs-control) / dot(pred-control,pred-control), through origin mapping predicted to observed; median within unit then median across units",
            full_scope="Reports stored census outputs including explicitly recorded control placeholders; preserves declined counts and is separate from shared-support model performance.",
            pearson="Pearson across retained genes per stratum, then equal-stratum and equal-unit mean; zero vector gives census convention score zero"),
        null="No inferred null band, no model-selection significance claim, no matched-gene or perturbation-label null calculated in this protocol",
        revision_timing="Metric revision selected during revision/pilot review; protocol fixed before this final frozen-bundle reanalysis, not preregistered before the study.",
        roster=[dict(task=t, model=m, status=data[(t,m,next(u for tt,mm,u in data if tt==t and mm==m))]["status"])
                for t,m in sorted({(t,m) for t,m,u in data})],
        exact_masks=masks)


def freeze_protocol(record):
    path = OUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != record:
            raise ValueError("Protocol/input change after freeze; coordinate a versioned protocol before proceeding")
    else:
        dump_json(path, record)


def rank_weights(profile):
    n, g = profile.shape
    threshold = max(1, int(round(.05*g)))
    order = np.argsort(-profile, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(g)[None, :]
    return np.maximum(threshold-ranks, 0) / threshold


def correlate(pred, obs, n_members, pred_ptp, obs_ptp):
    if n_members == 0:
        return np.nan, "no_measured_program_members"
    if len(obs) < 3:
        return np.nan, "fewer_than_three_strata_policy"
    if not np.isfinite(pred).all() or not np.isfinite(obs).all():
        return np.nan, "nonfinite_score"
    if np.std(obs) <= SCALAR_TOL:
        return np.nan, "constant_observed_score"
    if obs_ptp < PROFILE_TOL:
        return np.nan, "numerically_constant_observed_profile"
    if np.std(pred) <= SCALAR_TOL:
        return np.nan, "constant_predicted_score"
    if pred_ptp < PROFILE_TOL:
        return np.nan, "numerically_constant_predicted_profile"
    return float(np.corrcoef(pred, obs)[0, 1]), "estimable"


def profile_ptp(x):
    return float(np.ptp(x, axis=0).max()) if len(x) and x.shape[1] else 0.0


def response_score(pred, obs):
    a = pred-pred.mean(1, keepdims=True)
    b = obs-obs.mean(1, keepdims=True)
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a*b, axis=1), denominator,
                     out=np.zeros(len(a)), where=denominator >= SCALAR_TOL)


def analyze(data, canonical, programs):
    unit_rows, stratum_rows, response_units, response_strata, coverage = [], [], [], [], []
    for (task, model, unit), record in sorted(data.items()):
        a = canonical[(task, unit)]
        genes, full_strata = a["genes"], a["strata"]
        base = dict(task=task, model=model, unit=unit, status=record["status"])
        keep = np.ones(len(genes), bool)
        keep[a["excluded"]] = False
        for scope in SCOPES:
            use = np.array([s not in a["union_declined"] for s in full_strata]) if scope==SCOPES[0] else np.ones(len(full_strata), bool)
            strata = full_strata[use]
            pred, obs, ctrl = record["aligned_pred"][use], a["observed"][use], a["control"]
            declined = record["declined"] & set(strata)
            dp, do = (pred-ctrl)[:, keep], (obs-ctrl)[:, keep]
            pp = response_score(dp, do)
            pn, on = np.linalg.norm(dp, axis=1), np.linalg.norm(do, axis=1)
            ratio = np.divide(pn, on, out=np.full(len(strata), np.nan), where=on>SCALAR_TOL)
            slope = np.divide(np.sum(dp*do, axis=1), np.sum(dp*dp, axis=1),
                              out=np.full(len(strata), np.nan), where=np.sum(dp*dp, axis=1)>SCALAR_TOL)
            response_units.append(dict(**base, scope=scope, n_strata=len(strata), full_n_strata=len(full_strata),
                n_response_genes=int(keep.sum()), n_explicit_declined=len(declined),
                pearson_delta=float(np.mean(pp)), response_norm_ratio=float(np.median(ratio)) if np.isfinite(ratio).all() else np.nan,
                mean_response_norm_ratio=float(np.mean(ratio)) if np.isfinite(ratio).all() else np.nan,
                calibration_slope=float(np.median(slope)) if np.isfinite(slope).all() else np.nan,
                mean_calibration_slope=float(np.mean(slope)) if np.isfinite(slope).all() else np.nan,
                n_norm_estimable=int(np.isfinite(ratio).sum()), n_slope_estimable=int(np.isfinite(slope).sum()),
                all_response_norms_estimable=bool(np.isfinite(ratio).all()),
                includes_declined_placeholders=bool(declined)))
            for i, s in enumerate(strata):
                response_strata.append(dict(**base, scope=scope, stratum=s,
                    explicit_declined=s in declined, pearson_delta=pp[i], predicted_norm=pn[i], observed_norm=on[i],
                    response_norm_ratio=ratio[i], calibration_slope=slope[i]))
            if task not in programs:
                continue
            for gene_mask in MASKS:
                universe = np.flatnonzero(keep) if gene_mask==MASKS[0] else np.arange(len(genes))
                ug = genes[universe]
                xp, xo, xc = pred[:, universe], obs[:, universe], ctrl[universe]
                weights_pred = rank_weights(xp)
                weights_obs = rank_weights(xo)
                weights_ctrl = rank_weights(xc[None, :])[0]
                pptp, optp = profile_ptp(xp), profile_ptp(xo)
                for program, listed in programs[task].items():
                    ix = np.flatnonzero(np.isin(ug, listed))
                    n_members = len(ix)
                    if model=="cell-mean":
                        coverage.append(dict(task=task, unit=unit, scope=scope, gene_mask=gene_mask,
                            program=program, n_listed=len(listed), n_measured_original=int(np.isin(genes, listed).sum()),
                            n_scored_members=n_members, scored_members=";".join(ug[ix]),
                            listed_members=";".join(listed), removed_by_mask=";".join(g for g in genes if g in listed and g not in set(ug)),
                            n_strata=len(strata), full_n_strata=len(full_strata), ranking_universe_size=len(universe)))
                    for method in METHODS:
                        if not n_members:
                            ys = ps = np.full(len(strata), np.nan)
                            identity_error = np.nan
                        elif method==METHODS[0]:
                            ps=(xp[:,ix]-xc[ix]).mean(1)
                            ys=(xo[:,ix]-xc[ix]).mean(1)
                            identity_error=float(np.max(np.abs((xo.copy()[:,ix]-xc[ix]).mean(1)-ys)))
                        else:
                            cs=weights_ctrl[ix].mean()
                            ps=weights_pred[:,ix].mean(1)-cs
                            ys=weights_obs[:,ix].mean(1)-cs
                            identity_error=float(np.max(np.abs(rank_weights(xo.copy())[:,ix].mean(1)-cs-ys)))
                        if n_members and identity_error != 0:
                            raise ValueError("Observed identity score mismatch")
                        value, reason=correlate(ps,ys,n_members,pptp,optp)
                        raw_reason=reason
                        if declined:
                            value, reason=np.nan,"explicit_declined_strata_on_full_support"
                        observed_sd=float(np.std(ys)) if n_members else np.nan
                        observed_eligible=bool(n_members and len(strata)>=3 and np.isfinite(observed_sd)
                                               and observed_sd>SCALAR_TOL and optp>=PROFILE_TOL)
                        unit_rows.append(dict(**base,scope=scope,gene_mask=gene_mask,method=method,program=program,
                            n_strata=len(strata), full_n_strata=len(full_strata), n_measured=n_members,
                            n_listed=len(listed), measured_genes=";".join(ug[ix]),
                            correlation=value,correlation_status=reason,raw_numerical_status=raw_reason,
                            observed_eligible=observed_eligible,observed_sd=observed_sd,
                            predicted_sd=float(np.std(ps)) if n_members else np.nan,
                            observed_profile_max_range=optp,predicted_profile_max_range=pptp,
                            rmse=float(np.sqrt(np.mean((ps-ys)**2))) if n_members else np.nan,
                            mae=float(np.mean(np.abs(ps-ys))) if n_members else np.nan,
                            sign_agreement=float(np.mean(np.sign(ps)==np.sign(ys))) if n_members else np.nan,
                            identity_max_error=identity_error,n_explicit_declined=len(declined),
                            ranking_universe_size=len(universe),rank_top_n=max(1,int(round(.05*len(universe))))))
                        for j,s in enumerate(strata):
                            stratum_rows.append(dict(**base,scope=scope,gene_mask=gene_mask,method=method,
                                program=program,stratum=s,predicted_delta=ps[j],observed_delta=ys[j],
                                n_measured=n_members,explicit_declined=s in declined))
    return unit_rows,stratum_rows,response_units,response_strata,coverage


def summarize_programs(frame):
    keys=["task","scope","gene_mask","method","program"]
    selection,summaries,best=[],[],[]
    for key,g in frame.groupby(keys,sort=True):
        meta=dict(zip(keys,key))
        selected=[]
        for unit,u in g.groupby("unit",sort=True):
            for column in ["observed_eligible","n_strata","n_measured","measured_genes","observed_sd"]:
                if u[column].nunique(dropna=False)!=1:
                    raise ValueError(f"Model-dependent observed selection {key}/{unit}/{column}")
            ref=u.loc[u.model.eq("cell-mean")].iloc[0]
            if ref.observed_eligible:selected.append(unit)
            selection.append(dict(**meta,unit=unit,observed_eligible=bool(ref.observed_eligible),
                n_strata=int(ref.n_strata),n_measured=int(ref.n_measured),observed_sd=ref.observed_sd,
                observed_profile_max_range=ref.observed_profile_max_range,
                reason="eligible" if ref.observed_eligible else ref.raw_numerical_status))
        for model,m in g.groupby("model",sort=True):
            s=m[m.unit.isin(selected)]
            valid=s.correlation_status.eq("estimable") & s.correlation.notna()
            complete=bool(selected and len(s)==len(selected) and valid.all())
            summaries.append(dict(**meta,model=model,status=m.status.iloc[0],n_task_units=int(m.unit.nunique()),
                expected_task_units=EXPECTED_UNITS[meta["task"]],n_comparison_units=len(selected),
                comparison_units=";".join(selected),n_estimable=int(valid.sum()),all_selected_units_estimable=complete,
                correlation=float(s.correlation.mean()) if complete else np.nan,
                correlation_unit_min=float(s.correlation.min()) if complete else np.nan,
                correlation_unit_max=float(s.correlation.max()) if complete else np.nan,
                mean_rmse=float(s.rmse.mean()) if len(s) and s.rmse.notna().all() and not s.n_explicit_declined.any() else np.nan,
                mean_mae=float(s.mae.mean()) if len(s) and s.mae.notna().all() and not s.n_explicit_declined.any() else np.nan,
                status_counts=json.dumps(s.correlation_status.value_counts().to_dict(),sort_keys=True),
                selection_scope="common observed-defined units; identical for every candidate"))
    sm=pd.DataFrame(summaries)
    for key,g in sm.groupby(keys,sort=True):
        meta=dict(zip(keys,key));candidates=g[g.status.isin(["native","adapted"])]
        valid=candidates[candidates.correlation.notna()]
        chosen=valid.sort_values(["correlation","model"],ascending=[False,True]).iloc[0] if len(valid) else None
        best.append(dict(**meta,model=chosen.model if chosen is not None else "",
            status=chosen.status if chosen is not None else "",correlation=float(chosen.correlation) if chosen is not None else np.nan,
            correlation_unit_min=float(chosen.correlation_unit_min) if chosen is not None else np.nan,
            correlation_unit_max=float(chosen.correlation_unit_max) if chosen is not None else np.nan,
            n_conditioned_candidates=len(candidates),n_comparable_candidates=len(valid),
            n_comparison_units=int(g.iloc[0].n_comparison_units),comparison_units=g.iloc[0].comparison_units,
            expected_task_units=EXPECTED_UNITS[meta["task"]],selection="descriptive maximum; no winner inference"))
    return selection,sm,best


def summarize_responses(frame):
    rows=[]
    for (task,scope),g in frame.groupby(["task","scope"],sort=True):
        floors={m:float(g.loc[g.model.eq(m),"pearson_delta"].mean()) for m in FLOORS}
        binding=max(FLOORS,key=lambda m:floors[m])
        reference=g.loc[g.model.eq(binding)].set_index("unit").pearson_delta
        for model,m in g.groupby("model",sort=True):
            assert len(m)==EXPECTED_UNITS[task]
            margins=m.set_index("unit").pearson_delta-reference
            rows.append(dict(task=task,scope=scope,model=model,status=m.status.iloc[0],n_units=len(m),
                pearson_delta=float(m.pearson_delta.mean()),binding_floor=binding,binding_floor_score=floors[binding],
                margin=float(margins.mean()),n_units_positive=int((margins>0).sum()),
                response_norm_ratio=float(np.median(m.response_norm_ratio)) if m.response_norm_ratio.notna().all() else np.nan,
                macro_mean_response_norm_ratio=float(m.mean_response_norm_ratio.mean()) if m.mean_response_norm_ratio.notna().all() else np.nan,
                calibration_slope=float(np.median(m.calibration_slope)) if m.calibration_slope.notna().all() else np.nan,
                macro_mean_calibration_slope=float(m.mean_calibration_slope.mean()) if m.mean_calibration_slope.notna().all() else np.nan,
                n_explicit_declined=int(m.n_explicit_declined.sum()),
                full_n_strata=int(m.full_n_strata.sum()),n_strata=int(m.n_strata.sum()),
                aggregation="response: equal-stratum/equal-unit mean; norm/slope: median within unit then median across units"))
    return pd.DataFrame(rows)


def legacy_magnitude(data):
    """Reconstruct existing S22's exact nested-median estimator on source arrays."""
    rows=[]
    for (task,model,unit),a in sorted(data.items()):
        if task!="T5c":continue
        dp=a["pred_means"]-a["control_mean"]
        do=a["obs_means"]-a["control_mean"]
        observed_norm=np.linalg.norm(do,axis=1)
        keep=observed_norm>0
        dp,do=dp[keep],do[keep]
        ratio=np.linalg.norm(dp,axis=1)/np.linalg.norm(do,axis=1)
        slope=np.array([(o@p)/(p@p) if p@p>SCALAR_TOL else np.nan for p,o in zip(dp,do)])
        rows.append(dict(task=task,model=model,unit=unit,status=a["status"],
            response_norm_ratio=float(np.nanmedian(ratio)),calibration_slope=float(np.nanmedian(slope)) if np.isfinite(slope).any() else np.nan,
            mean_response_norm_ratio=float(np.mean(ratio)),mean_defined_calibration_slope=float(np.nanmean(slope)) if np.isfinite(slope).any() else np.nan,
            n_strata=len(keep),n_norm_defined=int(keep.sum()),n_slope_defined=int(np.isfinite(slope).sum()),n_explicit_declined=len(a["declined"])))
    units=write_csv("legacy_full_support_magnitude_units.csv",rows)
    result=[]
    for model,m in units.groupby("model",sort=True):
        result.append(dict(task="T5c",model=model,status=m.status.iloc[0],scope="full_census_source_arrays",
            response_norm_ratio=float(np.median(m.response_norm_ratio)),calibration_slope=float(np.median(m.calibration_slope)),
            macro_mean_response_norm_ratio=float(m.mean_response_norm_ratio.mean()),
            macro_mean_defined_calibration_slope=float(m.mean_defined_calibration_slope.mean()),
            n_lineages=len(m),n_explicit_declined=int(m.n_explicit_declined.sum()),
            n_norm_defined=int(m.n_norm_defined.sum()),n_slope_defined=int(m.n_slope_defined.sum()),
            estimator="exact S22 reproduction: original per-bundle targets and controls; nanmedian over strata, median over lineages; declined placeholders retained"))
    return write_csv("legacy_full_support_magnitude.csv",result)


def add_error_comparators(frame):
    keys=["task","unit","scope","gene_mask","method","program"]
    result=[]
    for key,g in frame.groupby(keys,sort=True):
        floors=g[g.model.isin(FLOORS)]
        assert set(floors.model)==set(FLOORS)
        ref=float(floors.rmse.min()) if floors.rmse.notna().all() else np.nan
        for r in g.to_dict("records"):
            r.update(best_unit_floor_rmse=ref,rmse_improvement_vs_unit_floor=ref-r["rmse"],
                rmse_skill_vs_unit_floor=1-r["rmse"]/ref if np.isfinite(ref) and ref>SCALAR_TOL else np.nan,
                floor_definition="smaller RMSE of cell-mean and linear-PCA, chosen per unit/program; auxiliary error comparison")
            result.append(r)
    return pd.DataFrame(result)


def add_degeneracy_flags(frame):
    """Report numerical degeneracy independently of comparative-unit policy.

    The current primary correlation requires three strata before any numerical
    classification. Legacy audit counts instead assign constant-observed and
    constant-predicted causes before applying the two-stratum policy. Preserve
    both views without changing a score or a selected comparison unit.
    """
    frame=frame.copy()
    frame["fewer_than_three_strata"]=frame.n_strata.lt(3)
    frame["observed_scalar_constant"]=frame.observed_sd.le(SCALAR_TOL)
    frame["predicted_scalar_constant"]=frame.predicted_sd.le(SCALAR_TOL)
    frame["observed_profile_numerically_constant"]=frame.observed_profile_max_range.lt(PROFILE_TOL)
    frame["predicted_profile_numerically_constant"]=frame.predicted_profile_max_range.lt(PROFILE_TOL)
    def legacy_status(r):
        if not r.n_measured:return "no_measured_program_members"
        if r.n_strata<2:return "fewer_than_two_strata"
        if r.observed_scalar_constant or r.observed_profile_numerically_constant:
            return "constant_observed"
        if r.predicted_scalar_constant or r.predicted_profile_numerically_constant:
            return "constant_predicted"
        if r.fewer_than_three_strata:return "two_strata_not_comparative"
        return "estimable"
    frame["legacy_priority_numerical_status"]=[legacy_status(r) for r in frame.itertuples()]
    return frame


def validate_outputs(program, summary, response, data):
    identities=program.identity_max_error.dropna()
    assert len(identities) and np.all(identities==0)
    assert not program.duplicated(["task","model","unit","scope","gene_mask","method","program"]).any()
    for r in summary.itertuples():
        selected=set(r.comparison_units.split(";")) if r.comparison_units else set()
        q=program[(program.task==r.task)&(program.model==r.model)&(program.scope==r.scope)
            &(program.gene_mask==r.gene_mask)&(program.method==r.method)&(program.program==r.program)
            &program.unit.isin(selected)]
        if np.isfinite(r.correlation):
            assert len(q)==r.n_comparison_units and q.correlation_status.eq("estimable").all()
            assert np.isclose(r.correlation,q.correlation.mean(),atol=1e-14,rtol=0)
        else:assert not r.all_selected_units_estimable
    stored=pd.read_csv(BASE/"results/_paper/census_unit_scores.csv")
    full=response[response.scope.eq("full_census_strata")]
    check=full.merge(stored,left_on=["task","model","unit"],right_on=["task_key","model","unit"],suffixes=("_fresh","_stored"),validate="one_to_one")
    assert len(check)==len(data)
    max_delta=float(np.max(np.abs(check.pearson_delta_fresh-check.pearson_delta_stored)))
    assert max_delta<1e-5, max_delta
    # Meaningful invariant tests for the independent scoring operator.
    x=np.array([[1.,2.,4.],[2.,5.,3.],[4.,3.,2.]])
    ctrl=np.array([.5,1.,1.5]);idx=np.array([0,2])
    assert np.allclose((x[:,idx]-ctrl[idx]).mean(1),x[:,idx].mean(1)-ctrl[idx].mean())
    assert np.allclose(response_score(10*(x-ctrl),x-ctrl),response_score(x-ctrl,x-ctrl))
    assert correlate(np.ones(3),np.arange(3.),1,0.,2.)[1]=="constant_predicted_score"
    assert correlate(np.arange(3.),np.ones(3),1,2.,0.)[1]=="constant_observed_score"
    assert correlate(np.array([0.,1.]),np.array([0.,1.]),1,1.,1.)[1]=="fewer_than_three_strata_policy"
    return dict(all_input_hashes_verified=True,all_target_labels_and_masks_aligned=True,
        canonical_targets_shared_across_candidates=True,identity_score_max_error=float(identities.max()),
        common_observed_unit_selection_checked=True,all_finite_aggregates_match_fixed_member_units=True,
        no_undefined_correlation_encoded_as_zero=True,full_census_response_score_max_difference=max_delta,
        constant_and_two_stratum_cases_checked=True,bundle_count=len(data),program_unit_rows=len(program))


def main(argv=None):
    global OUT, INPUTS
    parser=argparse.ArgumentParser(description="recompute the Figure 3 immune-program analysis "
                                               "from the deposited bundles and compare it with "
                                               "results/_paper/immune_program_revision/")
    parser.add_argument("--freeze-only",action="store_true")
    parser.add_argument("-o","--out",default=None,
                        help="keep the rebuilt CSVs here (default: a temporary directory). This "
                             "never writes into the deposit.")
    parser.add_argument("--no-compare",action="store_true",
                        help="skip the comparison against the deposited values")
    args=parser.parse_args(argv)
    tmp=None
    if args.out:
        OUT=Path(args.out).resolve()
    else:
        tmp=tempfile.TemporaryDirectory(prefix="ivcbench_program_")
        OUT=Path(tmp.name)/"program"
    if OUT.resolve()==(BASE/"results/_paper/immune_program_revision").resolve():
        raise SystemExit("refusing to write into the deposit; it is the comparison target")
    OUT.mkdir(parents=True,exist_ok=True)
    INPUTS=_program_inputs.stage(BASE,OUT)
    programs,sources=read_programs()
    manifest,data,grouped=load_inputs()
    canonical,alignment,supports=align_inputs(data,grouped)
    record=protocol_record(manifest,data,canonical,programs,sources)
    freeze_protocol(record)
    print(f"Protocol frozen: {OUT/'protocol.json'}",flush=True)
    if args.freeze_only:return
    magnitude_protocol=dict(primary="median stratum norm ratio within each lineage, then median across four lineages; same nesting for defined slopes when complete",
        mean_sensitivity="mean stratum ratios within each lineage, then equal-lineage mean; explicitly separate from primary",
        legacy_reproduction="all stored full-support OP3 outputs and original per-bundle targets/controls; exact existing S22 nanmedian-then-median estimator; counts of defined slopes and explicit declines retained")
    mp=OUT/"magnitude_sensitivity_protocol.json"
    if mp.exists():
        assert json.loads(mp.read_text())==magnitude_protocol
    else:dump_json(mp,magnitude_protocol)
    write_csv("source_alignment.csv",alignment)
    write_csv("support_by_unit.csv",supports)
    pu,ps,ru,rs,coverage=analyze(data,canonical,programs)
    puf=write_csv("program_unit_metrics.csv",add_degeneracy_flags(add_error_comparators(pd.DataFrame(pu))))
    write_csv("program_stratum_scores.csv",ps)
    ruf=write_csv("response_unit_metrics.csv",ru)
    write_csv("response_stratum_metrics.csv",rs)
    write_csv("gene_coverage.csv",coverage)
    selection,summary,best=summarize_programs(puf)
    write_csv("observed_unit_selection.csv",selection)
    write_csv("model_program_summary.csv",summary)
    bf=write_csv("figure3b_best.csv",best)
    rms=summarize_responses(ruf)
    write_csv("response_model_summary.csv",rms)
    legacy_magnitude(data)
    pcf=summary[(summary.task=="T5c")&(summary.gene_mask==MASKS[0])&(summary.method==METHODS[0])&(summary.program=="type_I_IFN")]
    pcf=pcf.merge(rms,on=["task","model","scope","status"],validate="one_to_one",suffixes=("_program","_response"))
    pcf["plottable"]=pcf.correlation.notna()&pcf.response_norm_ratio.notna()
    write_csv("panel_c.csv",pcf)
    dg=puf.groupby(["task","scope","gene_mask","method","raw_numerical_status"],dropna=False).size().reset_index(name="n_model_unit_program_comparisons")
    write_csv("rank_degeneracy_counts.csv",dg)
    legacy=puf[puf.scope.eq("full_census_strata")&puf.gene_mask.eq("all_measured_original_members")&puf.method.eq("rank_top5_mean_profile")]
    dg_legacy=legacy.groupby(["task","legacy_priority_numerical_status"],dropna=False).size().reset_index(name="n_model_unit_program_comparisons")
    write_csv("legacy_rank_degeneracy_counts.csv",dg_legacy)
    validation=validate_outputs(puf,summary,ruf,data)
    validation["protocol_sha256"]=sha(OUT/"protocol.json")
    validation["script_sha256"]=sha(Path(__file__))
    validation["outputs"]={p.name:sha(p) for p in sorted(OUT.glob("*.csv"))}
    dump_json(OUT/"validation.json",validation)
    print(bf[(bf.scope==SCOPES[0])&(bf.gene_mask==MASKS[0])&(bf.method==METHODS[0])].to_string(index=False))
    print("VALIDATION",json.dumps({k:v for k,v in validation.items() if k!="outputs"},sort_keys=True),flush=True)
    rc=0 if args.no_compare else compare_with_deposit()
    if tmp is not None:tmp.cleanup()
    return rc


DEPOSIT_RTOL, DEPOSIT_ATOL = 0.0, 1e-9
# the one column that SHOULD differ: it records where each bundle was read from, and the archive
# reads its own predictions/ copy where the frozen workspace read its snapshot of it
PROVENANCE = {("source_alignment.csv", "snapshot_path")}


def compare_with_deposit():
    """Every rebuilt CSV against its deposited copy, numerically.

    Byte equality is not the contract and never was: five of these files reach the deposit through
    a pandas re-serialization that respells floats, so the same value is written two ways. What has
    to hold is that every cell agrees, and it does -- the residual is float text, not arithmetic.
    """
    deposit=BASE/"results/_paper/immune_program_revision"
    bad,checked,worst=[],0,0.0
    for built in sorted(OUT.glob("*.csv")):
        ref=deposit/built.name
        if not ref.is_file():continue
        a,b=pd.read_csv(built,keep_default_na=False),pd.read_csv(ref,keep_default_na=False)
        checked+=1
        if list(a.columns)!=list(b.columns) or len(a)!=len(b):
            bad.append(f"{built.name}: {a.shape} against the deposit's {b.shape}");continue
        for col in a.columns:
            if (built.name,col) in PROVENANCE:continue
            x,y=a[col],b[col]
            # a column can hold numbers and words at once (a correlation beside "n.a."), and the
            # deposit respells its floats, so coerce and compare numerically wherever both sides
            # are numbers and as text everywhere else
            xn=pd.to_numeric(x,errors="coerce").astype("float64")
            yn=pd.to_numeric(y,errors="coerce").astype("float64")
            num=xn.notna()&yn.notna()
            if num.any():
                d=float((xn[num]-yn[num]).abs().max())
                worst=max(worst,d)
                if d>DEPOSIT_ATOL:bad.append(f"{built.name}[{col}]: max |delta| {d:.3g}")
            rest=~num
            if rest.any() and not x[rest].astype(str).equals(y[rest].astype(str)):
                n=int((x[rest].astype(str)!=y[rest].astype(str)).sum())
                bad.append(f"{built.name}[{col}]: {n} non-numeric cell(s) differ")
    protocol=deposit/"protocol.json"
    if protocol.is_file():
        fresh=json.loads((OUT/"protocol.json").read_text())
        frozen=json.loads(protocol.read_text())
        # Two fields legitimately differ and only these two: the manifest is rebuilt here from the
        # census rather than read from the frozen workspace's copy, and the gene-definition sources
        # are recorded at their place in this archive rather than at a snapshot of it. Their
        # CONTENT is the same, which is what the hashes below check. Every rule must be identical.
        drift=[k for k in set(fresh)|set(frozen)
               if k not in ("selected_manifest_sha256","frozen_gene_definition_sources")
               and fresh.get(k)!=frozen.get(k)]
        if drift:bad.append(f"protocol.json: {sorted(drift)} differ from the frozen protocol")
        if sorted(fresh.get("frozen_gene_definition_sources",{}).values()) != \
           sorted(frozen.get("frozen_gene_definition_sources",{}).values()):
            bad.append("protocol.json: the frozen gene definitions are not the archive's")
        checked+=1
    if bad:
        print("IMMUNE-PROGRAM DEPOSIT: FAIL")
        for line in bad:print(f"  - {line}")
        return 1
    print(f"IMMUNE-PROGRAM DEPOSIT: PASS ({checked} deposited file(s) recomputed from the "
          f"prediction bundles; max |delta| {worst:.3g}, no cell differs, protocol identical "
          f"except the recorded input paths)")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
