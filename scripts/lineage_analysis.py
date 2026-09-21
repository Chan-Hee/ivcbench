#!/usr/bin/env python
"""Independently re-score the deposited T1/T5c bundles for the Figure 3d lineage anchors.

The references are the stronger of cell-mean and linear-PCA in EACH lineage. These are descriptive
single-execution comparisons, not significance calls. Like scripts/program_analysis.py this reads
only the deposited prediction bundles -- no cells, no model -- and compares what it computes with
results/_paper/immune_program_revision/.

    python scripts/lineage_analysis.py                 # recompute and compare, writes nothing
    python scripts/lineage_analysis.py -o work/lineage # keep the rebuilt CSVs for inspection
"""
import argparse
import sys
import tempfile
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _program_inputs  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT = None  # bound in main(); the frozen workspace's tables/ is not in the archive


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="recompute the Figure 3d lineage anchors from the "
                                             "deposited bundles and compare with the deposit")
    ap.add_argument('-o', '--out', default=None,
                    help='keep the rebuilt CSVs here (default: a temporary directory)')
    ap.add_argument('--no-compare', action='store_true', help='skip the deposit comparison')
    args = ap.parse_args(argv)
    tmp = None
    if args.out:
        OUT = Path(args.out).resolve()
    else:
        tmp = tempfile.TemporaryDirectory(prefix='ivcbench_lineage_')
        OUT = Path(tmp.name)
    DEPOSIT = BASE / 'results/_paper/immune_program_revision'
    if OUT.resolve() == DEPOSIT.resolve():
        raise SystemExit('refusing to write into the deposit; it is the comparison target')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(_program_inputs.stage(BASE, OUT) / 'selected_bundle_manifest.csv')
    census = pd.read_csv(BASE / 'results/_paper/census_unit_scores.csv')
    rows, strata_rows = [], []
    for row in manifest[manifest.task.isin(['T1', 'T5c'])].itertuples():
        with np.load(BASE / row.snapshot_path, allow_pickle=True) as z:
            pred, obs, ctrl = [np.asarray(z[k], dtype=np.float64) for k in ['pred_means', 'obs_means', 'control_mean']]
            keep = np.ones(pred.shape[1], dtype=bool)
            if 'exclude_gene_idx' in z.files:
                keep[np.asarray(z['exclude_gene_idx'], dtype=int)] = False
            dp, do = (pred-ctrl)[:, keep], (obs-ctrl)[:, keep]
            dp, do = dp-dp.mean(axis=1, keepdims=True), do-do.mean(axis=1, keepdims=True)
            denominator = np.linalg.norm(dp, axis=1)*np.linalg.norm(do, axis=1)
            corr = np.divide(np.einsum('ij,ij->i', dp, do), denominator,
                             out=np.zeros(len(dp)), where=denominator>=1e-12)
            for s, value in zip(z['strata'].astype(str), corr):
                strata_rows.append(dict(task=row.task,model=row.model,unit=row.unit,stratum=s,pearson_delta=value))
        ref = census[(census.task_key==row.task)&(census.model==row.model)&(census.unit==row.unit)]
        if len(ref)!=1:
            raise ValueError(f'Unexpected census match: {row.task}/{row.model}/{row.unit}')
        value=float(corr.mean()); recorded=float(ref.iloc[0].pearson_delta)
        if not np.isclose(value,recorded,atol=1e-10,rtol=0):
            raise AssertionError((row.task,row.model,row.unit,value,recorded))
        rows.append(dict(task=row.task,model=row.model,unit=row.unit,n_strata=len(corr),
                         pearson_delta=value,census_value=recorded,absolute_difference=abs(value-recorded),
                         snapshot_path=row.snapshot_path,scope='full_census_strata'))
    all_scores=pd.DataFrame(rows)
    out=[]
    for task,model in [('T1','scGen'),('T5c','FP-ridge')]:
        for row in all_scores[(all_scores.task==task)&(all_scores.model==model)].itertuples():
            floors=all_scores[(all_scores.task==task)&(all_scores.unit==row.unit)&all_scores.model.isin(['cell-mean','linear-PCA'])]
            assert len(floors)==2
            floor=floors.sort_values(['pearson_delta','model'],ascending=[False,True]).iloc[0]
            out.append(dict(task=task,model=model,unit=row.unit,model_score=row.pearson_delta,
                            floor_model=floor.model,floor_score=floor.pearson_delta,
                            margin=row.pearson_delta-floor.pearson_delta,n_strata=row.n_strata,
                            scope='full_census_strata'))
    out=pd.DataFrame(out)
    all_scores.to_csv(OUT/'lineage_all_models.csv',index=False)
    pd.DataFrame(strata_rows).to_csv(OUT/'lineage_stratum_scores.csv',index=False)
    out.to_csv(OUT/'figure3d_lineage.csv',index=False)
    report={'n_bundles':len(rows),'max_census_difference':float(all_scores.absolute_difference.max()),
            'anchors':[{**dict(task=t,model=m),'positive':int((g.margin>0).sum()),'n':len(g)}
                       for (t,m),g in out.groupby(['task','model'])]}
    (OUT/'lineage_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    rc = 0 if args.no_compare else compare(OUT, DEPOSIT)
    if tmp is not None:
        tmp.cleanup()
    return rc


# lineage_all_models.csv records where each bundle was read from, which is the one value that
# should differ: the archive reads its own predictions/ copy, the frozen workspace read a snapshot
PROVENANCE = {('lineage_all_models.csv', 'snapshot_path')}


def compare(out, deposit, atol=1e-9):
    bad, checked, worst = [], 0, 0.0
    for built in sorted(Path(out).glob('*.csv')):
        ref = Path(deposit) / built.name
        if not ref.is_file():
            continue
        a, b = pd.read_csv(built, keep_default_na=False), pd.read_csv(ref, keep_default_na=False)
        checked += 1
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            bad.append(f'{built.name}: {a.shape} against the deposit\'s {b.shape}')
            continue
        for col in a.columns:
            if (built.name, col) in PROVENANCE:
                continue
            x, y = a[col], b[col]
            xn = pd.to_numeric(x, errors='coerce').astype('float64')
            yn = pd.to_numeric(y, errors='coerce').astype('float64')
            num = xn.notna() & yn.notna()
            if num.any():
                d = float((xn[num] - yn[num]).abs().max())
                worst = max(worst, d)
                if d > atol:
                    bad.append(f'{built.name}[{col}]: max |delta| {d:.3g}')
            rest = ~num
            if rest.any() and not x[rest].astype(str).equals(y[rest].astype(str)):
                n = int((x[rest].astype(str) != y[rest].astype(str)).sum())
                bad.append(f'{built.name}[{col}]: {n} non-numeric cell(s) differ')
    if bad:
        print('LINEAGE ANCHORS: FAIL')
        for line in bad:
            print(f'  - {line}')
        return 1
    print(f'LINEAGE ANCHORS: PASS ({checked} deposited table(s) recomputed from the prediction '
          f'bundles; max |delta| {worst:.3g}, no cell differs)')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
