# Release steps for the BIB-26-1553 revision

The published release is **v1.1.7 (2 July 2026)**, which carries the **35-cell** census.
This revision reports **47** evaluations. Until a new version is published, the paper's
availability statement is not true: a reader who follows it downloads a 35-row
`cross_cluster_headline.csv` listing 14 methods, and biolord, CellFlow, PerturbNet and
PRnet — four of the predictors this revision was asked to add — are absent.

Everything needed is committed on this branch. Three author-only steps remain.

```bash
# 1. push the branch and the new tag
git push origin HEAD
git tag -a v1.2.0 -m "47-cell census; supplementary tables S1-S23 under the paper's numbering"
git push origin v1.2.0

# 2. create the GitHub release from v1.2.0 (Zenodo picks it up through the webhook)

# 3. confirm the new Zenodo version under concept DOI 10.5281/zenodo.20756042
#    and tell the editorial office which version is the version of record
```

## What the new version adds over v1.1.7

| | v1.1.7 | this branch |
|---|---|---|
| `results/_paper/cross_cluster_headline.csv` | 35 rows, 14 methods | 47 rows, 17 methods and comparators |
| `results/_paper/census_uncertainty.csv` | absent | 47 rows with interval, detectable margin, verdict |
| `results/_paper/immune_readout_summary.csv` | absent | 320 rows, per model, unit and program |
| `results/_paper/census_bundle_manifest.csv` | 35-cell state | 1,362 bundles covering all 47 cells |
| `results/_paper/supplementary_tables/` | absent | every table the paper cites, S1-S23, under the paper's own numbering, with `MANIFEST.csv` |

## Check before tagging

```bash
python - <<'PY'
import csv
h=list(csv.DictReader(open('results/_paper/cross_cluster_headline.csv')))
m=list(csv.DictReader(open('results/_paper/supplementary_tables/MANIFEST.csv')))
assert len(h) == 47, len(h)
assert len(m) == 25, len(m)
print('census', len(h), 'cells; supplementary tables', len(m))
PY
```
