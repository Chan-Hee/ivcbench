# Release steps for the BIB-26-1553 revision

The published release is **v1.1.7 (2 July 2026)**, which carries the **35-cell** census.
This revision reports **58** evaluations. Until a new version is published, the paper's
availability statement is not true: a reader who follows it downloads a 35-row
`cross_cluster_headline.csv` listing 14 methods, and biolord, CellFlow, PerturbNet and
PRnet — four of the predictors this revision was asked to add — are absent.

The tag is **v1.2.3**, matching `pyproject.toml`, `CITATION.cff` and the README: a Zenodo version labelled anything else contradicts the package the paper cites.

Everything needed is committed on this branch. Three author-only steps remain, plus the two
tidy-ups in "Before the tree goes public" below.

```bash
# 1. push the branch and the new tag
git push origin HEAD
git tag -a v1.2.3 -m "58-cell census; supplementary tables S1-S23 under the paper's numbering"
git push origin v1.2.3

# 2. create the GitHub release from v1.2.3 (Zenodo picks it up through the webhook)

# 3. confirm the new Zenodo version under concept DOI 10.5281/zenodo.20756042
#    and tell the editorial office which version is the version of record
```

## Before the tree goes public

Both of these are decisions, not fixes, so they are left here rather than made:

1. **This file, and one clause of the README, go stale the moment the release exists.** It says the
   paper's availability statement "is not true", and `README.md` ends its licensing paragraph with
   "no public release of this revision has been performed". Delete `RELEASE_FOR_REVISION.md` and
   drop that clause in the same commit that creates the tag.

2. **`runs/` is 375 tracked files of internal GPU dispatch scaffolding** (126 `.cmd`, 120
   `.status`, 116 `.sh`, plus `preflight.py`, `launch.sh`, `status.sh` and `RESUME.md`). 117 of
   them embed an absolute path from the authoring machine, `RESUME.md` is a Korean session-resume
   note about screen sessions and GPU assignment, and the directory appears nowhere in the
   README's package map. Nothing in the deposit reads it. Either `git rm -r --cached runs`
   and gitignore it, or add it to the package map and describe what it is.

## What the new version adds over v1.1.7

| | v1.1.7 | this branch |
|---|---|---|
| `results/_paper/cross_cluster_headline.csv` | 35 rows, 14 methods | 58 rows, 16 methods and comparators |
| `results/_paper/census_uncertainty.csv` | absent | 58 rows with interval, detectable margin, verdict |
| `results/_paper/immune_readout_summary.csv` | absent | 456 rows, per model, unit and program |
| `results/_paper/census_bundle_manifest.csv` | 35-cell state | 1,401 bundles covering all 58 cells |
| `results/_paper/supplementary_tables/` | absent | every table the paper cites, S1-S23, under the paper's own numbering, with `MANIFEST.csv` |

## Check before tagging

```bash
python - <<'PY'
import csv
h=list(csv.DictReader(open('results/_paper/cross_cluster_headline.csv')))
m=list(csv.DictReader(open('results/_paper/supplementary_tables/MANIFEST.csv')))
assert len(h) == 58, len(h)
assert len(m) == 25, len(m)
print('census', len(h), 'cells; supplementary tables', len(m))
PY
```
