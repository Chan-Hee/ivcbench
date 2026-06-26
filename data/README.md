# Data availability

**Raw data are not shipped in this repository.** Every dataset is a public or access-controlled deposit
and is fetched by the download scripts in `scripts/` into `data/<cluster>/<dataset>/`, then ingested by
the matching loader in `src/ivcbench/data/loaders/`. The machine-readable manifest is
[`scripts/datasets.csv`](../scripts/datasets.csv).

`scripts/datasets.csv` is the **download/access manifest for the datasets actually scored in the benchmark
(15 rows)**; it is deliberately smaller than the **21-dataset landscape survey in Supplementary Table S1**,
which inventories the full immune-perturbation dataset space (including datasets surveyed but not scored).
The two are not expected to match row-for-row.

The deposited result tables under [`results/`](../results) are sufficient to reproduce every figure and
Supplementary Table **without** downloading any raw data (see [`REPRODUCE.md`](../REPRODUCE.md)).

`data/` itself is git-ignored except for this README.

---

## Datasets

| Cluster | Dataset | Modality / role | Accession / DOI | Access | Download script | Loader |
|---|---|---|---|---|---|---|
| C1 | **Kang 2018** PBMC IFN-β | cytokine stimulation (cell-context, LOLO anchor) | GEO **GSE96583** | public | `download_kang.sh`, `download_public.sh` | `loaders/kang.py` |
| C1 | **Oesinghaus** cytokine dictionary | cytokine stimulation | bioRxiv 2025.12.12.693897 (Parse + Allen `theislab/HumanCytokineDict`) | public portal | — (portal) | — |
| C1 | **Human Cytokine Dictionary** summary table | pseudobulk-DE per (gene × celltype × cytokine); S11 unseen-cytokine LOCO | `theislab/HumanCytokineDict` (Parse + Allen, bioRxiv 2025.12.12.693897) | public portal | — (portal) → `data/human_cytokine_dict/hcd_mini.csv` | `scripts/newdata_cytokine_loco.py` |
| C1 | **Cano-Gamez** CD4⁺ effectorness | cytokine stimulation | EGA **EGAS00001003215** / EGAD00001005290 | DAC | — (EGA DAC; see `scripts/apply_ega_dac.md`) | — |
| C2 | **Soskic 2022** CD4⁺ activation | donor/temporal (106-donor LODO anchor) | trynkalab processed h5ad; raw EGA **EGAD00001008197** | public (processed) | `download_soskic.sh` | `loaders/soskic.py` |
| C3 | **Shifrut 2018** primary-T KO | CRISPR unseen-gene | GEO **GSE119450** | public | `download_public.sh` | `loaders/shifrut.py` |
| C3 | **Schmidt 2022** primary-T CRISPRa | CRISPR unseen-gene | GEO **GSE190604** | public | `download_public.sh` | `loaders/schmidt.py` |
| C3 | **McCutcheon 2023** primary-T CRISPRi/a | CRISPR unseen-gene | GEO **GSE218985** (subseries of GSE218988) | public | `download_public.sh` | `loaders/mccutcheon.py` |
| C3 | **Chen 2025** FOXP3 Perturb-icCITE-seq | CRISPR unseen-gene | DDBJ **PRJDB16517** / GEA **E-GEAD-648** (NOT GSE255832) | login | — (DDBJ/GEA login) | `loaders/chen.py` |
| C4 | **Frangieh 2021** Perturb-CITE-seq (tumour) | complex-context / modality stress test | scPerturb Zenodo **10.5281/zenodo.13350497** | public | `download_frangieh.sh` | `loaders/frangieh.py` |
| C4 | **Belk 2022** exhaustion Perturb-seq (mouse) | complex-context | GEO **GSE203592** (super-series GSE203593) | public | `download_public.sh` | — |
| C4 | **Zhou 2023** OT-I CD8 in vivo (mouse) | complex-context | GEO **GSE216800** (subseries GSE216909) | public | — | — |
| C4 | **Pretto 2025** OT-I CD8 metabolic CROP-seq (mouse) | complex-context | GEO **GSE255832** | public | — | — |
| C5 | **OP3 / Szałata 2024** PBMC chemical perturbation | small-molecule unseen-compound + cell-context anchor | GEO **GSE279945** | public | `download_op3.sh` | `loaders/op3.py` |

(The C3 `zhu` GSE314342 and `moonen` datasets are listed in `scripts/datasets.csv` as part of the
surveyed inventory but are obtained via the CZI Virtual Cells Platform CLI and are not part of the scored
result set.)

---

## How the loaders find the data

Each loader expects its files under `data/<cluster>/<dataset>/` (the layout the download scripts create),
for example:

```
data/C1/kang/GSE96583_RAW.tar  (+ batch tsne.df, genes.tsv)
data/C3/shifrut/GSE119450_RAW.tar
data/C4/frangieh/<Zenodo RNA + protein h5ad>
data/C5/op3/GSE279945_sc_counts_processed.h5ad
```

Run every public download in one command:

```bash
make data                          # = bash scripts/download_all.sh (all public datasets, with a summary)
bash scripts/download_all.sh --list   # preview the plan without downloading
```

or run the individual scripts:

```bash
bash scripts/download_public.sh    # Kang (C1) + Shifrut/Schmidt/McCutcheon/Chen (C3) + Belk (C4)
bash scripts/download_op3.sh       # OP3 (C5)
bash scripts/download_soskic.sh    # Soskic processed h5ad (C2)
bash scripts/download_frangieh.sh  # Frangieh Perturb-CITE-seq h5ad (C4), scPerturb Zenodo 13350497
```

Access-controlled datasets (Cano-Gamez EGA DAC; Chen DDBJ/GEA login; Frangieh Zenodo; CZI VCP datasets)
must be obtained from their archives per the accessions above; see `scripts/apply_ega_dac.md` and
`scripts/data_access_requests.md` for the access notes.

Each dataset retains its own license / data-use terms from the originating archive.
