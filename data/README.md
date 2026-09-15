# Data access and executed subsets

Raw single-cell objects are not bundled with the code. Mean profiles are sufficient for the main CPU replay. Study-wide resource sizes in Supplementary Table S1 differ from the filtered execution subsets below.

| Source | Used for | Processed-data route | Executed subset / access note |
|---|---|---|---|
| Kang | T1; supplementary donor validation | GEO GSE96583 | IFN-β PBMCs; eight lineages / eight donors; public |
| Soskic | T2 | Trynka-lab processed CD4 activation objects | 106 paired donors; public condition-specific residualized/scaled/clipped matrices, no raw-count layer; RNA reads EGA EGAD00001008197 and separate genotypes are controlled |
| Cano-Gamez | Supplementary donor validation | BioStudies S-BSST2978 | Four donors; processed counts public; EGA EGAS00001003215 raw reads controlled |
| Shifrut | T3 | GEO GSE119450 | Two held genes in the fixed 10% split; public |
| Schmidt | T3 | GEO GSE190604 | Seven held genes; public |
| McCutcheon | T3 | GEO GSE218985 | CRISPRa and CRISPRi arms, two held genes each; public |
| Chen | T3; supplementary surface readouts | GEA E-GEAD-648 / BioProject PRJDB16517 | 30 held genes; source registration/access procedure required |
| Frangieh | T4; supplementary independent protein fits | scPerturb Zenodo record 13350497 | IFNγ melanoma subset; 25%/50% KO holdouts; public |
| OP3 | T5c/T5u | GEO GSE279945 | Four coarse lineages / 28 held compounds; public |

The broader immune-resource survey includes studies not executed in the final panel. Their presence in the inventory is not evidence that their raw data were downloaded or included in a numerical result.

Chen is not GEO GSE255832: that accession belongs to the mouse Pretto study. Public processed Cano-Gamez counts do not require approval for the separately controlled raw reads. Repository access policies can change; use the original source's current terms.

The loaders in `src/ivcbench/data/loaders/` document input schemas. Original fitting scripts may use local data/checkpoint paths; configure those for your own installation and the exact recorded model environment. Downloading or retraining is unnecessary for `make reproduce`. Dataset licenses, access agreements and pretrained-model terms remain with their providers.
