# Within-family CONSISTENCY table

Generated from `within_family_consistency.csv`. For each family with ≥2 models on a task: do the members **agree** on the beat-floor verdict, and how correlated are their per-unit Pearson-Δ vectors (Spearman ρ)?  Floor = universal {cell-mean, linear-PCA}.

| cluster | task | split | family | models | n beat both floor | n models | verdict agreement | Spearman ρ (per-unit) | flag |
|---|---|---|---|---|---|---|---|---|---|
| C1 | cytokine/Kang | cell-context (LOCT) | Flow | CellFlow+PerturbNet | 2 | 2 | agree | 1.000 |  |
| C1 | cytokine/Kang | cell-context (LOCT) | Foundation | scFoundation+scGPT | 0 | 2 | agree | 0.667 |  |
| C1 | cytokine/Kang | cell-context (LOCT) | Latent | CPA+scGen | 0 | 2 | agree | −0.048 |  |
| C1 | cytokine/Kang | cell-context (LOCT) | OT | CellOT+scPRAM | 1 | 2 | split | 0.905 |  |
| C2 | donor/Soskic | donor (LODO) | Flow | CellFlow+PerturbNet | 0 | 2 | agree | 0.794 |  |
| C2 | donor/Soskic | donor (LODO) | Foundation | scFoundation+scGPT | 0 | 2 | agree | 0.690 |  |
| C2 | donor/Soskic | donor (LODO) | Latent | CPA+scGen | 0 | 2 | agree | 0.567 |  |
| C2 | donor/Soskic | donor (LODO) | OT | CellOT+scPRAM | 1 | 2 | split | 0.419 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Flow | CellFlow+PerturbNet | 0 | 2 | agree | 0.300 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Foundation | scFoundation+scGPT | 0 | 2 | agree | 0.800 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Graph | AttentionPert+GEARS | 0 | 2 | agree | 1.000 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Hybrid | PertAdapt+STATE | 0 | 2 | agree | 0.800 |  |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Flow | CellFlow+PerturbNet | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 biological-unit replicates); descriptive only, no inferential CI by design |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Foundation | scFoundation+scGPT | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 biological-unit replicates); descriptive only, no inferential CI by design |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Graph | AttentionPert+GEARS | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 biological-unit replicates); descriptive only, no inferential CI by design |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Hybrid | PertAdapt+STATE | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 biological-unit replicates); descriptive only, no inferential CI by design |
| C5 | small-mol/OP3 | cell-context (LOCT) | Chemistry | FP-ridge+PRnet | 1 | 2 | split | 0.800 |  |
| C5 | small-mol/OP3 | cell-context (LOCT) | Flow | CellFlow+PerturbNet | 0 | 2 | agree | 0.400 |  |
| C5 | small-mol/OP3 | cell-context (LOCT) | Foundation | scFoundation+scGPT | 2 | 2 | agree | 0.800 |  |
| C5 | small-mol/OP3 | cell-context (LOCT) | Latent | CPA+scGen | 0 | 2 | agree | −0.200 |  |
| C5 | small-mol/OP3 | cell-context (LOCT) | OT | CellOT+scPRAM | 0 | 2 | agree | 1.000 |  |
| C5 | small-mol/OP3 | unseen-compound | Chemistry | FP-ridge+PRnet | 0 | 2 | agree | — (n<3 units) |  |
| C5 | small-mol/OP3 | unseen-compound | Flow | CellFlow+PerturbNet | 0 | 2 | agree | — (n<3 units) |  |
| C5 | small-mol/OP3 | unseen-compound | Foundation | scFoundation+scGPT | 1 | 2 | split | — (n<3 units) |  |
| C5 | small-mol/OP3 | unseen-compound | Latent | Biolord+CPA | 0 | 2 | agree | — (n<3 units) |  |

## Notes

- **Verdict agreement: 21 of 25 family/task cells agree**, 4 split — C1 OT (CellOT+scPRAM); C2 OT (CellOT+scPRAM); C5 Chemistry (FP-ridge+PRnet); C5 Foundation (scFoundation+scGPT). In a split cell one member of the family clears both floor members and the other does not, so family membership does not by itself predict the beat-floor verdict.
- **C3 within-family rho** (Foundation 0.80, Graph 1.00, Hybrid 0.80, Flow 0.30): where rho is high the family's members rank datasets the same way even though both sit below the floor, which is consistent failure rather than noise.
- **Latent-family rho by cluster**: C1 −0.05, C2 0.57, C5 −0.20. These are two-model rank correlations over the units of one split; read them as indicative, not as an estimate with an interval.
- **C4 rho is undefined** for Flow, Foundation, Graph, Hybrid: the modality split has only two folds (LO-KO 25% and 50%), fewer than the three units a rank correlation needs. The census reports those cells by their observed unit range rather than an interval, for the same reason.
