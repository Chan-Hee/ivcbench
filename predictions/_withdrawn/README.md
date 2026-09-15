# Withdrawn prediction bundles

These bundles were produced during the revision but their model is **not** part of the reported
benchmark, so they are held here rather than in the census layer. Nothing is deleted: the runs
happened and the evidence is kept.

## SCREEN (16 bundles)

SCREEN (Xu et al., Front Comput Sci 2024) is native by interface on T1 and T2 — an optimal-transport
VAE that predicts the perturbed counterpart of a held-out group from that group's own control cells,
which is exactly the structure of both tasks. It was run on both.

**T2 does not converge.** Across 53 donors the released implementation raised
`ValueError: Input X contains NaN` on 45 of them, at roughly 1,000 s per donor, with and without the
documented training-cell cap (`IVCBENCH_SCREEN_MAXCELLS`, whose own comment records that "the VAE
diverged to NaN on some folds"). Eight donors completed.

Reporting T1 alone would have made the model's coverage a matter of which cells happened to run
rather than of what its interface supports, which is the inconsistency the revision's admission rule
exists to prevent. SCREEN is therefore withdrawn in full, T1 included, and the reason is stated in
the per-method supplementary entries.

The 8 completed T2 donors and the 8 T1 lineages are kept here so the claim can be checked.
