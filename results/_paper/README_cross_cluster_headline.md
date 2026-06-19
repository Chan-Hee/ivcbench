# Provenance — `cross_cluster_headline.csv`

This table is the cross-cluster headline census of conditioned-model performance against the
**universal two-member floor = {cell-mean shift, linear-PCA shift}** (PREREGISTRATION §2). It is
generated mechanically by `scripts/assemble_cross_cluster.py` from the deposited per-cluster results;
the human-readable companion is `cross_cluster_headline.md`.

## Which column is the verdict, and which is only descriptive

The CSV carries two different readings of how a model compares to the floor. They are **not**
interchangeable, and only one of them is the verdict used in the paper:

- **`beats_both_floor_members` (boolean) — THIS IS THE BINDING VERDICT.** A method counts as
  *working* on a task only if it beats **BOTH** universal-floor members, i.e. its response-direction
  Pearson-Δ exceeds **both** the cell-mean shift (`floor_cell_mean`) **and** the linear-PCA shift
  (`floor_linear_PCA`). This two-member, beat-both rule is the one the manuscript uses for every
  "works / does not work" claim and for the figure verdict rings.

- **`delta_vs_floor_mean` (numeric) — DESCRIPTIVE MARGIN ONLY.** This column is the margin of the
  model's Pearson-Δ over the **mean** of the two floor members
  (`delta_vs_floor_mean = pearson_delta − floor_mean`, where `floor_mean = mean(floor_cell_mean,
  floor_linear_PCA)`). It is provided as a continuous, at-a-glance descriptive magnitude. It is
  **not** the verdict: a model can post a positive `delta_vs_floor_mean` (it clears the *average* of
  the two members) while still failing `beats_both_floor_members` because it does not clear the
  *binding* (larger) member. The per-member descriptive margins `delta_vs_cell_mean` and
  `delta_vs_linear_PCA` are provided for the same descriptive purpose.

In short: **read `beats_both_floor_members` for the verdict; read `delta_vs_floor_mean` only as a
descriptive margin.** Several rows in this CSV illustrate the gap between the two — e.g. CINEMA-OT on
C3 (`delta_vs_floor_mean = +0.063` but `beats_both_floor_members = False`, because it is below the
binding cell-mean member). `donor-shift` and `FP-ridge`-as-context are context-only comparators and
are **not** universal-floor members.

The CI-gated descriptive *fit* verdict (cluster-bootstrap CI_low > 0 on the family-minus-floor gap,
PREREGISTRATION §5) is a separate artifact, `descriptive_fit_matrix.csv`; it is not asserted in this
headline table.
