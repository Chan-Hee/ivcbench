#!/usr/bin/env python
"""C4 conditioned-model run (analyst, reviewer request 5/11/19): run scGen (latent, adapted gene-side
representation — the SAME extension used on C3) on the Frangieh leave-one-KO split in BOTH RNA and
protein-CITE modalities, so the modality axis reports a real conditioned-vs-floor contrast rather than
a simple-only floor. Also runs a CPU in-process 'linear-shift + KO-embedding' conditioned baseline
(request 11) as a second, fully reproducible conditioned comparator. Leak-safe via the existing
build_split + audit_split + run_job pipeline (held-KO cells removed from train; predicted from
non-held control + a gene-side-conditioned shift). Heavy scGen shells out to scperturbench_eval (CPU).
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
os.environ.setdefault("IVCBENCH_SCGEN_EPOCHS", "40")

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.runner.run import run_job
from ivcbench.baselines.heavy import ScGen
from ivcbench.baselines.base import BaselineAdapter, PredResult
from ivcbench.splits.builder import build_split


# ---- A fully reproducible CPU conditioned baseline: per-KO shift regressed on a leak-safe gene-side
#      embedding (control-only PCA gene-loadings) — the same 'adapted gene-side repr' idea as scGen but
#      a linear gene-space shift instead of a VAE latent. Genuinely conditioned: each held KO gets a
#      DIFFERENT predicted shift from its embedding (vs cell-mean's one global shift). ----
class LinearShiftKOEmb(BaselineAdapter):
    name, family, gpu = "linear-shift-KOemb", "latent", False

    def fit(self, cs, split, side_info=None):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        tr = split.train_idx
        obs = cs.obs.iloc[tr]
        is_ctrl = obs["is_control"].to_numpy()
        pert = obs["perturbation"].to_numpy().astype(str)
        genes = list(cs.var_names)
        gpos = {g: i for i, g in enumerate(genes)}
        Xtr = cs.X[tr]
        # leak-safe gene embedding: control-only PCA gene loadings
        ctrl_X = Xtr[is_ctrl]
        k = int(min(50, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
        gpca = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X)
        self.gene_emb = gpca.components_.T          # (n_feat, k)
        self.gpos = gpos
        ctrl_mean = ctrl_X.mean(0)
        # per-train-KO shift in gene space, only for KO genes present as features (so they have an emb)
        train_kos = [g for g in np.unique(pert[~is_ctrl]) if g in gpos]
        D, E = [], []
        for g in train_kos:
            m = (pert == g) & (~is_ctrl)
            if m.sum() == 0:
                continue
            D.append(Xtr[m].mean(0) - ctrl_mean)
            E.append(self.gene_emb[gpos[g]])
        D = np.vstack(D); E = np.vstack(E)
        self.reg = Ridge(alpha=1.0).fit(E, D)       # gene-emb -> gene-space shift
        self.ctrl = self._control_mean(cs, split)

    def predict(self, cs, split, side_info=None) -> PredResult:
        test_perts = cs.obs.iloc[split.test_idx]["perturbation"].to_numpy().astype(str)
        preds = []
        for p in test_perts:
            if p in self.gpos:
                shift = self.reg.predict(self.gene_emb[self.gpos[p]][None, :])[0]
                preds.append(self.ctrl + shift)
            else:
                preds.append(self.ctrl)             # no embedding -> fall back to control
        return PredResult(np.vstack(preds), self.ctrl)


def main():
    out_rows = []
    for modality, mod_tag in [("rna", "RNA"), ("protein", "protein-CITE")]:
        cs = load(modality=modality)
        g = cs.uns["genes_perturbed"]
        ds_name = cs.uns.get("dataset", f"frangieh_{modality}")
        # The census deposits ONLY the RNA C4_Axis2 bundles; the protein modality is the analysis-only
        # modality comparator (its numbers go to results/C4/conditioned_rows.json). The RNA and protein
        # runs share one cluster/model/split bundle filename (no modality key), so dumping protein would
        # overwrite the RNA bundle the census reads. RNA runs first and dumps; drop the dump for protein.
        if modality != "rna":
            os.environ.pop("IVCBENCH_PRED_DUMP", None)
        for frac, lbl in [(0.25, "25"), (0.50, "50")]:
            held = c4.held_ko_fraction(g, frac, seed=0)
            spec = c4.modality_lo_ko(held, lbl)
            excl = list(spec.held_values)        # downstream_only=True -> exclude held KO genes
            for B in [LinearShiftKOEmb, ScGen]:
                t0 = time.time()
                adapter = B()
                try:
                    r = run_job(cs, spec, adapter, seed=0, exclude_genes=excl,
                                adapted_implemented=True)
                except Exception as e:  # noqa: BLE001
                    r = {"baseline": adapter.name, "family": adapter.family, "split": spec.name,
                         "action": "failed", "ran": False, "error": f"{type(e).__name__}: {e}"}
                r.update(cluster="C4", dataset=ds_name, modality=mod_tag, elapsed_s=round(time.time() - t0, 1))
                out_rows.append(r)
                keep = {k: r.get(k) for k in ("baseline", "modality", "split", "action", "ran",
                                              "leak_free", "n_train", "n_test", "pearson_delta",
                                              "pearson_delta_ontarget", "pearson_delta_lo",
                                              "pearson_delta_hi", "e_distance", "elapsed_s", "error")}
                print(json.dumps(keep), flush=True)
    Path("results/C4/conditioned_rows.json").write_text(json.dumps(out_rows, indent=2, default=str))
    print("WROTE results/C4/conditioned_rows.json")


if __name__ == "__main__":
    main()
