"""Chemistry-aware reference baseline (FP-ridge).

A lightweight, genuinely *applicable* comparator for the unseen-compound split: it conditions on a
compound-side representation (Morgan fingerprint) by ridge-regressing the control→treated expression
shift onto the fingerprint, so it can predict a never-seen compound's effect from chemistry alone —
exactly what vanilla label-conditioned baselines cannot. It stands in for the chemCPA/CPA family as
the applicable chemistry model until the full latent stacks are wired behind the same interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from ..data.schema import CONTROL_TOKEN
from .base import BaselineAdapter, PredResult


class FPRidge(BaselineAdapter):
    name, family, gpu = "FP-ridge", "chemistry", False

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.ridge = None

    def fit(self, cs, split, side_info=None):
        fps = (side_info or {}).get("fingerprint", {})
        tr = split.train_idx
        obs = cs.obs.iloc[tr]
        is_ctrl = obs["is_control"].to_numpy()
        Xtr = cs.X[tr]
        self.ctrl_mean = Xtr[is_ctrl].mean(0) if is_ctrl.any() else Xtr.mean(0)
        perts = obs["perturbation"].to_numpy()

        feats, targets = [], []
        for p in np.unique(perts):
            if p == CONTROL_TOKEN or p not in fps:
                continue
            delta = Xtr[perts == p].mean(0) - self.ctrl_mean
            feats.append(fps[p])
            targets.append(delta)
        if feats:
            self.ridge = Ridge(alpha=self.alpha).fit(np.asarray(feats), np.asarray(targets))

    def predict(self, cs, split, side_info=None) -> PredResult:
        fps = (side_info or {}).get("fingerprint", {})
        ctrl = self._control_mean(cs, split)
        perts = cs.obs.iloc[split.test_idx]["perturbation"].to_numpy()
        pred = np.empty((len(perts), cs.n_genes), dtype=np.float32)
        for i, p in enumerate(perts):
            fp = fps.get(p)
            if fp is None or self.ridge is None:
                pred[i] = ctrl  # no chemistry available -> falls back to control (floor)
            else:
                pred[i] = ctrl + self.ridge.predict(fp[None, :])[0]
        return PredResult(pred, ctrl)
