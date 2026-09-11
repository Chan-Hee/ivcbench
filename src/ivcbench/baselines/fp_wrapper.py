"""Unseen-compound extension: wrap any conditioned predictor so it can be scored on T5u.

The disclosed interface already used for scGen and STATE on this cell is "regress the model's
per-compound response on the compound's Morgan fingerprint, then predict the held compound's response
from its fingerprint". This wraps that pattern once, model-agnostically, instead of re-implementing it
per runner:

  1. the wrapped model is trained on the leak-safe train fold, exactly as it is on any other cell;
  2. it is asked to predict every TRAIN compound, which gives one model-derived response vector per
     compound -- the model's own learned response manifold, not the observed data;
  3. ridge regression maps Morgan fingerprint -> that response;
  4. a held compound's response is predicted from its fingerprint alone and added to the control mean.

The held compound's cells never enter training and never enter the regression, so the leak boundary
is the same one every other cell uses. What the entry bounds is this extension applied to that model,
which is why such cells are reported as adapted.
"""

from __future__ import annotations

import numpy as np

from .base import BaselineAdapter, PredResult


class FPUnseenCompound(BaselineAdapter):
    """Compose a base adapter with the fingerprint regression. `base_cls` is set per instance."""

    base_cls = None
    family = "adapted"

    def __init__(self, base_cls=None):
        if base_cls is not None:
            self.base_cls = base_cls
        self.base = self.base_cls()
        self.name = self.base.name
        self.family = getattr(self.base, "family", "adapted")
        self.gpu = getattr(self.base, "gpu", True)
        self.cuda_device = None

    def fit(self, cs, split, side_info=None):
        self.base.cuda_device = self.cuda_device
        self.base.fit(cs, split, side_info=side_info)
        self.ctrl = (
            self.base.ctrl
            if hasattr(self.base, "ctrl")
            else self._control_mean(cs, split)
        )

    def predict(self, cs, split, side_info=None) -> PredResult:
        from sklearn.linear_model import Ridge

        fps = (side_info or {}).get("fingerprint") or {}
        if not fps:
            raise NotImplementedError(
                f"{self.name}: T5u extension needs compound fingerprints"
            )

        obs = cs.obs
        tr = split.train_idx
        is_ctrl_tr = obs.iloc[tr]["is_control"].to_numpy().astype(bool)
        pert_tr = obs.iloc[tr]["perturbation"].to_numpy().astype(str)
        train_cpds = sorted({p for p in pert_tr[~is_ctrl_tr] if p in fps})
        if len(train_cpds) < 5:
            raise NotImplementedError(
                f"{self.name}: only {len(train_cpds)} train compounds with a "
                "fingerprint; the regression is not estimable"
            )

        # ask the trained model for each TRAIN compound's response, then regress it on chemistry
        shifts = (
            self.base.predict_perturbations(cs, split, train_cpds, side_info=side_info)
            if hasattr(self.base, "predict_perturbations")
            else None
        )
        if shifts is None:
            raise NotImplementedError(
                f"{self.name}: the adapter cannot be queried for a named "
                "compound, so the fingerprint extension is not defined for it"
            )
        E = np.vstack(
            [np.asarray(fps[c], dtype=np.float32).ravel() for c in train_cpds]
        )
        D = np.vstack(
            [
                np.asarray(shifts[c], dtype=np.float32).ravel() - self.ctrl
                for c in train_cpds
            ]
        )
        reg = Ridge(alpha=1.0).fit(E, D)

        test_perts = obs.iloc[split.test_idx]["perturbation"].to_numpy().astype(str)
        preds = []
        for p in test_perts:
            if p in fps:
                preds.append(
                    self.ctrl
                    + reg.predict(np.asarray(fps[p], np.float32).ravel()[None, :])[0]
                )
            else:
                preds.append(self.ctrl)  # a held compound whose SMILES failed to parse
        return PredResult(np.vstack(preds), self.ctrl)
