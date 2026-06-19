"""The 4 Simple-family baselines (the reference floors).

None of them may use the held-out label — that is the point: they establish the floor that gating
protects. ctrl-pred predicts no effect; cell-mean predicts the average training effect; donor-shift
applies a control->treated shift learned on training; linear-PCA does the same in a denoised PCA
subspace. Ordering: ctrl-pred (≈0) sits below the mean-shift baselines everywhere. linear-PCA vs
cell-mean/donor-shift is regime-dependent: when the perturbation signal lives in the top-k PCs
(e.g. synthetic fixtures, ev50≈90%) linear-PCA ≳ donor-shift; on real T-cell CRISPR where the top-50
PCs hold only ~19% variance, PC-truncation discards real signal so linear-PCA < cell-mean ≈ donor-shift.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..data.schema import CellSet
from ..splits.builder import Split
from .base import BaselineAdapter, PredResult


class CtrlPred(BaselineAdapter):
    name, family = "ctrl-pred", "simple"

    def fit(self, cs, split, side_info=None):
        self.ctrl = self._control_mean(cs, split)

    def predict(self, cs, split, side_info=None) -> PredResult:
        n = len(split.test_idx)
        return PredResult(np.tile(self.ctrl, (n, 1)), self.ctrl)


class CellMean(BaselineAdapter):
    name, family = "cell-mean", "simple"

    def fit(self, cs, split, side_info=None):
        tr = split.train_idx
        treated = tr[~cs.obs.iloc[tr]["is_control"].to_numpy()]
        self.treated_mean = cs.X[treated].mean(axis=0)
        self.ctrl = self._control_mean(cs, split)

    def predict(self, cs, split, side_info=None) -> PredResult:
        n = len(split.test_idx)
        return PredResult(np.tile(self.treated_mean, (n, 1)), self.ctrl)


class DonorShift(BaselineAdapter):
    name, family = "donor-shift", "simple"

    def fit(self, cs, split, side_info=None):
        tr = split.train_idx
        is_ctrl = cs.obs.iloc[tr]["is_control"].to_numpy()
        self.shift = cs.X[tr[~is_ctrl]].mean(0) - cs.X[tr[is_ctrl]].mean(0)
        self.ctrl = self._control_mean(cs, split)

    def predict(self, cs, split, side_info=None) -> PredResult:
        n = len(split.test_idx)
        return PredResult(np.tile(self.ctrl + self.shift, (n, 1)), self.ctrl)


class LinearPCA(BaselineAdapter):
    name, family = "linear-PCA", "simple"

    def fit(self, cs, split, side_info=None):
        tr = split.train_idx
        is_ctrl = cs.obs.iloc[tr]["is_control"].to_numpy()
        k = int(min(50, cs.X[tr].shape[0] - 1, cs.X.shape[1]))
        self.pca = PCA(n_components=max(2, k), random_state=0).fit(cs.X[tr])
        shift = cs.X[tr[~is_ctrl]].mean(0) - cs.X[tr[is_ctrl]].mean(0)
        # Denoise the shift by projecting the *delta* directly onto the top-k principal axes.
        # NOTE: do NOT use pca.inverse_transform(pca.transform(shift)) — that round-trips pca.mean_
        # (projects shift - mean_, then adds mean_ back), leaving the large data-mean residual that
        # lies outside the top-k PCs riding on the shift. On real mean-heavy expression that residual
        # dominates (‖mean_‖ ≫ ‖shift‖) and collapses the delta-correlation to ~0. components_ rows
        # are orthonormal. See results/_qc/qc_C1C3C5_2026-05-26.md.
        V = self.pca.components_                       # (k, n_genes)
        self.shift = V.T @ (V @ shift)
        self.ctrl = self._control_mean(cs, split)

    def predict(self, cs, split, side_info=None) -> PredResult:
        n = len(split.test_idx)
        return PredResult(np.tile(self.ctrl + self.shift, (n, 1)), self.ctrl)


SIMPLE_BASELINES = [CtrlPred, CellMean, DonorShift, LinearPCA]
