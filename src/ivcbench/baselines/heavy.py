"""Heavy-baseline adapters that shell out to a pinned conda env.

The split/audit/metric core stays GPU-free; each heavy model (GEARS, scGPT, scGen, …) runs in its
own conda env via a small *runner* script (`benchmark/model_runners/<model>_runner.py`). The adapter
here only (1) serialises a leak-safe payload built from `split.train_idx` + the held group's control
cells, (2) invokes `<env_python> <runner> <payload.npz> <out.npz>`, and (3) reads back a predicted
mean profile per held perturbation, which it tiles onto the test cells exactly like the Simple
baselines. The runner never sees `split.test_idx` expression — only the held perturbation *labels* to
predict — so the leak boundary is preserved on the model side too.

Env discovery: the existing envs `scgpt` (scGPT 0.2.4 + cell-gears 0.0.2), `scfoundation`
(cell-gears 0.1.2), and `scperturbench_eval` (pertpy 0.10 → scGen, CINEMA-OT) cover the roster.
ENV_PYTHON maps each conda env name to its interpreter; override with $IVCBENCH_<ENV>_PYTHON.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .base import BaselineAdapter, PredResult

_CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", str(Path.home() / "miniconda3")))
_RUNNER_DIR = Path(__file__).resolve().parents[3] / "model_runners"


def env_python(env: str) -> str:
    """Interpreter for a conda env; overridable via $IVCBENCH_<ENV>_PYTHON (upper, '-'→'_')."""
    override = os.environ.get(f"IVCBENCH_{env.upper().replace('-', '_')}_PYTHON")
    if override:
        return override
    return str(_CONDA_ROOT / "envs" / env / "bin" / "python")


class SubprocessAdapter(BaselineAdapter):
    """Base for env-shelling adapters. Subclasses set name/family/conda_env/runner and may set
    `requires_gene_side` (True for 'adapted' models undefined on an unseen gene without it)."""
    conda_env: str = "base"
    runner: str = ""                      # filename in model_runners/
    requires_gene_side: bool = False
    requires_compound_side: bool = False  # True for C5 chemistry models: need side_info['fingerprint']
    timeout_s: int = 3600
    cuda_device: str | None = None        # set by the parallel dispatcher to pin this job's GPU

    def fit(self, cs, split, side_info=None):
        # Training happens inside the runner (own env/GPU); here we just hold the leak-safe context.
        self._cs, self._split, self._side = cs, split, side_info
        self.ctrl = self._control_mean(cs, split)

    def _build_payload(self, cs, split, side_info):
        tr = split.train_idx
        # held perturbations actually present in the held-out test cells (labels only, no expression)
        test_perts = cs.obs.iloc[split.test_idx]["perturbation"].to_numpy().astype(str)
        payload = dict(
            X_train=cs.X[tr].astype(np.float32),
            is_control_train=cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool),
            pert_train=cs.obs.iloc[tr]["perturbation"].to_numpy().astype(str),
            X_ctrl_inf=cs.X[split.inference_input_idx].astype(np.float32)
            if len(split.inference_input_idx) else cs.X[tr][cs.obs.iloc[tr]["is_control"].to_numpy()].astype(np.float32),
            # str (unicode '<U'), NOT object dtype — object arrays pickle with the host numpy's
            # internal module path (numpy._core on ≥2.0) and fail to load in envs pinned to old
            # numpy (e.g. ivc-cpa's 1.23). Unicode arrays are pickle-free and cross-version safe.
            genes=np.asarray([str(g) for g in cs.var_names]),
            test_perts=test_perts,                       # one row per test cell (prediction is tiled per pert)
            model=self.name,
        )
        gemb = (side_info or {}).get("gene_embedding")
        if gemb is not None:
            payload["gene_embedding_keys"] = np.asarray([str(g) for g in gemb.keys()])
            payload["gene_embedding_vals"] = np.asarray(list(gemb.values()), dtype=np.float32)
        # compound-side representation (C5): Morgan fingerprint per compound. Serialize as a unicode key
        # array + a rectangular float matrix (same cross-version-safe convention as above). Only emit
        # when all fingerprints share a length (a compound whose SMILES failed to parse is simply absent).
        fps = (side_info or {}).get("fingerprint")
        if fps:
            keys = [str(c) for c in fps.keys()]
            vals = [np.asarray(v, dtype=np.float32).ravel() for v in fps.values()]
            if vals and len({v.shape[0] for v in vals}) == 1:
                payload["fingerprint_keys"] = np.asarray(keys)
                payload["fingerprint_vals"] = np.asarray(vals, dtype=np.float32)
        return payload

    def predict(self, cs, split, side_info=None) -> PredResult:
        if self.requires_gene_side and not (side_info or {}).get("gene_embedding"):
            raise NotImplementedError(f"{self.name}: adapted model needs a gene-side representation "
                                      "(side_info['gene_embedding']); not provided for this split.")
        if self.requires_compound_side and not (side_info or {}).get("fingerprint"):
            raise NotImplementedError(f"{self.name}: C5 chemistry model needs a compound-side "
                                      "representation (side_info['fingerprint']); not provided.")
        runner = _RUNNER_DIR / self.runner
        if not runner.exists():
            raise NotImplementedError(f"{self.name}: runner {runner} not found.")
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
            np.savez(inp, **self._build_payload(cs, split, side_info), allow_pickle=True)
            env = os.environ.copy()
            if self.cuda_device is not None:           # pin this job's GPU (parallel dispatch)
                env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
            proc = subprocess.run([env_python(self.conda_env), str(runner), str(inp), str(out)],
                                  capture_output=True, text=True, timeout=self.timeout_s, env=env)
            if proc.returncode != 0 or not out.exists():
                # surface the traceback, not just the trailing log noise: prefer the last
                # Error/Exception line plus a wide stderr tail.
                err = proc.stderr or ""
                key = [ln for ln in err.splitlines()
                       if any(k in ln for k in ("Error", "Exception", "Traceback", "assert"))]
                raise RuntimeError(f"{self.name} runner failed (rc={proc.returncode}):\n"
                                   + ("… " + key[-1] + "\n" if key else "")
                                   + err[-4000:])
            r = np.load(out, allow_pickle=True)
            pred_by_pert = {str(k): v for k, v in zip(r["pred_perts"], r["pred_means"])}
        # tile each test cell's predicted mean by its perturbation label (fall back to control)
        test_perts = cs.obs.iloc[split.test_idx]["perturbation"].to_numpy().astype(str)
        pred = np.vstack([pred_by_pert.get(p, self.ctrl) for p in test_perts])
        return PredResult(pred, self.ctrl)


class GEARS(SubprocessAdapter):
    name, family, gpu = "GEARS", "graph", True
    conda_env, runner = "scgpt", "gears_runner.py"


class AttentionPert(SubprocessAdapter):
    name, family, gpu = "AttentionPert", "graph", True
    conda_env, runner = "scgpt", "attentionpert_runner.py"
    timeout_s = 7200                                   # Chen needs >3600s even with the cell cap


class ScGPT(SubprocessAdapter):
    name, family, gpu = "scGPT", "foundation", True
    conda_env, runner = "scgpt", "scgpt_runner.py"


class ScFoundation(SubprocessAdapter):
    """scFoundation — frozen 19264-gene foundation encoder (`cell` checkpoint) + a GEARS-style
    fine-tune head ([frozen cell embedding ‖ perturbed-gene one-hot] → response-gene profile, ~15
    epochs). Native unseen-gene capability (the held gene is predicted from its one-hot conditioning +
    a control cell's frozen embedding, never from held-gene expression). 2nd foundation model on
    C3_LO_gene alongside scGPT. Response panel + PCA basis fit on the train fold only (leak-safe),
    refit per fold."""
    name, family, gpu = "scFoundation", "foundation", True
    conda_env, runner = "scfoundation", "scfoundation_runner.py"
    timeout_s = 7200                                   # per-cell frozen embedding fwd is the cost driver


class UCE(SubprocessAdapter):
    name, family, gpu = "UCE", "foundation", True
    conda_env, runner = "scgpt", "uce_runner.py"


class ScGen(SubprocessAdapter):
    name, family, gpu = "scGen", "latent", True
    # gene-side repr (the `adapted` extension) is built INSIDE the runner (leak-safe control-only
    # PCA gene-loadings), so no external side_info is required.
    conda_env, runner, requires_gene_side = "scperturbench_eval", "scgen_runner.py", False


class ScGenC5(SubprocessAdapter):
    """scGen adapted to C5: latent δ regressed on the compound Morgan fingerprint (adapted* on
    C5_unseen_cpd). name='scGen' → registry status for C5; distinct C5 runner."""
    name, family, gpu = "scGen", "latent", True
    conda_env, runner, requires_compound_side = "scperturbench_eval", "scgen_c5_runner.py", True


class ScGenC1(SubprocessAdapter):
    """scGen for C1 cytokine-response (Kang IFN-β cross-cell-type): classic latent δ-arithmetic, seen
    cytokine, held cell type. name='scGen' → registry C1_LOCT status (applicable)."""
    name, family, gpu = "scGen", "latent", True
    conda_env, runner = "scperturbench_eval", "scgen_c1_runner.py"


class CPAC1(SubprocessAdapter):
    """CPA for C1 cytokine-response: classic latent δ-arithmetic (seen cytokine, held cell type)."""
    name, family, gpu = "CPA", "latent", True
    conda_env, runner, timeout_s = "ivc-cpa", "cpa_c1_runner.py", 7200


class CPA(SubprocessAdapter):
    name, family, gpu = "CPA", "latent", True
    # dedicated env (cpa-tools pins an old torch/scvi stack); gene-side repr built inside the runner.
    conda_env, runner, requires_gene_side = "ivc-cpa", "cpa_runner.py", False
    timeout_s = 7200                                   # Chen (60k-capped) needs >3600s for 40-60 epochs


class CPAchem(SubprocessAdapter):
    """chemCPA — CPA conditioned on the compound Morgan fingerprint (C5). name='CPA' so the registry
    resolves it to `applicable` on C5_unseen_cpd (the canonical chemistry headline model); a distinct
    C5 runner does the fingerprint→δ regression. Use this in the C5 roster, the gene-axis CPA in C3."""
    name, family, gpu = "CPA", "latent", True
    conda_env, runner, requires_compound_side = "ivc-cpa", "cpa_c5_runner.py", True
    timeout_s = 7200


class CINEMAOT(SubprocessAdapter):
    name, family, gpu = "CINEMA-OT", "ot", True
    # not_defined† on C3_LO_gene → runs as a perturbation-agnostic OT FLOOR (excluded from ranking).
    # The floor doesn't need a gene-side repr (same global OT shift for every held gene).
    conda_env, runner, requires_gene_side = "scperturbench_eval", "cinemaot_runner.py", False


class CellOT(SubprocessAdapter):
    name, family, gpu = "CellOT", "ot", True
    conda_env, runner, requires_gene_side = "scperturbench_eval", "cellot_runner.py", True


class ScPRAM(SubprocessAdapter):
    """scPRAM (Jiang et al. 2024, Bioinformatics btae265) — 2nd CONDITIONED Optimal-Transport model
    alongside CellOT. VAE latent space + OT cell-matching + per-cell attention over reference deltas,
    conditioned on (cell_type, condition). It predicts a HELD cell type's (Kang C1 LOCT) or HELD donor's
    (Soskic C2 LODO) stimulated state from that unit's OWN control cells; the perturbation is SEEN, the
    held axis is the cell/donor. Needs PAIRED ctrl/stimulation, so it is registered on the Fig1
    OT-STRONG paired-stimulation cells (Cytokine C1_LOCT, Donor C2_LODO) and NOT on Frangieh (CRISPR KO,
    not paired stimulation). Runs in the dedicated `ivc-scpram` env (`pip install scpram --no-deps` on
    the ivc-cpa torch2.0/cu117 stack). VAE + OT matching refit per fold (leak-safe): the runner trains on
    the train fold only and the held unit's stimulated expression never enters training.
    Official: github.com/jiang-q19/scPRAM, PyPI scpram 0.0.3 (MIT)."""
    name, family, gpu = "scPRAM", "optimal-transport", True
    conda_env, runner = "ivc-scpram", "scpram_runner.py"
    timeout_s = 7200


class STATEc5(SubprocessAdapter):
    """STATE adapted to C5: perturbation_features = compound Morgan fingerprint (adapted* on
    C5_unseen_cpd). name='STATE' → registry status; distinct C5 runner; from-scratch ST lower bound."""
    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner, requires_compound_side = "ivc-state", "state_c5_runner.py", True
    timeout_s = 7200


class STATE(SubprocessAdapter):
    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner = "ivc-state", "state_runner.py"   # arc-state; ST predicts held genes (fewshot)
    timeout_s = 7200


class PertAdapt(SubprocessAdapter):
    """PertAdapt (Bai et al. 2025) — 2nd Hybrid-family model alongside STATE. FROZEN scFoundation
    backbone + condition-sensitive perturbation adapter (gene-similarity/GO–masked self-attention) +
    adaptive DE-reweighting loss (`loss_adapt`). Native unseen-gene capability: the held gene is
    predicted from its learned pert embedding + a frozen control-cell embedding through the GO-masked
    adapter, never from held-gene expression. Runs in the `scfoundation` env (reuses the local
    models.ckpt; no weights redistributed). Response panel + GO mask + per-pert DE indices are fit on
    the train fold only (leak-safe), refit per fold. The GO mask is reconstructed from the local
    gene2go (the authors' exact go_mask_19264.npz is OneDrive-gated) — faithful reimplementation; the
    *published-anchor* reproduction is gated separately (scripts/pertadapt_validate.py)."""
    name, family, gpu = "PertAdapt", "hybrid", True
    conda_env, runner = "scfoundation", "pertadapt_runner.py"
    timeout_s = 7200                                   # per-cell frozen embedding fwd is the cost driver


# applicable-on-C3 (native unseen-gene) first; adapted/OT need a gene-side repr (requires_gene_side)
HEAVY_BASELINES = [GEARS, AttentionPert, ScGPT, ScFoundation, UCE, STATE, PertAdapt, ScGen, CPA,
                   CINEMAOT, CellOT, ScPRAM]
