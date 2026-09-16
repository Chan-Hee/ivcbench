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


def _assert_runner_dir():
    """Fail loudly when the imported package and its runner tree are the stale copy.

    Two trees exist: ivcbench/ (42 runners, current) and benchmark/ (20 runners, kept as historical
    evidence). benchmark/.venv installs ivcbench editable from benchmark/src, so running any entry
    point with that interpreter silently resolves _RUNNER_DIR to benchmark/model_runners: the
    STATE output-selection fix is absent there, the newer runners do not exist at all, and a
    re-run reproduces the historical numbers exactly. That cost a full round of GPU jobs before it
    was noticed, so it is now an error rather than a silent fallback.
    """
    marker = _RUNNER_DIR / "state_output.py"
    if not marker.exists():
        raise RuntimeError(
            f"Stale runner tree: {_RUNNER_DIR} has no state_output.py, so this is the historical "
            f"benchmark/ copy. Run with ivcbench/.venv/bin/python (or set PYTHONPATH to "
            f"ivcbench/src) so the current runners are used."
        )


_assert_runner_dir()


def env_python(env: str) -> str:
    """Interpreter for a conda env; overridable via $IVCBENCH_<ENV>_PYTHON (upper, '-'→'_')."""
    override = os.environ.get(f"IVCBENCH_{env.upper().replace('-', '_')}_PYTHON")
    if override:
        return override
    return str(_CONDA_ROOT / "envs" / env / "bin" / "python")


class SubprocessAdapter(BaselineAdapter):
    """Base for env-shelling adapters. Subclasses set name/family/conda_env/runner and may set
    `requires_gene_side` (True for 'adapted' models undefined on an unseen gene without it).
    """

    conda_env: str = "base"
    runner: str = ""  # filename in model_runners/
    pred_key_is_group: bool = (
        False  # runner keys its output by the held group, not by pert label
    )
    requires_gene_side: bool = False
    requires_compound_side: bool = (
        False  # True for C5 chemistry models: need side_info['fingerprint']
    )
    # A fixed ceiling has to survive the WORST contention the box will see, not the best. Two
    # PertAdapt T3 units died at epoch 11 and 10 of 15 against an 8 h ceiling after slowing from
    # 27 to 46 minutes an epoch when sixteen PerturbNet shards came up alongside them -- seven
    # hours of training thrown away for a limit, not a defect. $IVCBENCH_TIMEOUT_SCALE multiplies
    # every adapter's budget so a loaded box can be given headroom without editing each class.
    timeout_s: int = 3600
    cuda_device: str | None = (
        None  # set by the parallel dispatcher to pin this job's GPU
    )

    def _budget(self) -> int:
        """timeout_s scaled by $IVCBENCH_TIMEOUT_SCALE (default 1.0), read at call time."""
        import os as _os

        try:
            scale = float(_os.environ.get("IVCBENCH_TIMEOUT_SCALE", "1"))
        except ValueError:
            scale = 1.0
        return int(self.timeout_s * max(scale, 1.0))

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
            X_ctrl_inf=(
                cs.X[split.inference_input_idx].astype(np.float32)
                if len(split.inference_input_idx)
                else cs.X[tr][cs.obs.iloc[tr]["is_control"].to_numpy()].astype(
                    np.float32
                )
            ),
            # str (unicode '<U'), NOT object dtype — object arrays pickle with the host numpy's
            # internal module path (numpy._core on ≥2.0) and fail to load in envs pinned to old
            # numpy (e.g. ivc-cpa's 1.23). Unicode arrays are pickle-free and cross-version safe.
            genes=np.asarray([str(g) for g in cs.var_names]),
            test_perts=test_perts,  # one row per test cell (prediction is tiled per pert)
            model=self.name,
        )
        # Training-side group labels (the biological unit of the split, e.g. lineage or donor).
        # Some published methods -- SCREEN, for instance -- are designed around a multi-group panel
        # and are handicapped if every training cell is collapsed into one group. These are
        # TRAINING-side labels only; the held unit's identity is not revealed by them.
        for col in ("cell_type_coarse", "donor_id"):
            if col in cs.obs.columns:
                payload["group_train"] = cs.obs.iloc[tr][col].to_numpy().astype(str)
                break
        # Held-GROUP context keys. Runners written for the cell-context / donor entry points consume
        # the split's group structure directly (which lineage and which donor each training and each
        # inference cell belongs to, and which unit is held out) rather than re-deriving it. All of
        # these are TRAINING-side or inference-CONTROL-side labels; no held-out response is exposed.
        _inf = (
            split.inference_input_idx
            if len(split.inference_input_idx)
            else np.asarray(tr)[cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)]
        )
        for _src, _dst in (("cell_type_coarse", "celltype"), ("donor_id", "gem")):
            if _src in cs.obs.columns:
                payload[f"{_dst}_train"] = cs.obs.iloc[tr][_src].astype(str).to_numpy()
                payload[f"{_dst}_inf"] = cs.obs.iloc[_inf][_src].astype(str).to_numpy()
        _held = list(getattr(getattr(split, "spec", None), "held_values", None) or [])
        if _held:
            payload["held_lineage"] = str(_held[0])
            payload["held_label"] = str(_held[0])

        gemb = (side_info or {}).get("gene_embedding")
        if gemb is not None:
            payload["gene_embedding_keys"] = np.asarray([str(g) for g in gemb.keys()])
            payload["gene_embedding_vals"] = np.asarray(
                list(gemb.values()), dtype=np.float32
            )
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

    def runner_for(self, split):
        """Which runner this split needs. A cluster roster is shared by every split in the cluster,
        so an adapter written for one held axis was being invoked on the other; C5's compound-axis
        scGen runner ran on the held-LINEAGE split and its cell had to be withdrawn as non-native.
        Subclasses that publish one operation per held axis override this."""
        return self.runner

    def _invoke(self, cs, split, side_info, test_perts_override=None):
        """Run the model in its own env and return {perturbation label -> predicted profile}."""
        runner = _RUNNER_DIR / self.runner_for(split)
        if not runner.exists():
            raise NotImplementedError(f"{self.name}: runner {runner} not found.")
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
            payload = self._build_payload(cs, split, side_info)
            if test_perts_override is not None:
                payload["test_perts"] = np.asarray(
                    [str(p) for p in test_perts_override]
                )
            np.savez(inp, **payload, allow_pickle=True)
            env = os.environ.copy()
            if self.cuda_device is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
            # capture_output buffers the runner's stderr in a pipe that is only readable once the
            # process exits, so a long run is invisible until it succeeds or times out. That is how
            # three scFoundation units burned 4, 4.4 and 8 GPU-hours each before anyone could see
            # that batch_size=2 was the reason. Tee stderr to a file instead: the same text still
            # reaches proc.stderr for the error paths below, and it is readable WHILE the job runs.
            _ld = os.environ.get("IVCBENCH_RUNNER_LOG_DIR")
            # repo root: .../ivcbench/src/ivcbench/baselines/heavy.py -> parents[3]
            log_dir = Path(_ld) if _ld else Path(__file__).resolve().parents[3] / "logs" / "runners"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                live = log_dir / f"{self.name.replace('/', '-')}_{Path(runner).stem}_{os.getpid()}.log"
                fh = live.open("a", buffering=1)
            except Exception:                      # never let logging stop a run
                fh, live = None, None
            try:
                if fh is not None:
                    fh.write(f"\n=== {self.name} :: {Path(runner).name} :: pid {os.getpid()} ===\n")
                proc = subprocess.run(
                    [env_python(self.conda_env), str(runner), str(inp), str(out)],
                    stdout=subprocess.PIPE,
                    stderr=fh if fh is not None else subprocess.PIPE,
                    text=True,
                    timeout=self._budget(),
                    env=env,
                )
                if fh is not None:
                    fh.flush()
                    # the error paths below read proc.stderr, so give them the file's contents
                    proc = subprocess.CompletedProcess(
                        proc.args, proc.returncode, proc.stdout,
                        live.read_text(errors="replace")[-20000:] if live.exists() else "")
            except subprocess.TimeoutExpired:
                if fh is not None:
                    fh.write(f"=== TIMED OUT after {self._budget()}s ===\n"); fh.flush()
                raise
            finally:
                if fh is not None:
                    fh.close()
            if proc.returncode != 0 or not out.exists():
                err = proc.stderr or ""
                key = [
                    ln
                    for ln in err.splitlines()
                    if any(
                        k in ln for k in ("Error", "Exception", "Traceback", "assert")
                    )
                ]
                raise RuntimeError(
                    f"{self.name} runner failed (rc={proc.returncode}):\n"
                    + ("… " + key[-1] + "\n" if key else "")
                    + err[-4000:]
                )
            r = np.load(out, allow_pickle=True)
            return {str(k): v for k, v in zip(r["pred_perts"], r["pred_means"])}

    def predict_perturbations(self, cs, split, perts, side_info=None):
        """Predicted profile for each NAMED perturbation. Used by the unseen-compound extension to
        harvest the model's own response for every training compound before regressing it on
        chemistry; the held perturbations are not among them, so no leak boundary is crossed.
        """
        got = self._invoke(cs, split, side_info, test_perts_override=list(perts))
        missing = [p for p in perts if p not in got]
        if missing:
            raise RuntimeError(
                f"{self.name}: runner returned no profile for {len(missing)} of "
                f"{len(perts)} requested perturbations, e.g. {missing[:3]}"
            )
        return got

    def predict(self, cs, split, side_info=None) -> PredResult:
        if self.requires_gene_side and not (side_info or {}).get("gene_embedding"):
            raise NotImplementedError(
                f"{self.name}: adapted model needs a gene-side representation "
                "(side_info['gene_embedding']); not provided for this split."
            )
        if self.requires_compound_side and not (side_info or {}).get("fingerprint"):
            raise NotImplementedError(
                f"{self.name}: C5 chemistry model needs a compound-side "
                "representation (side_info['fingerprint']); not provided."
            )
        pred_by_pert = self._invoke(cs, split, side_info)
        test_perts = cs.obs.iloc[split.test_idx]["perturbation"].to_numpy().astype(str)
        if not pred_by_pert:
            raise RuntimeError(
                f"{self.name}: the runner returned no predicted profiles."
            )
        if self.pred_key_is_group:
            # Held-GROUP runners key their output by the held unit, not by the test cells'
            # perturbation label. When the runner produced ONE profile that is the whole
            # prediction and it is tiled over the test cells.
            #
            # When it produced SEVERAL, they are keyed '<held unit>::<stratum value>' -- STATE
            # emits 'stim::1256' against the stratum 'donor_id=1256', PertAdapt 'D348::CD4_Naive'
            # against 'cell_type_coarse=CD4_Naive'. Averaging those together used to discard the
            # keying: neither profile survived, and the model was scored on a blend it never
            # predicted. Map them onto their own strata instead.
            profiles = {k: np.asarray(v, dtype=np.float32).ravel() for k, v in pred_by_pert.items()}
            n_strata = len(set(map(str, np.asarray(split.test_strata))))
            if len(profiles) == 1:
                prof = next(iter(profiles.values()))
                # One profile tiled over every test cell certifies a prediction for strata the
                # runner never produced -- that is how a pooled map came to be reported per
                # compound. But it is only a pathology when the strata lie on the axis the model
                # was asked to condition on. What the stratum labels name decides that: on a
                # held-group split they are a nuisance axis ('donor_id=1256',
                # 'cell_type_coarse=CD4_Naive') and one profile for the held unit IS the whole
                # prediction -- the universal floor members tile identically on those same
                # splits, eval/bundle.py documents the repeated-profile case as required, and
                # test_constant_prediction_scores_zero.py asserts it. Refusing there broke the
                # regeneration path for six deposited cells (scGPT, scFoundation and CellFlow on
                # T1 and T2), whose runners return a single profile by design.
                #
                # 'perturbation=...' is the other case: those strata ARE the held entity, so a
                # single profile means the rest went unpredicted. Refuse, as before. An
                # unrecognised label shape is refused too -- the permissive branch has to be
                # positively established, not assumed.
                axes = {str(s).split("=", 1)[0] for s in np.asarray(split.test_strata)}
                nuisance = axes and axes <= {"donor_id", "cell_type_coarse"}
                if n_strata > 1 and not nuisance:
                    raise RuntimeError(
                        f"{self.name}: the runner returned ONE profile for a split with "
                        f"{n_strata} strata on {sorted(axes)}. Tiling it would score "
                        f"{n_strata - 1} strata on a prediction that was never made. Fix the "
                        "runner's keying, or have it decline the strata it cannot produce."
                    )
                return PredResult(
                    np.repeat(prof[None, :], len(test_perts), axis=0),
                    self.ctrl,
                    declined=np.zeros(len(test_perts), dtype=bool),
                )
            strata = np.asarray(split.test_strata).astype(str)
            # stratum labels are spelled 'field=value'; the runner keys on the value alone
            by_value = {}
            for k, v in profiles.items():
                by_value.setdefault(k.split("::")[-1], v)
            rows, declined = [], []
            unmatched = set()
            for st in strata:
                value = st.split("=", 1)[1] if "=" in st else st
                prof = by_value.get(value)
                if prof is None:
                    unmatched.add(st)
                    rows.append(self.ctrl)
                    declined.append(True)
                else:
                    rows.append(prof)
                    declined.append(False)
            if len(unmatched) == len(set(strata)):
                raise RuntimeError(
                    f"{self.name}: the runner returned {len(profiles)} keyed profiles "
                    f"{sorted(profiles)[:4]} but none matched a test stratum "
                    f"{sorted(set(strata))[:4]}; averaging them would score a blend the model "
                    "never predicted."
                )
            if unmatched:
                print(
                    f"[decline] {self.name}: {len(unmatched)} of {len(set(strata))} strata had no "
                    f"keyed profile: {', '.join(sorted(unmatched)[:8])}",
                    file=sys.stderr,
                    flush=True,
                )
            return PredResult(
                np.vstack(rows), self.ctrl, declined=np.asarray(declined, dtype=bool)
            )
        matched = sum(p in pred_by_pert for p in test_perts)
        if matched == 0:
            # Silently substituting the control mean here produced a plausible number that was in
            # fact the ctrl-pred floor. Never fall back without saying so.
            raise RuntimeError(
                f"{self.name}: none of the {len(set(test_perts))} test perturbation"
                f" labels {sorted(set(test_perts))[:4]} matched the runner's predicted"
                f" keys {sorted(pred_by_pert)[:4]}; the prediction would be the control"
                " mean."
            )
        # A label with no predicted profile keeps the control mean. That is a legitimate outcome --
        # a model may not support a target -- but it is NOT a prediction, and it used to be
        # indistinguishable from one: the audit found declined rows scored as real responses in
        # T3/T4/T5u. Record which rows they are, and say so on stderr, so the deposited bundle and
        # the census carry the coverage instead of hiding it.
        declined = np.array([p not in pred_by_pert for p in test_perts], dtype=bool)
        missing = sorted({p for p in test_perts if p not in pred_by_pert})
        if missing:
            print(
                f"[decline] {self.name}: {len(missing)} of "
                f"{len(set(test_perts))} requested perturbations returned no profile "
                f"({declined.sum()}/{len(declined)} test rows fall back to the control mean): "
                + ", ".join(missing[:12])
                + (" ..." if len(missing) > 12 else ""),
                file=sys.stderr,
                flush=True,
            )
        pred = np.vstack([pred_by_pert.get(p, self.ctrl) for p in test_perts])
        return PredResult(pred, self.ctrl, declined=declined)


class GEARS(SubprocessAdapter):
    name, family, gpu = "GEARS", "graph", True
    conda_env, runner = "scgpt", "gears_runner.py"


class AttentionPert(SubprocessAdapter):
    name, family, gpu = "AttentionPert", "graph", True
    conda_env, runner = "scgpt", "attentionpert_runner.py"
    timeout_s = 7200  # Chen needs >3600s even with the cell cap


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
    timeout_s = 7200  # per-cell frozen embedding fwd is the cost driver


class ScFoundationC1(SubprocessAdapter):
    """scFoundation on a CELL-CONTEXT split (C1 Kang LOCT / C5 LOCT): frozen encoder + trained decoder
    head, with the seen perturbation applied as a latent shift estimated on the training units. Same
    frozen-representation regime as the unseen-gene runner; the task lives in the head. name='scFoundation'
    -> registry C1_LOCT status (applicable)."""

    name, family, gpu = "scFoundation", "foundation", True
    conda_env, runner = "scfoundation", "scfoundation_c1_runner.py"
    timeout_s = 7200
    # The runner collapses the perturbation axis into ONE global "exposed" profile (that is the
    # documented adapter: a single latent shift for the seen exposure). It therefore keys its output
    # by the held GROUP, not by a perturbation label. Without this flag heavy.py matched the single
    # key against the split's perturbation labels, so on C5_loct (141 compounds) 140 of 141 strata
    # silently fell back to the control mean and the cell scored as ctrl-pred.
    pred_key_is_group = True


class BiolordC1(SubprocessAdapter):
    """Biolord (Piran et al., Nat Biotechnol 2024) on the cell-context split: decomposed latent space with
    the seen perturbation as a disentangled categorical attribute, generating the stimulated counterpart of
    the held unit's own control cells. Added in revision as a recent conditioned predictor on T1.
    """

    name, family, gpu = "Biolord", "latent", True
    conda_env, runner = "ctx-biolord", "biolord_c1_runner.py"
    timeout_s = 7200


class BiolordC5(SubprocessAdapter):
    """Biolord on the OP3 COMPOUND cluster — NATIVE, not adapted. The published sci-Plex 3 model
    conditions on chemistry directly ("We use RDKit chemically informed features embedding of the
    drugs, as well as the dosage as ordered attributes"; biolord_reproducibility
    scripts/biolord/sciplex3: ordered_attributes_keys=["rdkit2d_dose"],
    categorical_attributes_keys=["cell_type"]), so the compound enters as a CONTINUOUS attribute and
    an unseen compound is simply an unseen value of it — no external fingerprint→response regression.
    One runner serves both C5 splits: T5c (held lineage, compounds seen — cell_type categorical
    omitted, lineage carried by the residual state) and T5u (held compounds, all lineages seen —
    cell_type registered). requires_compound_side because the RDKit vector IS the model's attribute.
    """

    name, family, gpu = "Biolord", "latent", True
    conda_env, runner = "ctx-biolord", "biolord_c5_runner.py"
    requires_compound_side = True
    timeout_s = 7200


class ScGPTC1(SubprocessAdapter):
    """scGPT on a CELL-CONTEXT split: the TransformerGenerator fine-tuned end-to-end from the pretrained
    checkpoint, as on the unseen-gene axis, with the seen stimulus entering as a single global condition
    flag. name='scGPT' -> registry C1_LOCT status (applicable)."""

    name, family, gpu = "scGPT", "foundation", True
    conda_env, runner = "scgpt", "scgpt_c1_runner.py"
    timeout_s = 7200
    # Same as ScFoundationC1: one global condition flag -> one profile for the held group. See there.
    pred_key_is_group = True


class ScreenC1(SubprocessAdapter):
    """SCREEN (Xu et al., Front Comput Sci 2024): optimal transport in a VAE latent space, predicting
    the perturbed counterpart of a held-out group from that group's own control cells. The published
    interface is exactly this task, so it runs native with the authors' default settings. Added at
    review for a broader recent-method panel, and because it tests whether the donor-axis OT result
    is CellOT-specific or general to the optimal-transport family."""

    name, family, gpu = "SCREEN", "ot", True
    conda_env, runner = "cellot", "screen_c1_runner.py"
    timeout_s = 14400


class StateC1(SubprocessAdapter):
    """STATE on a held-GROUP split (cell-context / donor). The unseen-gene runner predicts a held
    gene from its perturbation features; here the perturbation is SEEN and the held axis is a group,
    so the published cell-state transition is driven by the group context instead. Same released
    model and env, different task entry point."""

    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner = "ivc-state", "state_c1_runner.py"
    pred_key_is_group, timeout_s = True, 7200


class PertAdaptC1(SubprocessAdapter):
    """PertAdapt on a held-GROUP split. The perturbation is SEEN, so the GO-masked adapter carries a
    single learned stimulus token and the group context routes the transition; the held group's
    perturbed cells never enter training. Same frozen scFoundation backbone and env as PertAdapt.
    """

    name, family, gpu = "PertAdapt", "hybrid", True
    conda_env, runner = "scfoundation", "pertadapt_soskic_runner.py"
    pred_key_is_group, timeout_s = True, 7200


# UCE (Rosen et al. 2023) is encoder-only: the released model has no decoder, so it cannot emit a
# predicted expression profile for any task in this benchmark and no runner exists for it. It is kept
# in the surveyed inventory and in the coverage table (state: no expression output) but is not a
# runnable adapter, so no class is defined here.


class ScGen(SubprocessAdapter):
    name, family, gpu = "scGen", "latent", True
    # gene-side repr (the `adapted` extension) is built INSIDE the runner (leak-safe control-only
    # PCA gene-loadings), so no external side_info is required.
    conda_env, runner, requires_gene_side = (
        "scperturbench_eval",
        "scgen_runner.py",
        False,
    )


class ScGenC5(SubprocessAdapter):
    """scGen on C5. The held axis decides which published operation applies.

    Held LINEAGE (C5_loct_*): every scored compound is present in the training lineages, which is
    scGen's own setting — a condition key seen in training, decoded on the held group's own control
    cells. NATIVE, via scgen_c5_loct_runner.py.

    Held COMPOUND (C5_global_compound_holdout): the molecule never appears in training, so reaching
    it needs a fingerprint-to-latent-shift regression that published scGen does not contain. That
    cell is not native and is not carried in the census.
    """

    name, family, gpu = "scGen", "latent", True
    conda_env, runner, requires_compound_side = (
        "scperturbench_eval",
        "scgen_c5_runner.py",
        True,
    )

    def runner_for(self, split):
        name = getattr(getattr(split, "spec", None), "name", "") or ""
        return "scgen_c5_loct_runner.py" if name.startswith("C5_loct") else self.runner


class ScGenC1(SubprocessAdapter):
    """scGen for C1 cytokine-response (Kang IFN-β cross-cell-type): classic latent δ-arithmetic, seen
    cytokine, held cell type. name='scGen' → registry C1_LOCT status (applicable)."""

    name, family, gpu = "scGen", "latent", True
    conda_env, runner = "scperturbench_eval", "scgen_c1_runner.py"


class CINEMAOTC1(SubprocessAdapter):
    """CINEMA-OT on a held-GROUP split through its PUBLISHED per-condition call.

    The existing `CINEMAOT` adapter runs a perturbation-agnostic reduction, which is the intended
    diagnostic on the unseen-entity tasks but is not the published operation on T1/T2, where the
    stimulus is seen and the held axis is a group. Here `expr_label` selects the condition and the
    control-indexed return supplies the barycentric counterfactual directly."""

    name, family, gpu = "CINEMA-OT", "ot", False
    conda_env, runner = "scperturbench_eval", "cinemaot_c1_runner.py"
    timeout_s = 14400


class StateC1Split(SubprocessAdapter):
    """STATE on a held-GROUP split with the fitting set and the inference set kept apart.

    `StateC1` builds one AnnData and routes the held cell type through [zeroshot]; the installed
    loader then folds that cell type's observational cells into the fit
    (cell_load/.../perturbation_dataloader.py:909-916). No held response is exposed, but the fit is
    then not train_idx. This adapter trains on a training-only dataset and supplies the held unit's
    own controls to `state tx predict --toml`, which the released CLI supports directly."""

    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner = "ivc-state", "state_c1_split_runner.py"
    pred_key_is_group = True
    timeout_s = 14400


class StateC5cSplit(SubprocessAdapter):
    """STATE on the held-LINEAGE compound split (T5c), fitting set and inference set kept apart.

    `STATEc5` serves the unseen-COMPOUND split and is handed the same payload shape here, so on T5c
    it puts the SEEN compounds into the fewshot `test` list -- which cell_load then withholds from
    training (perturbation_dataloader.py:768-787), turning a lineage holdout into a compound-label
    holdout. It also collapses obs['cell_type'] to a constant and draws the query cells from the
    training control pool instead of the held lineage's own. This adapter trains on the training
    lineages with every compound present and predicts from the held lineage's own controls through
    `state tx predict --toml`. See model_runners/state_c5c_split_runner.py."""

    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner = "ivc-state", "state_c5c_split_runner.py"
    requires_compound_side = True
    pred_key_is_group = False
    timeout_s = 14400


class PerturbNetC1(SubprocessAdapter):
    """PerturbNet on a held-GROUP split (T1 lineage, T2 donor) through the authors' CATEGORICAL
    conditioning variant and the released cINN. The held axis is a group and the stimulus is seen,
    so no unseen-entity encoder (ChemicalVAE / GenotypeVAE) is required -- the held unit's identity
    is carried by its own control cells. Keyed by the requested perturbation label, so the adapter
    matches labels rather than averaging one group profile."""

    name, family, gpu = "PerturbNet", "generative", True
    conda_env, runner = "ivc-perturbnet", "perturbnet_c1_runner.py"
    timeout_s = 14400


class PertAdaptC3(SubprocessAdapter):
    """PertAdapt on the unseen-gene split through the PUBLISHED graph path (the same backbone and
    GO mask the T4 runner uses), rather than the author-written head that produced the withdrawn
    T2 cell."""

    name, family, gpu = "PertAdapt", "hybrid", True
    # the same env the T4 PertAdapt adapter uses: this runner shares the frozen scFoundation
    # backbone, and `scgpt` lacks omegaconf / local_attention.
    conda_env, runner = "scfoundation", "pertadapt_c3_runner.py"
    # 8 h, not 4, and run sharded. Both T4 units stopped at exactly 14,401 s with ran=False and no
    # bundle; on T3 the largest dataset (chen) finished at 15,280 s total, i.e. just inside. A
    # budget a unit only just fits is a budget that reports a timeout as a model limitation the
    # next time the machine is busier.
    timeout_s = 28800


class ScFoundationGene(SubprocessAdapter):
    """scFoundation on the unseen-gene / unseen-KO splits through its RELEASED GEARS pathway, which
    is the operation the plan specifies for these cells -- not an author-written task head."""

    name, family, gpu = "scFoundation", "foundation", True
    # `scgpt` is missing omegaconf and local_attention, which scFoundation's released GEARS path
    # imports; `scfoundation` has both and is the env the other scFoundation adapters use.
    conda_env, runner = "scfoundation", "scfoundation_gene_runner.py"
    # 8 h, not 4. A unit is a 15-epoch GEARS-style head on the frozen 19,264-gene encoder at batch
    # size 2; alone it fits in four hours, but with two other jobs on the card it gets about a third
    # of the SM and does not. Both T3 and T4 spent four hours per unit and returned
    # TimeoutExpired with ran=False and no bundle -- a timeout that reads in the CSV exactly like a
    # model that cannot do the task. Run these sharded (--chunk i n, one unit each) so the budget
    # covers one unit rather than a whole sweep.
    timeout_s = 28800


class CPAC1(SubprocessAdapter):
    """CPA for C1 cytokine-response: classic latent δ-arithmetic (seen cytokine, held cell type)."""

    name, family, gpu = "CPA", "latent", True
    conda_env, runner, timeout_s = "ivc-cpa", "cpa_c1_runner.py", 7200


class CPA(SubprocessAdapter):
    name, family, gpu = "CPA", "latent", True
    # dedicated env (cpa-tools pins an old torch/scvi stack); gene-side repr built inside the runner.
    conda_env, runner, requires_gene_side = "ivc-cpa", "cpa_runner.py", False
    timeout_s = 7200  # Chen (60k-capped) needs >3600s for 40-60 epochs


class CPAchem(SubprocessAdapter):
    """Historical non-native fingerprint-to-latent CPA adaptation, excluded from the census.

    This class is NOT native chemCPA. Native chemCPA provenance is the separate
    scripts/chemcpa_native_op3.py and scripts/chemcpa_evaluate.py workflow.
    Kept only to identify old artifacts; it must not be relabelled as native.
    """

    name, family, gpu = "CPA", "latent", True
    conda_env, runner, requires_compound_side = "ivc-cpa", "cpa_c5_runner.py", True
    timeout_s = 7200


class CINEMAOT(SubprocessAdapter):
    name, family, gpu = "CINEMA-OT", "ot", True
    # not_defined† on C3_LO_gene → runs as a perturbation-agnostic OT FLOOR (excluded from ranking).
    # The floor doesn't need a gene-side repr (same global OT shift for every held gene).
    conda_env, runner, requires_gene_side = (
        "scperturbench_eval",
        "cinemaot_runner.py",
        False,
    )


class CellOT(SubprocessAdapter):
    """Published seen-response transport on held groups, using the payload runner.

    The helper module cellot_runner.py is not a CLI payload runner and cannot
    produce out.npz. Unseen intervention labels have no trained target map.
    """

    name, family, gpu = "CellOT", "ot", True
    conda_env, runner, requires_gene_side = "cellot", "cellot_c1_runner.py", False
    pred_key_is_group = True
    timeout_s = 7200

    def fit(self, cs, split, side_info=None):
        train_labels = set(cs.obs.iloc[split.train_idx]["perturbation"].astype(str))
        test_labels = set(cs.obs.iloc[split.test_idx]["perturbation"].astype(str))
        unseen = test_labels - train_labels
        if unseen:
            raise NotImplementedError(
                "CellOT has no trained target distribution for unseen intervention"
                " labels. A pooled map is not a conditioned unseen-intervention"
                f" predictor. Missing labels: {sorted(unseen)[:5]}"
            )
        return super().fit(cs, split, side_info)


class CellOTC5(CellOT):
    """CellOT on the held-lineage COMPOUND split (T5c) — ONE TRANSPORT PER COMPOUND.

    `CellOTC1` pools every treated cell into one target cloud, which is correct where the treatment
    is a single seen stimulus but collapses the ~141 OP3 compounds into one profile that carries no
    compound-specific information. CellOT's published operation is a map between a control cloud and
    ONE treated cloud, so the native use on this split is one map per compound; the unconditional
    autoencoder is shared, only the f/g potentials are fitted per compound.

    pred_key_is_group is FALSE here: the runner returns real per-perturbation keys, so the adapter
    must match them to the test labels rather than average them into one profile.
    """

    name, family, gpu = "CellOT", "ot", True
    conda_env, runner, requires_gene_side = "cellot", "cellot_c5_runner.py", False
    pred_key_is_group = False
    timeout_s = 86400  # one transport per compound; see the runner's budget note


class CellOTC1(CellOT):
    """CellOT on a held-GROUP split (cell-context / donor). The transport map is learned between the
    control and perturbed clouds of the SEEN condition on the training groups and applied to the held
    group's own control cells. This alias preserves the historical held-group entry point; it shares
    the explicit rejection of unseen intervention labels with CellOT."""

    name, family, gpu = "CellOT", "ot", True
    conda_env, runner, requires_gene_side = "cellot", "cellot_c1_runner.py", False
    pred_key_is_group = True
    timeout_s = 7200


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
    C5_unseen_cpd). name='STATE' → registry status; distinct C5 runner; from-scratch ST lower bound.
    """

    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner, requires_compound_side = "ivc-state", "state_c5_runner.py", True
    timeout_s = 7200


class STATE(SubprocessAdapter):
    name, family, gpu = "STATE", "hybrid", True
    conda_env, runner = (
        "ivc-state",
        "state_runner.py",
    )  # arc-state; ST predicts held genes (fewshot)
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
    *published-anchor* reproduction is gated separately (scripts/pertadapt_validate.py).
    """

    name, family, gpu = "PertAdapt", "hybrid", True
    conda_env, runner = "scfoundation", "pertadapt_runner.py"
    timeout_s = 7200  # per-cell frozen embedding fwd is the cost driver


# applicable-on-C3 (native unseen-gene) first; adapted/OT need a gene-side repr (requires_gene_side)
class PerturbNetC5(SubprocessAdapter):
    """PerturbNet (Yu, Qian, Song & Welch, Mol Syst Biol 2025, s44320-025-00131-3) on the C5/OP3
    compound cluster. A conditional invertible neural network maps a frozen pretrained ChemicalVAE
    representation of the compound's SMILES (196-d, standardised by the released ZINC mu/std) to a
    per-fold VAE cell-state latent, exactly the authors' LINCS-Drug "unadjusted" configuration.

    T5u (C5_global_compound_holdout) is NATIVE -- the paper's own unseen-compound experiment: the
    held SMILES is an unseen point of the chemical latent space and the flow samples its cell-state
    distribution directly. T5c (C5_loct_<lineage>) is ADAPTED: PerturbNet publishes no held-cell-type
    protocol (its sci-Plex covariate variant conditions on cell line, undefined for a lineage absent
    from training), so the runner uses the model's own counterfactual operator generate_zprime to
    carry the held lineage's own DMSO cells from the control condition to each compound.

    Emits ONE PROFILE PER COMPOUND on both splits, so pred_key_is_group stays False. The runner
    consumes SMILES (ChemicalVAE's input), taking them from the payload when present and otherwise
    reading the OP3 obs off disk; the one OP3 compound whose SMILES exceeds ChemicalVAE's fixed
    120-character input (Navitoclax, 122) is declined and keeps the control mean.
    Official code github.com/welch-lab/PerturbNet; pretrained ChemicalVAE from the authors'
    HuggingFace record cyclopeta/PerturbNet_reproduce (no weights redistributed)."""

    name, family, gpu = "PerturbNet", "generative", True
    conda_env, runner = "ivc-perturbnet", "perturbnet_c5_runner.py"
    timeout_s = 10800  # measured ~37 min/unit at the published 81 VAE + 50 cINN epochs


HEAVY_BASELINES = [
    GEARS,
    AttentionPert,
    ScGPT,
    ScFoundation,
    STATE,
    PertAdapt,
    ScGen,
    CPA,
    CINEMAOT,
    CellOT,
    ScPRAM,
]


class CellFlowC5(SubprocessAdapter):
    """CellFlow (Klein et al., bioRxiv 2025.04.11.648220; theislab) on the OP3 compound cluster —
    NATIVE. CellFlow learns a conditional flow from the control cell distribution to the perturbed
    one, with the perturbation supplied as a condition representation; the published evaluation
    includes unseen drugs, so a held compound is an unseen value of an input the model already
    consumes. One runner serves both C5 splits: T5c (held lineage, compounds seen) and T5u (held
    compounds). The harness supplies its own Morgan fingerprints (radius 2, 1024 bits) rather than
    CellFlow's get_molecular_fingerprints (radius 4), which keeps the compound representation
    identical to every other C5 entrant; that substitution is disclosed. Emits one profile per
    compound, so pred_key_is_group stays False."""

    name, family, gpu = "CellFlow", "flow", True
    conda_env, runner = "ivc-cellflow", "cellflow_c5_runner.py"
    requires_compound_side = True
    timeout_s = 7200


class PRnetC5(SubprocessAdapter):
    """PRnet (Qi et al., Nat Commun 2024, 15:9256; Apache-2.0) on the OP3 compound cluster — NATIVE.
    The Perturb-adaptor embeds the compound's rFCFP4 fingerprint so the model generalises to
    compounds absent from training, and the released repository exposes an unseen-compound partition
    as a first-class split flag, so both C5 splits run through the published interface. OP3 carries a
    single dose, so the dose term is constant. The gene panel is the harness's uniform 2000 HVGs
    rather than the authors' 5000, which is a hyper-parameter of the released training script; that
    substitution is disclosed. Emits one profile per compound."""

    name, family, gpu = "PRnet", "generative", True
    conda_env, runner = "ivc-prnet", "prnet_c5_runner.py"
    requires_compound_side = True
    timeout_s = 7200


class PerturbNetC3(SubprocessAdapter):
    """PerturbNet on the GENETIC axis — T3 (primary-T CRISPR, C3_true_lo_gene) and T4 (Frangieh KO,
    C4_modality_lo_ko). NATIVE: this is the model's own published genetic protocol. The held gene is
    represented by the frozen pretrained GenotypeVAE applied to its binary GO-annotation vector
    (15,988 terms), so a gene that never appears in training still has a representation, and the cINN
    samples its cell-state distribution — the authors' unseen-perturbation recipe
    (notebooks/Tutorial_PerturbNet_Genetic.ipynb cells 16-19 and 28-31).

    Because PerturbNet carries its OWN gene-side representation, `requires_gene_side` stays False:
    the harness's side_info['gene_embedding'] is neither consumed nor needed.

    T3/T4 hold out the PERTURBATION, so the runner emits ONE PROFILE PER HELD GENE and
    `pred_key_is_group` stays False. A held gene with no row in the released GO table is declined by
    name and keeps the control mean; the runner refuses to write a prediction that is constant across
    the held genes.

    Released artefacts (GenotypeVAE weights + gene x GO annotation matrix) come from the authors'
    HuggingFace record cyclopeta/PerturbNet_reproduce, which the official README names as the source
    of "the required data, toy examples, and model weights"; nothing is redistributed here.
    """

    name, family, gpu = "PerturbNet", "generative", True
    conda_env, runner = "ivc-perturbnet", "perturbnet_c3_runner.py"
    timeout_s = 10800


class CellFlowC1(SubprocessAdapter):
    """CellFlow on the held-GROUP tasks — T1 (Kang IFN-β, held cell type) and T2 (Soskic CD4
    activation, held donor). This is CellFlow's own published PBMC experiment: one stimulus, seen in
    training, and a held unit whose stimulated cells are hidden. The stimulus is registered as a
    single CATEGORICAL perturbation covariate (CellFlow fits its own OneHotEncoder — no external
    representation is needed for a seen categorical), and the held unit is registered as a SPLIT
    covariate, the released API's construct for "this unit's own control cells are the source of the
    flow". A split covariate is never embedded into the condition, which is precisely why the held
    unit may be a value absent from training; a sample_covariate would be embedded and would raise on
    the unseen value. Same env and model setup as the compound runner (model_runners/
    cellflow_common.py), only the condition and the held axis differ.

    The perturbation axis has ONE seen value, so the runner emits ONE profile for the held group ->
    pred_key_is_group = True. Without that flag heavy.py would match the single key against the test
    labels; here they happen to coincide, but the flag states the regime rather than relying on it.
    """

    name, family, gpu = "CellFlow", "flow", True
    conda_env, runner = "ivc-cellflow", "cellflow_c1_runner.py"
    pred_key_is_group = True
    timeout_s = 14400


class CellFlowGene(SubprocessAdapter):
    """CellFlow on the held-GENE tasks — T3 (primary-T CRISPR, held genes) and T4 (Frangieh CRISPR,
    held knockout; RNA and protein readouts). The held entity is the perturbation, so the categorical
    entry point is undefined and the perturbation is supplied as a REPRESENTATION through CellFlow's
    own `perturbation_covariate_reps` — the same mechanism the compound runner uses for an unseen
    drug, with a gene-side vector in place of a fingerprint. The runner consumes
    side_info['gene_embedding'] when a split provides it and otherwise builds the harness's standard
    leak-safe control-only PCA gene-loading embedding in-runner, exactly as scgen_runner.py and
    run_c4_conditioned.LinearShiftKOEmb do on these two tasks; requires_gene_side therefore stays
    False, matching ScGen and CPA, which carry the same in-runner construction.

    ONE PROFILE PER HELD GENE, so pred_key_is_group stays False. A perturbed gene that is not a
    feature of the expression panel has no loading vector and is declined; heavy.py keeps the control
    mean for that label, and the runner prints the coverage and the across-gene spread.
    """

    name, family, gpu = "CellFlow", "flow", True
    conda_env, runner = "ivc-cellflow", "cellflow_gene_runner.py"
    requires_gene_side = False
    timeout_s = 14400


class MAPC5(SubprocessAdapter):
    """MAP (Feng et al., Nat Mach Intell 2026, s42256-026-01286-w; MIT) on the OP3 compound
    cluster.  A frozen SE-600M state encoder embeds a bulk of control cells, a FROZEN knowledge
    encoder pre-trained on MAP-KG embeds the compound's SMILES into a mechanism-aware space, and a
    llama-backbone perturbation transformer + gene decoder emit the perturbed profile.

    MAP consumes SMILES, not Morgan bits, so this adapter injects the OP3 loader's own compound
    SMILES map (`cs.uns['smiles']`, written at data/loaders/op3.py:117) into the payload as
    `smiles_keys`/`smiles_vals` and leaves `fingerprint_*` unused; `requires_compound_side` stays
    False because the fingerprint is not this model's compound representation.

    Emits ONE PROFILE PER COMPOUND on both splits -- the drug token is per-compound and the gene
    decoder is evaluated once per compound -- so `pred_key_is_group` stays False, as for BiolordC5,
    CellFlowC5, PRnetC5 and PerturbNetC5.  (T5c holds a GROUP, but the harness's test labels there
    are still the held lineage's compound labels and MAP predicts each of them, so the per-label
    keying is the correct regime; the group-keyed flag belongs to runners that collapse the
    perturbation axis into a single profile.)

    Status by the interface rule: T5u is the published unprofiled-drug regime (a global fraction of
    drugs withheld, all their profiles removed from training) -- ours withholds 20% where MAP
    withholds 5%.  T5c has no published analogue: MAP's OP3 cell-context experiment holds out
    drug x cell-type PAIRS ("5% of drugs per cell type ... unseen cell type-drug combinations"),
    never a whole lineage.

    The runner carries a LEAK GATE.  MAP's protocol excludes held drugs and their aliases from
    MAP-KG before knowledge pre-training; the released encoder excluded the authors' holdout, not
    ours, and it is frozen inside the model.  The runner checks the held compounds against the
    released MAP-KG drug node table and refuses to emit a prediction if any of them is an entity
    the encoder was pre-trained on.  See model_runners/map_c5_runner.py."""

    name, family, gpu = "MAP", "knowledge", True
    conda_env, runner = "ivc-map", "map_c5_runner.py"
    timeout_s = 10800

    def _build_payload(self, cs, split, side_info):
        p = super()._build_payload(cs, split, side_info)
        smi = (getattr(cs, "uns", None) or {}).get("smiles") or {}
        if smi:
            p["smiles_keys"] = np.asarray([str(k) for k in smi])
            p["smiles_vals"] = np.asarray([str(v) for v in smi.values()])
        return p


class BiolordC3(SubprocessAdapter):
    """Biolord on the UNSEEN-PERTURBATION GENE splits (T3 primary-T CRISPR, T4 Frangieh unseen KO)
    — NATIVE, the same released entry point as the compound runner with a gene-side vector in place
    of the RDKit one. biolord's abstract claims "unseen drugs and genetic perturbations", and the
    genetic half is a released experiment, not a claim: biolord_reproducibility
    scripts/biolord/adamson/base_experiment_adamson.py scores the OOD subgroup `["unseen_single"]`
    (single-gene perturbations absent from training) with

        ordered_attributes_keys=[varying_arg["ordered_attributes_key"]], categorical_attributes_keys=None
        adamson_config_optimal.py:  "ordered_attributes_key": "perturbation_neighbors"
        dataset_pred[ordered_attributes_key] = repeat_n(dataset_reference[...][idx_ref, :], n_obs)
        test_preds, _ = model.module.get_expression(dataset_pred)

    i.e. control input with the held gene's ordered-attribute row swapped in — structurally identical
    to the sci-Plex compound recipe BiolordC5 runs. `perturbation_neighbors` is prior knowledge (the
    GO-Jaccard similarity of a gene to its top-20 GO neighbours; the released preprocessing notebook
    builds it from GEARS' go.csv), so an unseen gene is an unseen VALUE of an attribute the model
    already trains on and no external regression sits on top of biolord.

    requires_gene_side stays False: like ScGen/CPA on this axis, the gene-side vector is built INSIDE
    the runner (from the cached GEARS gene2go, falling back to the benchmark's control-only PCA gene
    loadings), because no C3/C4 loader supplies side_info['gene_embedding']; the runner still prefers
    a harness-supplied embedding when one is present. T3/T4 hold out the PERTURBATION and the runner
    emits ONE PROFILE PER HELD GENE, so pred_key_is_group stays False."""

    name, family, gpu = "Biolord", "latent", True
    conda_env, runner = "ctx-biolord", "biolord_c3_runner.py"
    requires_gene_side = False
    timeout_s = 7200


class ScGPTC5Cond(SubprocessAdapter):
    """scGPT on the OP3 COMPOUND cluster with a COMPOUND-CONDITIONED head — ADAPTED, added because
    reviewer 2 comment 1 asks for the foundation models on the tasks they are input-valid for.

    Neither existing scGPT entry point can represent a compound. `ScGPTC1` collapses all 141 OP3
    compounds into one global "exposed" flag (one profile for the held group). `fp_wrapper`'s
    `FPUnseenCompound` fits Ridge(fingerprint -> the model's OWN predicted training-compound
    responses), so the encoder never sees the held compound and the cell is bounded above by
    FP-ridge; that composition is circular and is not an evaluation of scGPT.

    This adapter gives scGPT the shape chemCPA already has -- a frozen molecular vector feeding a
    trainable map whose output is decoded through the model's own cell representation, trained
    against the OBSERVED response:

        prediction = control_mean + mean_i head([ E_frozen(control cell i) || Morgan(compound) ])

    E_frozen is scGPT_human's released encoder held FIXED (the library's own cell-embedding path:
    <cls> token, 51-bin value binning, L2 normalisation); the MLP head is the only trained part and
    is fit on the TRAIN fold against observed (lineage x compound) mean responses. Freezing the
    encoder is deliberate and disclosed: it makes scGPT and scFoundation directly comparable on this
    split, and scFoundation's cell-context runner is already exactly this shape. Not bounded by
    FP-ridge, because the head also sees the cell representation.

    One runner serves both C5 splits: T5u (held COMPOUNDS, inference input = the control pool) and
    T5c (held LINEAGE, inference input = that lineage's own DMSO cells). Both emit ONE PROFILE PER
    COMPOUND keyed by the compound label, so `pred_key_is_group` stays False. The runner detects the
    regime from the payload and enforces the matching leak assertion before any head gradient step.

    The line against the graph-conditioned methods is mechanical: GEARS and AttentionPert condition
    through a gene graph with no slot for a molecular vector, whereas scGPT emits a reusable cell
    representation that a molecular vector can be concatenated to."""

    name, family, gpu = "scGPT", "foundation", True
    conda_env, runner = "scgpt", "scgpt_c5_cond_runner.py"
    requires_compound_side = True
    timeout_s = 14400


class ScFoundationC5Cond(SubprocessAdapter):
    """scFoundation on the OP3 COMPOUND cluster through a COMPOUND-CONDITIONED head — ADAPTED.

    Replaces the circular `FPUnseenCompound` wiring on T5u. That wrapper regressed a fingerprint onto
    the model's OWN predicted training-compound responses, so the frozen encoder never saw the held
    compound and the cell was bounded above by FP-ridge; it is not an evaluation of the model. Here the
    compound representation passes THROUGH the model's own decoding path instead, which is the shape
    that makes chemCPA non-circular:

        prediction = mean_ctrl(held context) + mean_i head([ E_frozen(control cell i) ‖ Morgan fp ])

    E_frozen is scFoundation's released `cell` encoder, held fixed (identical embedding path to
    scfoundation_c1_runner.py / scfoundation_runner.py); `head` is the only trainable part and is
    trained on the TRAIN FOLD ONLY against OBSERVED perturbed profiles; at inference the held
    compound's fingerprint goes through the SAME head. This is strictly more information than
    Ridge(fingerprint -> observed response), so the cell is not bounded by the FP-ridge baseline.

    The same class serves both compound splits, because the runner detects the regime from the payload:
      * T5c (held lineage, compounds seen) - one profile per seen compound, from the held lineage's
        own control cells;
      * T5u (held compounds, lineages seen) - one profile per held compound, from the control pool.
    Both hold out a PERTURBATION-keyed set of strata, so the runner emits ONE PROFILE PER COMPOUND and
    `pred_key_is_group` stays False (the defect that silently turned the T5c cells into ctrl-pred).

    requires_compound_side because the Morgan fingerprint IS the head's compound input. The line
    against the graph models is mechanical: GEARS and AttentionPert condition on a gene graph with no
    slot for a molecular vector, whereas scGPT and scFoundation emit a reusable cell representation
    that a molecular vector can be concatenated to. ADAPTED, never native: the head is written here,
    not published by the authors, and it is admitted only because R2-1 asked for these tasks.
    """

    name, family, gpu = "scFoundation", "foundation", True
    conda_env, runner = "scfoundation", "scfoundation_c5cond_runner.py"
    requires_compound_side = True
    pred_key_is_group = False
    timeout_s = 14400
