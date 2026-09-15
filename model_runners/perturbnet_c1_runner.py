#!/usr/bin/env python
"""Categorical PerturbNet context transfer; run with the ivc-perturbnet Python.

Usage: python perturbnet_c1_runner.py in.npz out.npz

The released Labelator encodes each *seen* stimulus separately. The released
VAE and Net2NetFlow_TFVAEFlow.train_cinn learn only from X_train. Inference is
X_ctrl_inf -> VAE.encode -> generate_zprime(z, control, target) -> VAE.decoder.
Thus the held context enters the basal latent, without an untrained group ID.
The parameter-free Labelator wrapper only supplies the four-result interface
expected by train_cinn's ChemicalVAE-era conditioning hook. No learned head,
latent shift, per-target pooled surrogate, or upstream patch is introduced.

This composes released categorical/cINN operations; it does not claim that the
authors published a T1/T2-specific checkpoint or evaluated these exact splits.
Public implementation: https://github.com/welch-lab/PerturbNet (0.0.3b1).

All requested non-control labels must occur in training; unseen categories are
explicitly declined, never mapped to another category or a control profile.
Gene columns are preserved exactly: this categorical route has no gene-symbol
vocabulary lookup, so there is no gene alias eligibility filter.

Knobs: IVCBENCH_PERTURBNET_{VAE_EPOCHS=81,CINN_EPOCHS=50,MAXCELLS=60000,
NGEN=500,BATCH=128,VAE_LR=1e-4,CINN_LR=4.5e-6,SEED=42,THREADS=8}.
MAXCELLS and NGEN cap actual cells, never tile mean profiles. No checkpoint or
prediction cache is written or reused. Output contains one row per condition.
Optional train_indices/inference_input_indices or train_cell_ids/ctrl_inf_cell_ids
(legacy ctrl_cell_ids is also accepted)
allow an exact row-identity disjointness assertion. The required payload lacks
these IDs; in that case an exact expression-row overlap assertion is used and
the unavailable global-index proof is explicitly reported.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np


CONTROL = "control"


def _prefer_bundled_cuda_libs() -> None:
    """Resolve this host's older global CUDA libraries before importing torch."""
    import glob
    import sysconfig

    if os.environ.get("IVCBENCH_PERTURBNET_REEXEC") == "1":
        return
    libs = sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib")))
    if libs:
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libs) + (os.pathsep + old if old else "")
        os.environ["IVCBENCH_PERTURBNET_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get("IVCBENCH_PERTURBNET_" + name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _assert_disjoint(data: dict, x: np.ndarray, controls: np.ndarray) -> None:
    checked_ids = False
    pairs = [("train_indices", "inference_input_indices")]
    # The two inference-name aliases share train_cell_ids; absence of the
    # unused alias must not invalidate a complete pair under the other name.
    if "ctrl_inf_cell_ids" in data:
        pairs.append(("train_cell_ids", "ctrl_inf_cell_ids"))
    if "ctrl_cell_ids" in data:
        pairs.append(("train_cell_ids", "ctrl_cell_ids"))
    if "train_cell_ids" in data and not any(k in data for k in ("ctrl_inf_cell_ids", "ctrl_cell_ids")):
        raise ValueError("train_cell_ids requires ctrl_inf_cell_ids (or legacy ctrl_cell_ids)")
    for train_key, inf_key in pairs:
        if (train_key in data) != (inf_key in data):
            raise ValueError(f"Provide both {train_key} and {inf_key} for the leak check")
        if train_key in data:
            a, b = np.asarray(data[train_key]).reshape(-1), np.asarray(data[inf_key]).reshape(-1)
            assert len(a) == len(x) and len(b) == len(controls), "Cell ID array lengths disagree"
            overlap = set(a.astype(str)) & set(b.astype(str))
            assert not overlap, f"Training/inference cell identities overlap: {sorted(overlap)[:8]}"
            print(f"[leak] {train_key}/{inf_key} intersection=0", flush=True)
            checked_ids = True
    # Checking the entire supplied training array also covers every later subset.
    def digest(row):
        return hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
    train_rows = {digest(row) for row in x}
    overlap_rows = [i for i, row in enumerate(controls) if digest(row) in train_rows]
    assert not overlap_rows, f"Exact training/control expression rows overlap: {overlap_rows[:8]}"
    print(f"[leak] exact_expression_row_intersection=0 train={len(x)} inference={len(controls)}", flush=True)
    if not checked_ids:
        print("[leak] global_cell_index_disjointness=UNVERIFIED (IDs absent from payload); "
              "exact expression-row overlap assertion passed", flush=True)
    if "group_train" in data:
        groups = np.asarray(data["group_train"]).astype(str)
        assert groups.shape == (len(x),), "group_train length disagrees"
        # The held axis is the one whose INFERENCE values do not occur in training -- not the one
        # that happens to equal group_train. heavy.py fills group_train from cell_type_coarse
        # whenever that column exists, so on the donor-held T2 split group_train == celltype_train
        # is true and the old test asserted that the held DONOR's cell types were absent from
        # training. They never are (every donor has CD4_Naive and CD4_Memory), so every T2 unit
        # aborted before a cell was trained.
        for train_key, inf_key in (("gem_train", "gem_inf"), ("celltype_train", "celltype_inf")):
            if train_key not in data or inf_key not in data:
                continue
            _tr = set(np.asarray(data[train_key]).astype(str))
            _inf = set(np.asarray(data[inf_key]).astype(str))
            if _inf and not (_inf & _tr):
                held_groups = np.asarray(data[inf_key]).astype(str)
                assert held_groups.shape == (len(controls),), f"{inf_key} length disagrees"
                overlap = _inf & _tr
                assert not overlap, f"Held {inf_key} groups entered training: {sorted(overlap)}"
                print(f"[leak] held_group_axis={train_key} train/held_intersection=0 "
                      f"held={sorted(set(held_groups))}", flush=True)
                break
        else:
            print("[leak] held_group_axis=UNVERIFIED (group_train does not identify a supplied context axis)", flush=True)


def _report_and_write(out_path, requested, labels, profiles, control_mean, reasons):
    missing = sorted(set(requested) - set(labels))
    extra = sorted(set(labels) - set(requested))
    assert not extra and set(missing) == set(reasons), "Coverage has an unexplained missing/extra label"
    p = np.asarray(profiles, dtype=np.float32).reshape(len(labels), len(control_mean))
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite model prediction; refusing to write")
    print(f"[coverage] requested={requested} predicted={labels} missing={missing} extra={extra}", flush=True)
    for label in missing:
        print(f"[coverage] decline_reason {label}: {reasons[label]}", flush=True)
    identical = [(labels[i], labels[j]) for i in range(len(labels))
                 for j in range(i + 1, len(labels)) if np.array_equal(p[i], p[j])]
    print(f"[condition] bit_exact_identical_pairs={identical}; "
          f"all_rows_identical={len(labels) > 1 and len(identical) == len(labels)*(len(labels)-1)//2}", flush=True)
    for label, row in zip(labels, p):
        difference = float(np.max(np.abs(row.astype(np.float64) - control_mean)))
        print(f"[control] {label}: max_abs_pred_minus_training_control_mean={difference:.12g} "
              f"within_1e-5={difference <= 1e-5}", flush=True)
    np.savez(out_path, pred_perts=np.asarray(labels, dtype=object), pred_means=p)
    print(f"[output] {out_path}: pred_means={p.shape} gene_order=input", flush=True)


class _IdentityStandardizer:
    def standardize_z_torch(self, x):
        return x


def main(in_path: str, out_path: str) -> None:
    _prefer_bundled_cuda_libs()
    import torch
    import perturbnet.cinn.flow as flow_module
    from perturbnet.cinn.flow import ConditionalFlatCouplingFlow, Net2NetFlow_TFVAEFlow
    from perturbnet.data_vae.vae import VAE
    from perturbnet.net2net.modules.labels.model import Labelator

    seed = int(os.environ.get("IVCBENCH_PERTURBNET_SEED", os.environ.get("IVCBENCH_SEED", "42")))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(_positive_int("THREADS", 8))
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(in_path, allow_pickle=True) as payload:
        data = {k: payload[k] for k in payload.files}
    x = np.asarray(data["X_train"], dtype=np.float32)
    ctrl = np.asarray(data["X_ctrl_inf"], dtype=np.float32)
    mask = np.asarray(data["is_control_train"], dtype=bool)
    train_labels = np.asarray(data["pert_train"]).astype(str)
    genes = np.asarray(data["genes"]).astype(str)
    requested = sorted(set(np.asarray(data["test_perts"]).astype(str)) - {CONTROL})
    if x.ndim != 2 or ctrl.ndim != 2 or x.shape[1] != ctrl.shape[1] or x.shape[1] != len(genes):
        raise ValueError("X_train/X_ctrl_inf/genes shapes disagree")
    if mask.shape != (len(x),) or train_labels.shape != (len(x),):
        raise ValueError("Training label/control mask lengths disagree")
    if not np.isfinite(x).all() or not np.isfinite(ctrl).all():
        raise ValueError("Non-finite input expression")
    if not mask.any() or not len(ctrl):
        raise ValueError("Training controls and held-unit own controls are both required")
    if np.any((train_labels == CONTROL) != mask):
        raise ValueError("pert_train control labels disagree with is_control_train")
    if not requested:
        raise ValueError("No requested non-control condition")
    _assert_disjoint(data, x, ctrl)
    control_mean = x[mask].mean(axis=0, dtype=np.float64)
    reasons = {p: "unseen categorical stimulus; no trained conditioning category" for p in requested
               if p not in set(train_labels)}
    for p, reason in reasons.items():
        print(f"[decline] {p}: {reason}", file=sys.stderr, flush=True)
    targets = [p for p in requested if p not in reasons]
    if not targets:
        _report_and_write(out_path, requested, [], [], control_mean, reasons)
        return

    # Select actual training rows while retaining every training condition.
    maximum = _positive_int("MAXCELLS", 60000)
    selected = np.arange(len(x))
    if len(x) > maximum:
        groups = [np.flatnonzero(train_labels == label) for label in sorted(set(train_labels))]
        if maximum < len(groups):
            raise ValueError("MAXCELLS is too small to retain every condition")
        mandatory = np.asarray([rng.choice(indices) for indices in groups])
        remaining = np.setdiff1d(selected, mandatory)
        selected = np.sort(np.concatenate((mandatory, rng.choice(remaining, maximum-len(mandatory), replace=False))))
    x_fit, labels_fit = x[selected], train_labels[selected]
    if len(x_fit) < 10:
        raise ValueError("At least 10 training cells are required for native train/validation and BatchNorm")
    # The released training API makes this exact 80/20 split internally. A
    # category present solely in validation has no trained cINN condition.
    n_validation = len(x_fit) - int(len(x_fit) * 0.8)
    native_train_rows = np.random.RandomState(seed).permutation(len(x_fit))[n_validation:]
    native_train_labels = set(labels_fit[native_train_rows])
    if CONTROL not in native_train_labels:
        raise ValueError("Native cINN training subset has no control condition")
    for target in targets:
        if target not in native_train_labels:
            reasons[target] = "categorical stimulus has no cells in the native cINN training subset"
            print(f"[decline] {target}: {reasons[target]}", file=sys.stderr, flush=True)
    targets = [target for target in targets if target not in reasons]
    if not targets:
        _report_and_write(out_path, requested, [], [], control_mean, reasons)
        return
    batch = min(_positive_int("BATCH", 128), int(len(x_fit)*0.8), len(x_fit)-int(len(x_fit)*0.8))
    while batch > 1 and any(n % batch == 1 for n in (len(x_fit), int(len(x_fit)*0.8), len(x_fit)-int(len(x_fit)*0.8))):
        batch -= 1
    if batch < 2:
        raise ValueError("No BatchNorm-safe batch size for this training fold")
    n_inf = min(_positive_int("NGEN", 500), len(ctrl))
    infer_rows = np.sort(rng.choice(len(ctrl), n_inf, replace=False))
    vae_epochs, cinn_epochs = _positive_int("VAE_EPOCHS", 81), _positive_int("CINN_EPOCHS", 50)
    run_digest = hashlib.sha256(Path(in_path).read_bytes()).hexdigest()
    print(f"[PerturbNet-C1] device={device} seed={seed} input_sha256={run_digest} "
          f"held={data.get('held_label', data.get('held_lineage', 'unspecified'))} "
          f"fit={x_fit.shape} own_control_source={ctrl.shape} selected_controls={n_inf} "
          f"vae_epochs={vae_epochs} cinn_epochs={cinn_epochs} batch={batch}", flush=True)
    print(f"[upstream] {flow_module.__file__}; Labelator.encode + VAE.train_np/encode/decoder + "
          "Net2NetFlow_TFVAEFlow.train_cinn/generate_zprime; checkpoint_cache=disabled", flush=True)
    print(f"[inference] X_ctrl_inf row indices={infer_rows.tolist()}", flush=True)
    print(f"[genes] count={len(genes)} alias_lookup=not_applicable (no gene vocabulary in categorical route)", flush=True)

    # Interface only: the actual category encoding is upstream Labelator.encode.
    class CategoricalConditioner(Labelator):
        def forward(self, label_index):
            one_hot = self.encode(label_index.reshape(-1).long()).float()
            return None, None, None, one_hot

    categories = sorted(set(labels_fit))
    label_index = {p: i for i, p in enumerate(categories)}
    conditioner = CategoricalConditioner(num_classes=len(categories)).to(device)
    print(f"[conditioning] distinct_one_hot_categories={label_index}; "
          f"conditioner_learned_parameters={sum(p.numel() for p in conditioner.parameters())}", flush=True)
    vae = VAE(num_cells_train=len(x_fit), x_dimension=x.shape[1],
              learning_rate=float(os.environ.get("IVCBENCH_PERTURBNET_VAE_LR", "1e-4")),
              BNTrainingMode=False, device=device)
    vae.train_np(train_data=x_fit, n_epochs=vae_epochs, batch_size=batch, verbose=True)
    vae.eval()
    flow = ConditionalFlatCouplingFlow(conditioning_dim=len(categories), embedding_dim=vae.z_dim,
        conditioning_depth=2, n_flows=20, in_channels=vae.z_dim, hidden_dim=1024,
        hidden_depth=2, activation="none", conditioner_use_bn=True)
    model = Net2NetFlow_TFVAEFlow(configured_flow=flow, first_stage_data=x_fit,
        cond_stage_data=labels_fit, perturbToOnehotLib=label_index,
        oneHotData=np.arange(len(categories), dtype=np.int64)[:, None],
        model_con=conditioner, std_model=_IdentityStandardizer(), model_cell=vae).to(device)
    model.train_cinn(n_epochs=cinn_epochs, batch_size=batch, seed=seed,
        lr=float(os.environ.get("IVCBENCH_PERTURBNET_CINN_LR", "4.5e-6")), auto_save=0)
    model.eval()
    if not np.isfinite(vae.train_loss).all() or not np.isfinite(model.train_loss).all():
        raise RuntimeError("Training produced a non-finite loss")

    # Encode real held controls once and use the same basal latent for each
    # condition, so distinct conditions cannot be simulated by resampling noise.
    latent = vae.encode(ctrl[infer_rows])
    profiles = []
    with torch.no_grad():
        basal = torch.as_tensor(latent, dtype=torch.float32, device=device)[:, :, None, None]
        c = conditioner.encode(torch.full((n_inf,), label_index[CONTROL], device=device, dtype=torch.long)).float()
        for target in targets:
            cprime = conditioner.encode(torch.full((n_inf,), label_index[target], device=device, dtype=torch.long)).float()
            translated = model.generate_zprime(basal, c, cprime).squeeze(-1).squeeze(-1)
            decoded, _ = vae.decoder(translated)
            cells = decoded.cpu().numpy()
            if cells.shape != (n_inf, len(genes)) or not np.isfinite(cells).all():
                raise RuntimeError(f"Invalid native output for {target}: {cells.shape}")
            profiles.append(cells.mean(axis=0, dtype=np.float64))
            print(f"[prediction] {target}: native_translated_cells={n_inf} "
                  f"mean_gene_cell_variance={np.var(cells.astype(np.float64), axis=0).mean():.12g}", flush=True)
    _report_and_write(out_path, requested, targets, profiles, control_mean, reasons)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: perturbnet_c1_runner.py in.npz out.npz")
    main(sys.argv[1], sys.argv[2])
