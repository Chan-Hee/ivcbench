#!/usr/bin/env python
"""PerturbNet runner for the COMPOUND cluster (C5 / OP3) — `ivc-perturbnet` conda env.

Invoked by ivcbench.baselines.heavy.PerturbNetC5:
    <ivc-perturbnet python> perturbnet_c5_runner.py <in.npz> <out.npz>

The published model
-------------------
PerturbNet (Yu, Qian, Song & Welch, Mol Syst Biol 2025, s44320-025-00131-3;
github.com/welch-lab/PerturbNet) links a PERTURBATION representation to a CELL-STATE representation
with a conditional invertible neural network (cINN):

  chemistry :  SMILES --one-hot(120x35)--> frozen pretrained ChemicalVAE --> z_pert (196-d)
               --> standardised by the released ZINC mu/std  (`StandardizeLoad`)
  cell      :  log-normalised expression --> VAE --> z_cell (10-d)      (`perturbnet.data_vae.vae.VAE`)
  mapping   :  ConditionalFlatCouplingFlow(conditioning_dim=196, embedding_dim=10, in_channels=10,
               n_flows=20, hidden_dim=1024, hidden_depth=2, conditioning_depth=2,
               activation="none", conditioner_use_bn=True)   trained by `Net2NetFlow_TFVAEFlow.train_cinn`

Those are verbatim the settings of the authors' own LINCS-Drug ("unadjusted") tutorial,
notebooks/Tutorial_PerturbNet_Chemicals.ipynb cells 26-34 and 51-54.  Only the pretrained
ChemicalVAE is reused (16 MB, `pretrained_model/chemicalVAE/model_params_525.pt` on the authors'
HuggingFace record cyclopeta/PerturbNet_reproduce); the cell VAE and the cINN are refit per fold on
the training fold only, so no held-out cell ever enters fitting.

Task coverage (see the module note in heavy.py)
----------------------------------------------
T5u  C5_global_compound_holdout — NATIVE.  This is the paper's own experiment ("We randomly selected
     30 drug treatments as unseen perturbations ..."): the held compound's SMILES is an unseen point
     of a continuous chemical latent space, so the cINN samples its cell-state distribution with no
     external regression.  Recipe = tutorial cells 53-54: tile the held compound's one-hot, encode
     with ChemicalVAE, standardise, `TFVAEZ_CheckNet2Net.sample_data`.

T5c  C5_loct_<lineage> — ADAPTED, not native.  PerturbNet has no published held-cell-type protocol
     (its sci-Plex "adjusted" model puts cell line in the conditioning vector, which is undefined for
     a lineage that never appears in training, so that variant cannot be used here).  We instead use
     the model's OWN counterfactual-translation operator, `generate_zprime(x, c, c')`: the held
     lineage's own DMSO cells are encoded to z_cell, mapped through the flow under the CONTROL
     (DMSO) chemical embedding, and reversed under the target compound's embedding.  The lineage
     identity is therefore carried by the held cells' own latent state, exactly as in the
     delta-arithmetic entry points of scGen / biolord in this benchmark.  Registered as adapted*.

Leak-safety: only payload["X_train"] (the training fold) trains the VAE and the cINN.  X_ctrl_inf is
decoded/translated only.  Held compounds' treated cells are never in the payload at all.

Input contract beyond the standard payload
------------------------------------------
ChemicalVAE consumes SMILES, not Morgan bits, so this runner needs `smiles_keys` / `smiles_vals`
in the payload (the harness already carries them in `cs.uns['smiles']`; see heavy.py note).  If they
are absent it falls back to reading the compound->SMILES map straight off the OP3 h5ad obs, the same
way scripts/chemcpa_native_op3.py does.  Compounds whose SMILES exceeds ChemicalVAE's fixed 120-char
input, or use a character outside its 35-symbol alphabet, cannot be encoded by the RELEASED
checkpoint and are declined (the harness then keeps the control mean for them and says so).  On OP3
that is exactly one compound of 141: Navitoclax (122 characters, canonical form also 122).

Knobs: $IVCBENCH_PERTURBNET_{CHEMVAE_DIR,VAE_EPOCHS,CINN_EPOCHS,MAXCELLS,NGEN,BATCH,VAE_LR,CINN_LR}
"""
from __future__ import annotations

import os
import sys

import numpy as np


def _prefer_bundled_cuda_libs() -> None:
    """Put the env's own nvidia/*/lib (torch's bundled cuDNN/cuBLAS) FIRST on LD_LIBRARY_PATH.

    This host exports LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.6/lib64, whose
    cuDNN 8.4 shadows the cuDNN 8.5 that torch 2.0.0+cu117 was built against; the ChemicalVAE's GRU
    then dies in `flatten_parameters()` with "cuDNN version incompatibility". The dynamic loader
    reads LD_LIBRARY_PATH at process start, so the fix has to happen before torch is imported --
    hence the one-shot re-exec, guarded by an env flag so it can never loop.
    """
    import glob
    import sysconfig

    if os.environ.get("IVCBENCH_PERTURBNET_REEXEC") == "1":
        return
    libs = sorted(
        glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib"))
    )
    if not libs:
        return
    want = os.pathsep.join(libs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = want + (os.pathsep + cur if cur else "")
    os.environ["IVCBENCH_PERTURBNET_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_prefer_bundled_cuda_libs()

# ChemicalVAE's fixed input alphabet (Gomez-Bombarelli ZINC charset), verbatim from
# perturbnet.util.smiles_to_hot's default char_list. 35 symbols, max_len 120.
CHEMVAE_CHARS = [
    "7",
    "6",
    "o",
    "]",
    "3",
    "s",
    "(",
    "-",
    "S",
    "/",
    "B",
    "4",
    "[",
    ")",
    "#",
    "I",
    "l",
    "O",
    "H",
    "c",
    "1",
    "@",
    "=",
    "n",
    "P",
    "8",
    "C",
    "2",
    "F",
    "5",
    "r",
    "N",
    "+",
    "\\",
    " ",
]
MAX_LEN, N_CHAR, Z_PERT = 120, 35, 196
CONTROL_TOKEN = "control"
# OP3's control wells are DMSO; this is the literal obs['SMILES'] value for sm_name
# "Dimethyl Sulfoxide" in GSE279945_sc_counts_processed.h5ad (same constant as chemcpa_native_op3.py).
DMSO_SMILES = os.environ.get("IVCBENCH_PERTURBNET_CTRL_SMILES", "C[S+](C)[O-]")

_DEF_CHEMVAE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark",
    "vendor",
    "PerturbNet",
    "pretrained_model",
    "chemicalVAE",
)


def _encodable(smi: str) -> bool:
    return (
        isinstance(smi, str)
        and 0 < len(smi) <= MAX_LEN
        and not (set(smi) - set(CHEMVAE_CHARS))
    )


def _smiles_from_disk() -> dict:
    """compound -> SMILES read off the OP3 obs (used only if the payload carries no smiles_*)."""
    import anndata

    p = os.environ.get(
        "IVCBENCH_OP3_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "C5",
            "op3",
            "GSE279945_sc_counts_processed.h5ad",
        ),
    )
    obs = anndata.read_h5ad(p, backed="r").obs
    return (
        obs[["sm_name", "SMILES"]]
        .astype(str)
        .drop_duplicates("sm_name")
        .set_index("sm_name")["SMILES"]
        .to_dict()
    )


def main(in_path: str, out_path: str) -> None:
    import torch
    from perturbnet.util import smiles_to_hot, StandardizeLoad
    from perturbnet.chemicalvae.chemicalVAE import ChemicalVAE
    from perturbnet.data_vae.vae import VAE
    from perturbnet.cinn.flow import ConditionalFlatCouplingFlow, Net2NetFlow_TFVAEFlow
    from perturbnet.cinn.flow_generate import TFVAEZ_CheckNet2Net

    vae_epochs = int(
        os.environ.get("IVCBENCH_PERTURBNET_VAE_EPOCHS", "81")
    )  # tutorial value
    cinn_epochs = int(
        os.environ.get("IVCBENCH_PERTURBNET_CINN_EPOCHS", "50")
    )  # tutorial value
    max_cells = int(os.environ.get("IVCBENCH_PERTURBNET_MAXCELLS", "60000"))
    n_gen = int(os.environ.get("IVCBENCH_PERTURBNET_NGEN", "500"))
    batch = int(os.environ.get("IVCBENCH_PERTURBNET_BATCH", "128"))
    vae_lr = float(os.environ.get("IVCBENCH_PERTURBNET_VAE_LR", "1e-4"))
    cinn_lr = float(os.environ.get("IVCBENCH_PERTURBNET_CINN_LR", "4.5e-6"))
    chem_dir = os.environ.get("IVCBENCH_PERTURBNET_CHEMVAE_DIR", _DEF_CHEMVAE)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {CONTROL_TOKEN})
    if not test_perts:
        raise RuntimeError("PerturbNet-C5: payload carries no held perturbation label")

    if "smiles_keys" in d.files:
        smi_map = {str(k): str(v) for k, v in zip(d["smiles_keys"], d["smiles_vals"])}
    else:
        smi_map = _smiles_from_disk()
    smi_map[CONTROL_TOKEN] = DMSO_SMILES
    if not _encodable(DMSO_SMILES):
        raise RuntimeError(
            f"PerturbNet-C5: control SMILES {DMSO_SMILES!r} is not"
            " ChemicalVAE-encodable"
        )

    # ---- regime: are the held labels compounds the training fold never saw (T5u) or seen
    #      compounds on a held lineage (T5c)?  Decided from the payload alone.
    seen = {str(p) for p in pert_train[~is_ctrl]}
    unseen_cpd = len([p for p in test_perts if p in seen]) == 0

    # ---- cap training cells (stratified by label) --------------------------------------------
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        lab_all = np.where(is_ctrl, CONTROL_TOKEN, pert_train).astype(str)
        keep = []
        for lab, cnt in zip(*np.unique(lab_all, return_counts=True)):
            idx = np.where(lab_all == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]

    labels = np.where(is_ctrl, CONTROL_TOKEN, pert_train).astype(
        str
    )  # one label per training cell

    # ---- drop training cells whose compound the RELEASED ChemicalVAE cannot encode -------------
    bad_train = sorted({l for l in set(labels) if not _encodable(smi_map.get(l, ""))})
    if bad_train:
        keep = ~np.isin(labels, bad_train)
        X, labels = X[keep], labels[keep]
        print(
            f"[PerturbNet-C5] {len(bad_train)} training compound(s) not"
            f" ChemicalVAE-encodable (>{MAX_LEN} chars or out-of-alphabet), dropped:"
            f" {bad_train[:4]}",
            flush=True,
        )
    pred_targets = [p for p in test_perts if _encodable(smi_map.get(p, ""))]
    declined = [p for p in test_perts if p not in pred_targets]
    if declined:
        print(
            f"[PerturbNet-C5] declining {len(declined)} held compound(s) the released"
            f" ChemicalVAE cannot encode: {declined}",
            flush=True,
        )
    if not pred_targets:
        raise RuntimeError(
            "PerturbNet-C5: no held compound has a ChemicalVAE-encodable SMILES"
        )
    if X.shape[0] < 2 * batch:
        batch = max(8, X.shape[0] // 4)
    # Both the VAE and the cINN conditioner use BatchNorm1d, which raises on a final batch of one
    # cell. train_cinn also splits 80/20 internally, so guard all three loader lengths.
    _n = X.shape[0]
    _ntr = int(_n * 0.8)
    _nte = _n - _ntr
    while batch > 8 and any(m % batch == 1 for m in (_n, _ntr, _nte)):
        batch -= 1

    print(
        "[PerturbNet-C5]"
        f" regime={'T5u unseen-compound' if unseen_cpd else 'T5c held-lineage'} "
        f"train={X.shape} ctrl_inf={X_ctrl_inf.shape} targets={len(pred_targets)} dev={dev} "
        f"vae_ep={vae_epochs} cinn_ep={cinn_epochs}",
        flush=True,
    )

    # ---- 1. frozen pretrained ChemicalVAE + released ZINC standardiser ------------------------
    chem = ChemicalVAE(n_char=N_CHAR, max_len=MAX_LEN).to(dev)
    ck = os.path.join(chem_dir, "model_params_525.pt")
    chem.load_state_dict(torch.load(ck, map_location=dev))
    chem.eval()
    std_model = StandardizeLoad(
        np.load(os.path.join(chem_dir, "mu.npy")),
        np.load(os.path.join(chem_dir, "std.npy")),
        dev,
    )

    def embed(smiles_list, reps=1):
        """SMILES -> standardised 196-d ChemicalVAE representation, tiled `reps` times each."""
        sl = list(smiles_list)
        oh = smiles_to_hot(smiles=sl, max_len=MAX_LEN, padding="right", nchars=N_CHAR)
        # smiles_to_hot silently DROPS a SMILES longer than max_len (pad_smile returns None), which
        # would silently misalign compound -> row. Everything here is pre-filtered by _encodable, so
        # assert the alignment rather than trust it.
        assert oh.shape[0] == len(
            sl
        ), f"smiles_to_hot dropped {len(sl) - oh.shape[0]} SMILES"
        if reps > 1:
            oh = np.repeat(oh, reps, axis=0)
        out = []
        with torch.no_grad():
            for i in range(0, oh.shape[0], 512):
                _, _, _, z = chem(torch.tensor(oh[i : i + 512]).float().to(dev))
                out.append(z.cpu().numpy())
        return std_model.standardize_z(np.vstack(out)).astype(np.float32)

    # ---- 2. cell representation network: VAE on the training fold only ------------------------
    vae = VAE(
        num_cells_train=X.shape[0],
        x_dimension=X.shape[1],
        learning_rate=vae_lr,
        BNTrainingMode=False,
        device=dev,
    )
    vae.train_np(train_data=X, n_epochs=vae_epochs, batch_size=batch, verbose=False)
    vae.eval()

    # ---- 3. cINN: standardised chemical latent -> cell latent ---------------------------------
    uniq = sorted(set(labels))
    onehot_lib = smiles_to_hot(
        smiles=[smi_map[u] for u in uniq],
        max_len=MAX_LEN,
        padding="right",
        nchars=N_CHAR,
    )
    assert onehot_lib.shape[0] == len(uniq), "smiles_to_hot dropped a training compound"
    perturb_to_onehot = {u: i for i, u in enumerate(uniq)}
    flow = ConditionalFlatCouplingFlow(
        conditioning_dim=Z_PERT,
        embedding_dim=vae.z_dim,
        conditioning_depth=2,
        n_flows=20,
        in_channels=vae.z_dim,
        hidden_dim=1024,
        hidden_depth=2,
        activation="none",
        conditioner_use_bn=True,
    )
    model_c = Net2NetFlow_TFVAEFlow(
        configured_flow=flow,
        first_stage_data=X,
        cond_stage_data=np.asarray(labels),
        perturbToOnehotLib=perturb_to_onehot,
        oneHotData=onehot_lib,
        model_con=chem,
        std_model=std_model,
        model_cell=vae,
    )
    model_c.to(device=dev)
    model_c.train_cinn(n_epochs=cinn_epochs, batch_size=batch, lr=cinn_lr)
    model_c.eval()

    # ---- 4. predict one mean profile per held compound ----------------------------------------
    pnet = TFVAEZ_CheckNet2Net(model_c, dev, vae)
    pred_perts, pred_means = [], []

    if unseen_cpd:
        # published unseen-perturbation recipe (tutorial cells 53-54): sample the cell-state
        # distribution straight from the unseen compound's chemical representation.
        n = min(n_gen, max(64, X_ctrl_inf.shape[0]))
        for c in pred_targets:
            emb = embed([smi_map[c]], reps=n)
            _, gen = pnet.sample_data(emb)
            pred_perts.append(c)
            pred_means.append(np.asarray(gen, np.float32).mean(0))
    else:
        # held-lineage counterfactual translation with the model's own generate_zprime:
        # (held lineage's own DMSO cells, DMSO condition) -> same cells under the compound.
        src = X_ctrl_inf if X_ctrl_inf.shape[0] else X[labels == CONTROL_TOKEN]
        if src.shape[0] > n_gen:
            src = src[
                np.random.default_rng(0).choice(src.shape[0], n_gen, replace=False)
            ]
        z_cell = vae.encode(src)  # (n, z_dim)
        n = z_cell.shape[0]
        c_ctrl = torch.tensor(embed([DMSO_SMILES], reps=n)).float().to(dev)
        zt = torch.tensor(z_cell).float().to(dev).unsqueeze(-1).unsqueeze(-1)
        for c in pred_targets:
            c_new = torch.tensor(embed([smi_map[c]], reps=n)).float().to(dev)
            with torch.no_grad():
                zp = model_c.generate_zprime(zt, c_ctrl, c_new).squeeze(-1).squeeze(-1)
            gen = vae.decode(zp.cpu().numpy())
            pred_perts.append(c)
            pred_means.append(np.asarray(gen, np.float32).mean(0))

    P = np.vstack(pred_means).astype(np.float32)
    if P.shape[1] != len(genes):
        raise RuntimeError(
            f"PerturbNet-C5: emitted {P.shape[1]} genes, payload has {len(genes)}"
        )
    if not np.isfinite(P).all():
        raise RuntimeError("PerturbNet-C5: non-finite values in the predicted profiles")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[PerturbNet-C5] wrote {len(pred_perts)} compound profiles x"
        f" {P.shape[1]} genes ({len(declined)} declined) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
