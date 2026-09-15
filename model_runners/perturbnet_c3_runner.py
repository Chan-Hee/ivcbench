#!/usr/bin/env python
"""PerturbNet runner for the GENETIC clusters (T3 primary-T CRISPR / T4 Frangieh KO) — `ivc-perturbnet`.

Invoked by ivcbench.baselines.heavy.PerturbNetC3:
    <ivc-perturbnet python> perturbnet_c3_runner.py <in.npz> <out.npz>

The published model
-------------------
PerturbNet (Yu, Qian, Song & Welch, Mol Syst Biol 2025, s44320-025-00131-3;
github.com/welch-lab/PerturbNet) links a PERTURBATION representation to a CELL-STATE representation
with a conditional invertible neural network (cINN).  This runner is the GENETIC arm, i.e. the
authors' own Norman/CRISPR configuration:

  genotype  :  target gene -> binary GO-annotation vector (15,988 GO terms)
               -> frozen pretrained GenotypeVAE -> z_pert (10-d)
  cell      :  log-normalised expression -> VAE -> z_cell (10-d)   (`perturbnet.data_vae.vae.VAE`)
  mapping   :  ConditionalFlatCouplingFlow(conditioning_dim=10, embedding_dim=10, in_channels=10,
               n_flows=20, hidden_dim=1024, hidden_depth=2, conditioning_depth=2,
               activation="none", conditioner_use_bn=True)   trained by `train_cinn`

Those numbers are verbatim the authors' genetic tutorial,
notebooks/Tutorial_PerturbNet_Genetic.ipynb cells 16-19 (GenotypeVAE load; flow hyper-parameters).

Released artefacts consumed (nothing is re-derived here)
-------------------------------------------------------
`pretrained_model/genotypeVAE/` on the authors' HuggingFace record cyclopeta/PerturbNet_reproduce,
which the official README (line 52) names as the place "the required data, toy examples, and model
weights can be downloaded from":
    model_params.pt              64 MB  frozen pretrained GenotypeVAE (15988 -> 512 -> 256 -> 10)
    sparse_gene_anno_matrix.npz  344 kB gene x GO binary annotation matrix, 18,832 x 15,988
    gene.npy                     1.1 MB row labels (HGNC symbols) of that matrix
    anno.npy                     625 kB column labels (GO ids)
No weights are redistributed by this repository; the runner reads them from
benchmark/vendor/PerturbNet/pretrained_model/genotypeVAE (override with
$IVCBENCH_PERTURBNET_GENOVAE_DIR).

Task coverage
-------------
T3 (C3_true_lo_gene) and T4 (C4_modality_lo_ko) are NATIVE by the authors' own protocol: the held
gene never appears in training, and its representation comes from the frozen GenotypeVAE applied to
its GO annotation vector -- the published unseen-perturbation recipe (tutorial cells 28-31: tile the
held perturbation's one-hot, push it through GenotypeVAE, sample the cell-state distribution).  The
harness's `side_info['gene_embedding']` is NOT used and NOT required; PerturbNet brings its own
gene-side representation, so `requires_gene_side` stays False on the adapter.

ONE PROFILE PER HELD GENE
-------------------------
T3/T4 hold out the PERTURBATION, not a group, so the output is keyed by gene label and the adapter
keeps `pred_key_is_group = False`.  A gene absent from the released GO table is DECLINED by name
(the harness then keeps the control mean for it and the log says which); training cells carrying a
gene with no GO row are dropped rather than silently collapsed onto the all-zero (= control) vector.

Leak-safety: only payload["X_train"] trains the VAE and the cINN, and only training labels enter the
conditioning library.  The held gene's identity enters solely through the public GO table.  The
runner never sees held-out expression.

Self-check before writing: the emitted profiles must not be a constant repeated across held genes
(the ctrl-pred failure mode found twice in this revision), so the runner reports the across-gene SD
and refuses to write a prediction whose across-gene variation is exactly zero.

Knobs: $IVCBENCH_PERTURBNET_{GENOVAE_DIR,VAE_EPOCHS,CINN_EPOCHS,MAXCELLS,NGEN,BATCH,VAE_LR,CINN_LR}
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np


def _prefer_bundled_cuda_libs() -> None:
    """Put the env's own nvidia/*/lib (torch's bundled cuDNN/cuBLAS) FIRST on LD_LIBRARY_PATH.

    Same one-shot re-exec as perturbnet_c5_runner.py: this host exports a CUDA 11.2/11.6
    LD_LIBRARY_PATH whose cuDNN 8.4 shadows the cuDNN 8.5 torch 2.0.0+cu117 was built against. The
    loader reads LD_LIBRARY_PATH at process start, so the fix must precede the torch import.
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

CONTROL_TOKEN = "control"
Z_PERT = 10  # GenotypeVAE latent width (linear_3_mu: 256 -> 10)
N_GO = 15988  # GenotypeVAE input width (linear_1: 15988 -> 512)

_DEF_GENOVAE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark",
    "vendor",
    "PerturbNet",
    "pretrained_model",
    "genotypeVAE",
)

# A CRISPR label may name more than one target. The authors' recipe for a multi-gene perturbation is
# the OR of the single-gene one-hots (genetic tutorial cell 6), so split on the separators the
# loaders in this repo can produce and take the union.
_SEP = re.compile(r"[/+|,;]|__|\s+")


def _targets(label: str) -> list[str]:
    return [t for t in (s.strip() for s in _SEP.split(str(label))) if t]


# The released gene x GO matrix is keyed on the HGNC symbols current when the authors built it. Three
# targets in this benchmark's CRISPR panels carry the WITHDRAWN symbol, so a literal lookup would
# decline a gene the table actually annotates. Only unambiguous HGNC renames (previous symbol ->
# approved symbol) are listed; paralogues (BOLA2B) and lncRNAs (NEAT1, GAS5, *-AS1), which the table
# genuinely does not annotate, are left to be declined by name.
# Moved to model_runners/gene_alias.py so this runner and scgpt_runner.py cannot drift apart --
# scGPT declined TMEM173 while this runner was already mapping it.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from gene_alias import ALIASES as _ALIAS  # noqa: E402


class _Identity:
    """std_model stand-in with the StandardizeLoad interface.

    Net2NetFlow_TFVAEFlow always standardises its conditioning vector because it was written for the
    ChemicalVAE arm, which ships released ZINC mu/std. The GenotypeVAE arm has no released
    standardiser -- the authors' own genotype class is `Net2NetFlow_TFVAENonStdFlow` ("cINN module
    connecting GenotypeVAE latent space to VAE latent space"), i.e. NO standardisation -- but that
    class still takes a TensorFlow session and cannot drive the torch VAE of the released 0.0.3b1
    package. Passing mu=0, std=1 makes TFVAEFlow numerically identical to the NonStd class.
    """

    def __init__(self, dim, device):
        self.mu = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)
        self.device = device

    def standardize_z(self, z):
        return z

    def standardize_z_torch(self, z):
        return z


def main(in_path: str, out_path: str) -> None:
    import torch
    from perturbnet.genotypevae.genotypeVAE import GenotypeVAE
    from perturbnet.data_vae.vae import VAE
    from perturbnet.cinn.flow import ConditionalFlatCouplingFlow, Net2NetFlow_TFVAEFlow
    from perturbnet.cinn.flow_generate import TFVAEZ_CheckNet2Net

    vae_epochs = int(os.environ.get("IVCBENCH_PERTURBNET_VAE_EPOCHS", "81"))
    cinn_epochs = int(os.environ.get("IVCBENCH_PERTURBNET_CINN_EPOCHS", "50"))
    max_cells = int(os.environ.get("IVCBENCH_PERTURBNET_MAXCELLS", "60000"))
    n_gen = int(os.environ.get("IVCBENCH_PERTURBNET_NGEN", "500"))
    batch = int(os.environ.get("IVCBENCH_PERTURBNET_BATCH", "128"))
    vae_lr = float(os.environ.get("IVCBENCH_PERTURBNET_VAE_LR", "1e-4"))
    cinn_lr = float(os.environ.get("IVCBENCH_PERTURBNET_CINN_LR", "4.5e-6"))
    geno_dir = os.environ.get("IVCBENCH_PERTURBNET_GENOVAE_DIR", _DEF_GENOVAE)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {CONTROL_TOKEN})
    if not test_perts:
        raise RuntimeError("PerturbNet-C3: payload carries no held perturbation label")

    # ---- 1. released GO annotation table -> one-hot per perturbation label --------------------
    from scipy import sparse

    anno_genes = np.load(os.path.join(geno_dir, "gene.npy"), allow_pickle=True)
    anno = sparse.load_npz(
        os.path.join(geno_dir, "sparse_gene_anno_matrix.npz")
    ).toarray()
    if anno.shape[1] != N_GO:
        raise RuntimeError(
            f"PerturbNet-C3: GO table has {anno.shape[1]} columns, GenotypeVAE wants"
            f" {N_GO}"
        )
    row_of = {}
    for i, g in enumerate(anno_genes):
        row_of.setdefault(str(g), i)

    def onehot(label: str) -> np.ndarray | None:
        """Union of the released GO vectors of the label's targets; None if no target is annotated."""
        if label == CONTROL_TOKEN:
            return np.zeros(
                N_GO, dtype=np.float32
            )  # unperturbed genotype = empty annotation set
        tt = [_ALIAS.get(t, t) for t in _targets(label)]
        rows = [row_of[t] for t in tt if t in row_of]
        if not rows:
            return None
        v = anno[rows[0]].astype(bool)
        for r in rows[1:]:
            v = np.logical_or(
                v, anno[r]
            )  # authors' multi-gene recipe (tutorial cell 6)
        return v.astype(np.float32)

    # ---- 2. cap training cells (stratified by label) ------------------------------------------
    labels = np.where(is_ctrl, CONTROL_TOKEN, pert_train).astype(str)
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        keep = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, labels = X[sel], labels[sel]

    # ---- 3. drop training cells whose target has no released GO row; decline such held genes ---
    train_oh = {l: onehot(l) for l in sorted(set(labels))}
    bad_train = sorted([l for l, v in train_oh.items() if v is None])
    if bad_train:
        keep = ~np.isin(labels, bad_train)
        X, labels = X[keep], labels[keep]
        for l in bad_train:
            train_oh.pop(l)
        print(
            f"[PerturbNet-C3] {len(bad_train)} training gene(s) absent from the"
            f" released GO table, dropped: {bad_train[:6]}",
            flush=True,
        )
    if CONTROL_TOKEN not in train_oh:
        raise RuntimeError("PerturbNet-C3: no control cells in the training fold")

    test_oh = {p: onehot(p) for p in test_perts}
    pred_targets = [p for p in test_perts if test_oh[p] is not None]
    declined = [p for p in test_perts if test_oh[p] is None]
    if declined:
        print(
            f"[PerturbNet-C3] declining {len(declined)} held gene(s) with no row in the"
            f" released GO annotation table: {declined}",
            flush=True,
        )
    if not pred_targets:
        raise RuntimeError(
            "PerturbNet-C3: no held gene has a row in the released GO table"
        )

    # Both the VAE and the cINN conditioner use BatchNorm1d, which raises on a final batch of one
    # cell; train_cinn also splits 80/20 internally, so guard all three loader lengths.
    if X.shape[0] < 2 * batch:
        batch = max(8, X.shape[0] // 4)
    _n = X.shape[0]
    _ntr = int(_n * 0.8)
    _nte = _n - _ntr
    while batch > 8 and any(m % batch == 1 for m in (_n, _ntr, _nte)):
        batch -= 1

    print(
        "[PerturbNet-C3] regime=unseen-gene"
        f" train={X.shape} n_train_labels={len(train_oh)} "
        f"targets={len(pred_targets)} declined={len(declined)} dev={dev} "
        f"vae_ep={vae_epochs} cinn_ep={cinn_epochs} batch={batch}",
        flush=True,
    )

    # ---- 4. frozen pretrained GenotypeVAE ------------------------------------------------------
    geno = GenotypeVAE().to(dev)
    geno.load_state_dict(
        torch.load(os.path.join(geno_dir, "model_params.pt"), map_location=dev)
    )
    geno.eval()
    std_model = _Identity(Z_PERT, dev)

    def embed(vec: np.ndarray, reps: int = 1) -> np.ndarray:
        """GO one-hot -> GenotypeVAE representation, tiled `reps` times (tutorial cells 28-31)."""
        oh = np.repeat(vec[None, :], reps, axis=0).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, oh.shape[0], 512):
                _, _, _, z = geno(torch.tensor(oh[i : i + 512]).float().to(dev))
                out.append(z.cpu().numpy())
        return std_model.standardize_z(np.vstack(out)).astype(np.float32)

    # ---- 5. cell representation network: VAE on the training fold only -------------------------
    vae = VAE(
        num_cells_train=X.shape[0],
        x_dimension=X.shape[1],
        learning_rate=vae_lr,
        BNTrainingMode=False,
        device=dev,
    )
    vae.train_np(train_data=X, n_epochs=vae_epochs, batch_size=batch, verbose=False)
    vae.eval()

    # ---- 6. cINN: GenotypeVAE latent -> cell latent --------------------------------------------
    uniq = sorted(train_oh)
    onehot_lib = np.vstack([train_oh[u] for u in uniq]).astype(np.float32)
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
        model_con=geno,
        std_model=std_model,
        model_cell=vae,
    )
    model_c.to(device=dev)
    model_c.train_cinn(n_epochs=cinn_epochs, batch_size=batch, lr=cinn_lr)
    model_c.eval()

    # ---- 7. predict one mean profile per held gene ----------------------------------------------
    pnet = TFVAEZ_CheckNet2Net(model_c, dev, vae)
    # The unseen-perturbation branch samples the cell-state distribution straight from the held
    # gene's representation and consumes no control cells, so the sample count is a free knob rather
    # than a function of the payload's control cloud: draw the full budget (500 by default, above the
    # authors' tutorial n_cells = 200) and average, which is the comparison the paper itself makes
    # ("compute the mean gene expression values for every gene ... of both predicted and real cells").
    n = n_gen
    pred_perts, pred_means = [], []
    for g in pred_targets:
        emb = embed(test_oh[g], reps=n)
        _, gen = pnet.sample_data(emb)
        pred_perts.append(g)
        pred_means.append(np.asarray(gen, np.float32).mean(0))

    P = np.vstack(pred_means).astype(np.float32)
    if P.shape[1] != len(genes):
        raise RuntimeError(
            f"PerturbNet-C3: emitted {P.shape[1]} genes, payload has {len(genes)}"
        )
    if not np.isfinite(P).all():
        raise RuntimeError("PerturbNet-C3: non-finite values in the predicted profiles")
    # guard against the silent ctrl-pred failure mode: a genuine unseen-gene prediction must vary
    # across the held genes it was conditioned on
    spec = float(np.mean(P.std(axis=0))) if P.shape[0] > 1 else float("nan")
    if P.shape[0] > 1 and np.allclose(P, P[0]):
        raise RuntimeError(
            "PerturbNet-C3: identical profile for every held gene — the conditioning "
            "had no effect; refusing to emit a prediction that is the control mean"
        )
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[PerturbNet-C3] wrote {len(pred_perts)} gene profiles x {P.shape[1]} genes"
        f" ({len(declined)} declined); across-gene SD (specificity) = {spec:.5f} ->"
        f" {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
