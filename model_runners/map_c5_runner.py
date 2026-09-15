#!/usr/bin/env python
"""MAP runner for the COMPOUND cluster (C5 / OP3) -- `ivc-map` conda env.

Invoked by ivcbench.baselines.heavy.MAPC5:
    <env python> map_c5_runner.py <in.npz> <out.npz>

WHAT MAP IS
-----------
MAP (Feng et al., "A knowledge-driven framework for predicting single-cell responses for
unprofiled drugs", Nature Machine Intelligence 2026, s42256-026-01286-w; preprint bioRxiv
2026.02.25.708091; code github.com/MAGIC-AI4Med/MAP, MIT).  A frozen 600M-parameter state
encoder (Arc SE-600M) embeds a bulk of control cells; a FROZEN knowledge encoder embeds the
compound's SMILES into a mechanism-aware space pre-trained on MAP-KG (187,089 drugs / 22,924
genes / 694,246 mechanistic relations, aligning structure, protein targets and textual
mechanism); a llama-backbone perturbation transformer reads [cell | drug | gene] tokens and a
gene decoder emits the perturbed profile.  The compound therefore enters as SMILES + knowledge,
not as a Morgan fingerprint, which is why this runner takes `smiles_keys`/`smiles_vals` from the
payload (heavy.MAPC5 injects them from `cs.uns['smiles']`, the map the OP3 loader keeps at
src/ivcbench/data/loaders/op3.py:117) and never touches `fingerprint_*`.

MAP is the most on-point comparator in the C5 survey: it was benchmarked ON OP3 ITSELF --
"OP3 dataset ... 240,059 cells ... 147 perturbation conditions, including 144 compounds
administered at 1 uM", "annotated into six immune cell types".

THE PUBLISHED OP3 PROTOCOL, AND HOW OURS DIFFERS
------------------------------------------------
Paper, verbatim: "Following the same splitting strategy as for Tahoe-100M, we held out 5% of
drugs per cell type to evaluate generalization to unseen cell type-drug combinations, and an
additional 5% of all drugs to assess generalization to unseen drugs."

  * T5u  (C5_global_compound_holdout).  MAP's *unprofiled-drug* regime is the published analogue:
    a global fraction of drugs is withheld and every profile of those drugs is removed from
    training.  Ours withholds 28 of 141 fingerprinted compounds (20%, seed 0); MAP withholds 5%.
    Same regime, four times the holdout -- disclosed, not hidden.
  * T5c  (C5_loct_<lineage>).  MAP has NO published analogue.  Its OP3 cell-context experiment
    holds out drug x cell-type PAIRS ("unseen cell type-drug combinations"), and the released
    splitter (preprocess/Cs_split_unseen_combination.py::allocate_unseen_drugs, unseen_ratio=0.05,
    seed=42) allocates a disjoint 5% of drugs PER cell line -- every cell type is present in
    training for every other cell type's drugs.  Holding out an entire lineage is not an entry
    point MAP publishes, so T5c is not native by the benchmark's interface rule.

THE LEAK GATE (the reason this runner exists in this form)
---------------------------------------------------------
Paper, verbatim: "To avoid information leakage, we additionally exclude the held-out drug
entities (and their aliases, if applicable) from MAP-KG during knowledge pre-training."

The knowledge encoder is FROZEN inside the model (model/pert.py: `self.kg_smiles_encoder ...
requires_grad = False`), so whatever it learned about a compound is carried into every
prediction.  MAP's own protocol therefore requires that a held compound be absent from MAP-KG
*before* knowledge pre-training.  The released encoder (`mapkg_encoder_v3.pt`) was pre-trained
with THEIR holdout excluded; our seed-0 20% holdout is a construct of this benchmark and cannot
have been excluded from it.

So this runner performs the check MAP's protocol implies, against the released MAP-KG drug node
table (HuggingFace RainGate/MAP-KG :: DRUG_merged_drugs_with_residuals.csv), matching each held
compound by canonical SMILES and by normalised name/alias.  If any held compound is an entity in
that table, the unseen-compound cell LEAKS and the runner refuses to produce a number.

The refusal is escapable exactly once, and only honestly: set
    $IVCBENCH_MAP_KG_ENCODER   -> a knowledge encoder re-pre-trained with the held compounds and
                                  their aliases removed from MAP-KG
    $IVCBENCH_MAP_KG_EXCLUDED  -> the file listing what was excluded from that pre-training
and the gate verifies the held set is covered by that list before it lets the run proceed.

Knobs: $IVCBENCH_MAP_VENDOR, $IVCBENCH_MAP_SE_CKPT, $IVCBENCH_MAP_SE_CONFIG, $IVCBENCH_MAP_CKPT,
$IVCBENCH_MAP_KG_ENCODER, $IVCBENCH_MAP_MOLSTM, $IVCBENCH_MAP_MOLSTM_VOCAB,
$IVCBENCH_MAP_ESM2, $IVCBENCH_MAP_HVG_LIST, $IVCBENCH_MAP_SETSIZE, $IVCBENCH_MAP_MAXSRC.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

CONTROL = "control"
VENDOR = Path(
    os.environ.get(
        "IVCBENCH_MAP_VENDOR",
        "/data1/home/chlee/projects/immune virtual cell/benchmark/vendor/MAP",
    )
)
CKPTS = VENDOR / "checkpoints"

# --------------------------------------------------------------------------------------------
# released artefacts MAP needs before any forward pass is defined
# --------------------------------------------------------------------------------------------
ARTIFACTS = {
    "SE-600M weights (se600m.safetensors)": (
        os.environ.get("IVCBENCH_MAP_SE_CKPT", str(CKPTS / "se600m.safetensors")),
        "HuggingFace arcinstitute/SE-600M :: model.safetensors",
    ),
    "SE-600M config (se600m.yaml)": (
        os.environ.get(
            "IVCBENCH_MAP_SE_CONFIG", str(VENDOR / "MAP" / "configs" / "se600m.yaml")
        ),
        "in the MAP repo",
    ),
    "ESM2 gene embeddings": (
        os.environ.get(
            "IVCBENCH_MAP_ESM2",
            str(CKPTS / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"),
        ),
        "the authors' Google Drive folder",
    ),
    "MAP-KG knowledge encoder": (
        os.environ.get("IVCBENCH_MAP_KG_ENCODER", str(CKPTS / "mapkg_encoder_v3.pt")),
        "the authors' Google Drive folder",
    ),
    "MAP perturbation model": (
        os.environ.get("IVCBENCH_MAP_CKPT", str(CKPTS / "epoch_3.pt")),
        "the authors' Google Drive folder",
    ),
    "MoleculeSTM MegaMolBART weights": (
        os.environ.get("IVCBENCH_MAP_MOLSTM", str(CKPTS / "molecule_model.pth")),
        (
            "HuggingFace chao1224/MoleculeSTM ::"
            " demo/demo_checkpoints_SMILES/molecule_model.pth"
        ),
    ),
    "MegaMolBART vocabulary": (
        os.environ.get("IVCBENCH_MAP_MOLSTM_VOCAB", str(CKPTS / "bart_vocab.txt")),
        "github.com/chao1224/MoleculeSTM :: MoleculeSTM/bart_vocab.txt",
    ),
}

# released SOURCE files that the OP3 entry point imports but that the repository does not ship
REQUIRED_SOURCES = {
    "MAP/model/pert_ca.py": (
        "imported by MAP/model/model.py:8 (`from model.pert_ca import"
        " PerturbationEncoder_ca`); without it `import model.model` raises ImportError,"
        " so no MAP model can be built. It is also the class the RELEASED WEIGHTS"
        " belong to: epoch_3.pt stores `pert_model.smiles_encoder._model.*` and"
        " `pert_model.drug_projector.*`, neither of which exists on the shipped"
        " PerturbationEncoder (MAP/model/pert.py builds `kg_smiles_encoder` and no drug"
        " projector), so demo.py's `load_state_dict(..., strict=True)` cannot succeed"
        " against the released source"
    ),
    "MAP/data/ds_nips_lora_se.py": (
        "imported by MAP/train_nips.py:24 (`from data.ds_nips_lora_se import"
        " NIPSPerturbDatasetSE`); this is the OP3 data module -- without it MAP cannot"
        " be trained on OP3 at all. Retraining is the only route to an OP3 model: the"
        " released checkpoint epoch_3.pt is a TAHOE-100M run (its stored args carry"
        " cell_lines = CVCL_0023/0069/0131/0480/1056/1098, Cellosaurus lines, not OP3's"
        " six immune cell types), so no released OP3 MAP checkpoint exists"
    ),
    "preprocess/gene_vocab_hgnc.json": (
        "read by"
        " preprocess/data/ds_tahoe_se.py::CellDatasetCollatorSEOnly._load_gene_mapping;"
        " the local-gene-index -> HGNC map that turns counts into SE tokens"
    ),
}


def _missing_sources() -> list[str]:
    return [
        f"{rel}  -- {why}"
        for rel, why in REQUIRED_SOURCES.items()
        if not (VENDOR / rel).exists()
    ]


def _missing_artifacts() -> list[str]:
    out = []
    for label, (path, where) in ARTIFACTS.items():
        if not Path(path).exists():
            out.append(f"{label}: {path}  (obtain from: {where})")
    return out


# --------------------------------------------------------------------------------------------
# the leak gate
# --------------------------------------------------------------------------------------------
def _mapkg_csv() -> Path:
    p = Path(
        os.environ.get(
            "IVCBENCH_MAPKG_DRUG_CSV",
            str(
                VENDOR
                / "MAP-KG"
                / "data"
                / "selected_data_csvs"
                / "DRUG_merged_drugs_with_residuals.csv"
            ),
        )
    )
    if not p.exists():
        raise RuntimeError(
            "MAP-C5 leak gate: the released MAP-KG drug node table is not on disk at"
            f" {p}. Download it from HuggingFace RainGate/MAP-KG"
            " (DRUG_merged_drugs_with_residuals.csv) or point $IVCBENCH_MAPKG_DRUG_CSV"
            " at it. The gate cannot be skipped: without the table the leak status of"
            " the held compounds is unknown, and an unknown leak status is not a"
            " licence to report."
        )
    return p


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _aliases(name: str) -> set[str]:
    """OP3 sm_name spellings: 'A;B' synonym pairs and 'Name (CODE)' parentheticals."""
    out = set()
    for part in str(name).split(";"):
        part = part.strip()
        if not part:
            continue
        out.add(part)
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", part)
        if m:
            out |= {m.group(1).strip(), m.group(2).strip()}
    return {a for a in out if a}


def _canon(smiles):
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles)) if smiles else None
    return Chem.MolToSmiles(m) if m is not None else None


def leak_check(
    held: list[str], smiles_map: dict[str, str]
) -> list[tuple[str, str, str]]:
    """-> [(compound, how it was found in MAP-KG, the MAP-KG mechanism text)] for every held
    compound that is an entity of the released MAP-KG the knowledge encoder was pre-trained on.
    """
    import pandas as pd

    kg = pd.read_csv(_mapkg_csv(), low_memory=False)
    by_name: dict[str, list[int]] = {}
    for n, i in zip(kg["Drug Name"].astype(str), kg.index):
        by_name.setdefault(_norm(n), []).append(i)
    by_smiles: dict[str, list[int]] = {}
    for s, i in zip(kg["smiles"].astype(str), kg.index):
        c = _canon(s)
        if c:
            by_smiles.setdefault(c, []).append(i)

    hits = []
    for cpd in held:
        rows, how = [], []
        c = _canon(smiles_map.get(cpd))
        if c and c in by_smiles:
            rows += by_smiles[c]
            how.append("canonical SMILES")
        for a in _aliases(cpd):
            if _norm(a) in by_name:
                rows += by_name[_norm(a)]
                how.append(f"name '{a}'")
        if rows:
            moa = ""
            for i in sorted(set(rows)):
                v = kg.loc[i, "moa"]
                if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                    moa = v.strip()
                    break
            hits.append((cpd, " + ".join(dict.fromkeys(how)), moa))
    return hits


def _cleared_by_retraining(held: list[str]) -> bool:
    """The only honest escape: a knowledge encoder re-pre-trained with these compounds excluded."""
    enc = os.environ.get("IVCBENCH_MAP_KG_ENCODER_RETRAINED")
    lst = os.environ.get("IVCBENCH_MAP_KG_EXCLUDED")
    if not (enc and lst and Path(enc).exists() and Path(lst).exists()):
        return False
    excluded = {_norm(x) for x in Path(lst).read_text().splitlines() if x.strip()}
    uncovered = [c for c in held if not any(_norm(a) in excluded for a in _aliases(c))]
    if uncovered:
        raise RuntimeError(
            "MAP-C5 leak gate: $IVCBENCH_MAP_KG_EXCLUDED does not cover"
            f" {len(uncovered)} held compounds, e.g. {uncovered[:3]}. The"
            " re-pre-trained encoder is not leak-free for this split."
        )
    return True


# --------------------------------------------------------------------------------------------
def main(in_path: str, out_path: str) -> None:
    d = np.load(in_path, allow_pickle=True)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {CONTROL})
    if not test_perts:
        raise RuntimeError("MAP-C5: no held compound label in the payload")

    # ---- SMILES, not fingerprints.  MAP's drug branch is a SMILES encoder.
    if "smiles_keys" in d.files:
        smi = {str(k): str(v) for k, v in zip(d["smiles_keys"], d["smiles_vals"])}
        src = "payload smiles_keys/smiles_vals (from cs.uns['smiles'])"
    else:
        import anndata

        p = os.environ.get(
            "IVCBENCH_OP3_PATH", "data/C5/op3/GSE279945_sc_counts_processed.h5ad"
        )
        obs = anndata.read_h5ad(p, backed="r").obs
        smi = (
            obs[["sm_name", "SMILES"]]
            .astype(str)
            .drop_duplicates("sm_name")
            .set_index("sm_name")["SMILES"]
            .to_dict()
        )
        src = f"OP3 obs on disk ({p})"
    missing_smi = [c for c in test_perts if not smi.get(c)]
    print(
        f"[MAP-C5] SMILES source: {src};"
        f" {len(test_perts) - len(missing_smi)}/{len(test_perts)} held compounds have a"
        " SMILES string",
        flush=True,
    )

    # ---- which regime?  T5c holds a GROUP (all compounds seen); T5u holds the COMPOUNDS.
    pert_train = {str(p) for p in d["pert_train"]} - {CONTROL}
    unseen = [c for c in test_perts if c not in pert_train]
    regime = "T5u/unseen-compound" if unseen else "T5c/held-lineage"
    print(
        f"[MAP-C5] regime={regime}: {len(unseen)}/{len(test_perts)} held compounds are"
        f" absent from the {len(pert_train)} training compounds",
        flush=True,
    )

    # ---- LEAK GATE (unseen-compound regime only; on T5c every compound is seen by design) ----
    if unseen and not _cleared_by_retraining(unseen):
        hits = leak_check(unseen, smi)
        if hits:
            lines = "\n".join(
                f"      {c:46s} matched by {how}"
                + (f"   MAP-KG moa: {moa}" if moa else "")
                for c, how, moa in hits[:40]
            )
            raise RuntimeError(
                "MAP-C5 LEAK GATE FAILED -- this cell must not be reported.\n "
                f" {len(hits)} of the {len(unseen)} held-out compounds are entities of"
                " the released MAP-KG that the frozen knowledge encoder was"
                f" pre-trained on:\n{lines}\n  MAP's own protocol (paper, Methods):"
                ' "To avoid information leakage, we additionally exclude the held-out'
                " drug entities (and their aliases, if applicable) from MAP-KG during"
                ' knowledge pre-training."  The released encoder excluded the'
                " authors' 5% holdout, not this benchmark's seed-0 20% holdout, and"
                " the encoder is frozen inside the model (MAP/model/pert.py:"
                " `requires_grad = False`), so its knowledge of these compounds"
                " reaches every prediction.\n  Clear it by re-pre-training the"
                " knowledge encoder with these compounds and their aliases removed"
                " from MAP-KG (MAP-KG/train_resume.py; the node tables are public at"
                " HuggingFace RainGate/MAP-KG), then set"
                " $IVCBENCH_MAP_KG_ENCODER_RETRAINED and $IVCBENCH_MAP_KG_EXCLUDED."
            )
        print(
            f"[MAP-C5] leak gate: none of the {len(unseen)} held compounds is a MAP-KG"
            " entity",
            flush=True,
        )

    # ---- released-code and released-artefact preflight -------------------------------------
    miss_src, miss_art = _missing_sources(), _missing_artifacts()
    if miss_src or miss_art:
        raise RuntimeError(
            "MAP-C5: the released MAP distribution is incomplete for this task.\n"
            + (
                "  MISSING SOURCE FILES (github.com/MAGIC-AI4Med/MAP @ "
                + os.popen(f'git -C "{VENDOR}" rev-parse --short HEAD 2>/dev/null')
                .read()
                .strip()
                + "):\n"
                + "\n".join(f"    - {m}" for m in miss_src)
                if miss_src
                else ""
            )
            + (
                ("\n" if miss_src else "")
                + "  MISSING ARTEFACTS:\n"
                + "\n".join(f"    - {m}" for m in miss_art)
                if miss_art
                else ""
            )
        )

    # ---- build and run the published model --------------------------------------------------
    sys.path.insert(0, str(VENDOR / "MAP"))
    import torch
    from omegaconf import OmegaConf
    from model.model import MAPmodel  # noqa: E402

    cfg = OmegaConf.load(ARTIFACTS["SE-600M config (se600m.yaml)"][0])
    cfg.embeddings[cfg.embeddings.current].all_embeddings = ARTIFACTS[
        "ESM2 gene embeddings"
    ][0]
    model = MAPmodel(
        se_ckpt=ARTIFACTS["SE-600M weights (se600m.safetensors)"][0],
        se_cfg=cfg,
        smile_encoder="MAP-KG",
        hvg_info=None,
    )
    ckpt = torch.load(
        ARTIFACTS["MAP perturbation model"][0], map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    # The gene decoder emits MAP's own 2000-gene HVG panel, fixed at ITS training time. Mapping
    # those columns onto this benchmark's 2000 HVGs needs that panel's gene list, which the
    # release does not ship; without it the output columns are unlabelled and cannot be scored.
    hvg_list = os.environ.get("IVCBENCH_MAP_HVG_LIST")
    if not hvg_list or not Path(hvg_list).exists():
        raise RuntimeError(
            "MAP-C5: the released checkpoint's gene decoder outputs 2000 columns in"
            " MAP's own HVG panel (MAP/model/gene_decoders.py::LatentToGeneDecoder,"
            " gene_dim=2000, built by preprocess/E_hvg_multi_celllines.py), and the"
            " release does not ship that panel's gene list. Point"
            " $IVCBENCH_MAP_HVG_LIST at it, or retrain the perturbation head on this"
            " fold so the decoder is defined over this benchmark's gene panel."
        )

    genes = [str(g) for g in d["genes"]]
    map_genes = [
        ln.strip() for ln in Path(hvg_list).read_text().splitlines() if ln.strip()
    ]
    col = {g: i for i, g in enumerate(map_genes)}
    take = np.array([col.get(g, -1) for g in genes])
    if (take < 0).any():
        raise RuntimeError(
            f"MAP-C5: {(take < 0).sum()} of {len(genes)} benchmark genes are "
            "absent from MAP's HVG panel; the cell is not scoreable as-is."
        )

    set_size = int(os.environ.get("IVCBENCH_MAP_SETSIZE", "24"))
    max_src = int(os.environ.get("IVCBENCH_MAP_MAXSRC", "480"))
    from map_se_tokens import se_tokens_from_expression  # vendored beside runner

    Xsrc = d["X_ctrl_inf"].astype(np.float32)[:max_src]
    gene_ids, expr = se_tokens_from_expression(Xsrc, genes, cfg)
    pred_perts, pred_means = [], []
    for c in test_perts:
        if not smi.get(c):
            continue
        prof = []
        for i in range(0, gene_ids.shape[0], set_size):
            gi = torch.from_numpy(gene_ids[i : i + set_size]).unsqueeze(0).to(dev)
            ex = torch.from_numpy(expr[i : i + set_size]).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                _, hvg = model(gi, ex, [smi[c]], torch.tensor([1.0], device=dev))
            prof.append(hvg[0].float().cpu().numpy())
        pred_perts.append(c)
        pred_means.append(np.vstack(prof).mean(0)[take].astype(np.float32))

    P = np.vstack(pred_means).astype(np.float32)
    if not np.isfinite(P).all():
        raise RuntimeError("MAP-C5: non-finite values in the generated profiles")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[MAP-C5] wrote {len(pred_perts)} compound profiles x {P.shape[1]} genes",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
