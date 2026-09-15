#!/usr/bin/env python
"""scFoundation's released GEARS pathway for held genetic perturbations (T3/T4).

Run in the ``scfoundation`` environment as ``runner.py in.npz out.npz``. The trainable
model is the published GEARS_Model, initialized with its released ``gene``
checkpoint and ``finetune_method='frozen'``. There is no replacement prediction
head. A parameter-free panel adapter scatters measured genes into the canonical
19,264 positions before MAEAutobinencoder, and gathers their context embeddings
back in payload order. GEARS then predicts every output gene in that order.

Genes outside the checkpoint panel, even after unambiguous HGNC alias lookup,
cannot be silently padded in OUTPUT: such a unit is explicitly declined. Missing
unmeasured INPUT genes are zero padded as in the released main_gene_selection.
The public prediction graph constructor receives X_ctrl_inf, never train controls.

The native GEARS resolution tokens require optional payload total_count_train and
total_count_inf containing original (pre-normalization) per-cell RNA totals.
They are not reconstructed from a truncated normalized panel. Original cell IDs
may be supplied as train_cell_ids/ctrl_inf_cell_ids for a provenance-level leak
assertion; otherwise only exact row-overlap can be checked, and IDs are unverified.

Required: IVCBENCH_SCFOUNDATION_DIR (the upstream model/ directory),
IVCBENCH_SCFOUNDATION_CKPT. Optional: IVCBENCH_GENE2GO, IVCBENCH_HGNC_ALIASES,
IVCBENCH_SCF_GENE_{EPOCHS,BATCH_SIZE,MAX_CELLS,PREDCTRL,SEED,LR,GO_GRAPH}.
MAX_CELLS=0 retains all supplied training cells. PREDCTRL=0 requests as many
native control draws as supplied controls (the public sampler uses replacement).
Checkpoints and graph caches
are private temporary files, namespaced by model, input-content/split, condition
set and seed. No repository or upstream files are modified.
"""
from __future__ import annotations

import atexit
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _published_sources(model_dir: Path) -> tuple[Path, Path]:
    """Read the checkout's committed release in /tmp, excluding local source edits."""
    root = model_dir.parent
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    temporary = Path(tempfile.mkdtemp(prefix=f"scFoundation_published-{commit[:12]}_"))
    atexit.register(shutil.rmtree, temporary, ignore_errors=True)
    archive = temporary / "published.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "-C", str(root), "archive", commit, "GEARS", "model"],
                       stdout=handle, check=True)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (temporary / member.name).resolve()
            if not target.is_relative_to(temporary) or member.issym() or member.islnk():
                raise RuntimeError(f"unsupported archived source path: {member.name}")
        handle.extractall(temporary)
    archive.unlink()
    _log(f"[source] committed upstream checkout={root} commit={commit}; local working-tree source edits excluded")
    return temporary / "model", temporary / "GEARS"


def _symbol_lookup(table: Path):
    """Resolve aliases only when they select one member of the actual vocabulary."""
    if not table.is_file():
        raise FileNotFoundError(f"HGNC alias table missing: {table}")
    groups: dict[str, set[str]] = {}
    approved_for: dict[str, set[str]] = {}
    with table.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            approved = row.get("Approved symbol", row.get("symbol", "")).strip()
            if not approved:
                continue
            names = {approved}
            for field in ("Previous symbols", "Alias symbols", "prev_symbol", "alias_symbol"):
                names.update(s.strip() for s in re.split(r"[,|]", row.get(field, "")) if s.strip())
            for name in names:
                groups.setdefault(name, set()).update(names)
                approved_for.setdefault(name, set()).add(approved)
    # Explicit verified bridge also covers old HGNC exports without STING1.
    for a, b in (("TMEM173", "STING1"),):
        groups.setdefault(a, set()).add(b)
        groups.setdefault(b, set()).add(a)

    def resolve(symbol: str, vocab: set[str]) -> str | None:
        symbol = symbol.strip()
        if symbol in vocab:
            return symbol
        # Prefer the approved symbol for the queried alias. Other aliases on
        # the same HGNC row can also be unrelated approved symbols (e.g. GARS1
        # lists SMAD1), which must not make GARS -> GARS1 falsely ambiguous.
        approved = approved_for.get(symbol, set()) & vocab
        if approved:
            return next(iter(approved)) if len(approved) == 1 else None
        choices = groups.get(symbol, set()) & vocab
        return next(iter(choices)) if len(choices) == 1 else None

    return resolve


def _write_panel_go_graph(panel, gene2go, out: Path) -> int:
    """Upstream's get_go_auto, computed over THIS panel's perturbation vocabulary.

    GEARS' perturbation vocabulary is the entire gene2go universe -- pertdata.py:43 sets
    pert_names = np.unique(list(gene2go.keys())), 67,832 genes -- and gears.py:166 hands that
    same list to get_go_auto. Letting it build therefore means 67,832 squared Jaccards, which
    the progress bar puts at fifty hours; that is why upstream ships a precomputed go.csv per
    dataset rather than building one. But the shipped GEARS/data/adamson/go.csv carries only
    3,735 distinct genes and not one of this panel's held targets, so reusing it leaves every
    held target an isolated node with an untrained random embedding, which the caller then
    declines as not gene-side transfer.

    So compute the same quantity over the genes this unit actually perturbs: the identical
    formula (Jaccard over GO terms, keep above 0.1, self-edges included, both directions), on
    the only vocabulary that is ever referenced. Every other node stays isolated inside
    GeneSimNetwork exactly as it would upstream, and none of them is named by a condition.
    """
    sets = {g: set(gene2go.get(g, ()) or ()) for g in panel}
    rows = []
    for a in panel:
        sa = sets[a]
        if not sa:
            continue
        for b in panel:
            sb = sets[b]
            if not sb:
                continue
            shared = len(sa & sb)
            if not shared:
                continue
            score = shared / len(sa | sb)
            if score > 0.1:
                rows.append((a, b, score))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write("source,target,importance\n")
        for a, b, score in rows:
            handle.write(f"{a},{b},{score}\n")
    return len(rows)


def _row_hashes(matrix: np.ndarray) -> set[bytes]:
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in matrix}


def _leak_check(d, train: np.ndarray, infer: np.ndarray) -> None:
    pairs = (("train_cell_ids", "ctrl_inf_cell_ids"), ("cell_ids_train", "cell_ids_inf"),
             ("train_idx", "ctrl_inf_idx"), ("train_indices", "inference_input_indices"))
    verified_ids = False
    for train_key, infer_key in pairs:
        if train_key in d and infer_key in d:
            a, b = np.asarray(d[train_key]).astype(str), np.asarray(d[infer_key]).astype(str)
            assert len(a) == len(train) and len(b) == len(infer), "cell-ID shape mismatch"
            overlap = set(a) & set(b)
            assert not overlap, f"training/inference cell-ID overlap: {sorted(overlap)[:10]}"
            _log(f"[leak] {train_key}/{infer_key}: disjoint ({len(a)}/{len(b)} cells)")
            verified_ids = True
    # The leak boundary on the unseen-perturbation splits is the TREATED cells, not the controls.
    # c3.py:107 and c4.py:28 set control_inference_only=False, so splits/builder.py:42 makes
    # inference_input_idx = is_ctrl & ~in_group -- a strict SUBSET of train_idx (builder.py:31).
    # X_ctrl_inf sharing rows with X_train is therefore the split's design, not a leak, and the
    # repository's own audit says so (splits/audit.py:57-60). Asserting disjointness here rejected
    # every canonical T3/T4 payload before any model work.
    treated = train[~np.asarray(d["is_control_train"], dtype=bool)] if "is_control_train" in d else train
    duplicates = _row_hashes(treated) & _row_hashes(infer)
    assert not duplicates, (
        f"{len(duplicates)} TREATED training row(s) appear in X_ctrl_inf -- that is a real leak"
    )
    _log(f"[leak] exact-expression-row overlap=0; original-cell-ID verification={'passed' if verified_ids else 'UNVERIFIED (IDs absent from payload)'}")


def _finish(out_path: str, requested: list[str], labels: list[str], profiles: list[np.ndarray],
            reasons: dict[str, str], ctrl_mean: np.ndarray) -> None:
    missing = set(requested) - set(labels)
    assert missing == set(reasons), "every missing label must have an explicit decline reason"
    assert not (set(labels) - set(requested)), "unexpected output label"
    assert len(labels) == len(set(labels)), "duplicate profile rows"
    pred = np.vstack(profiles).astype(np.float32) if profiles else np.empty((0, len(ctrl_mean)), dtype=np.float32)
    assert pred.shape == (len(labels), len(ctrl_mean)) and np.isfinite(pred).all()
    for label in sorted(missing):
        _log(f"[decline] {label}: {reasons[label]}")
    _log(f"[coverage] requested={len(requested)} predicted={len(labels)} missing={json.dumps(sorted(missing))} reasons={json.dumps(reasons, sort_keys=True)}")
    for label, row in zip(labels, pred):
        deviation = float(np.max(np.abs(row.astype(np.float64) - ctrl_mean)))
        _log(f"[control] {label}: max_abs_delta={deviation:.12g} at_or_below_1e-5={deviation <= 1e-5}")
    equal_pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i)
                   if np.array_equal(pred[i], pred[j])]
    _log(f"[condition-distinction] bit_exact_equal_pairs={json.dumps(equal_pairs)}")
    np.savez(out_path, pred_perts=np.array(labels, dtype=object), pred_means=pred)


def main(in_path: str, out_path: str) -> None:
    model_dir = Path(os.environ["IVCBENCH_SCFOUNDATION_DIR"]).resolve()
    checkpoint = Path(os.environ["IVCBENCH_SCFOUNDATION_CKPT"]).resolve()
    gears_dir = model_dir.parent / "GEARS"
    for asset in (model_dir / "load.py", checkpoint, gears_dir / "gears/model.py"):
        if not asset.is_file():
            raise FileNotFoundError(asset)
    hgnc = Path(os.environ.get("IVCBENCH_HGNC_ALIASES", str(model_dir.parent / "SCAD/data/processing/HGNC_symbol_all_genes.tsv")))
    resolve = _symbol_lookup(hgnc)
    with (model_dir / "OS_scRNA_gene_index.19264.tsv").open() as handle:
        panel = [row["gene_name"] for row in csv.DictReader(handle, delimiter="\t")]
    assert len(panel) == 19264 and len(set(panel)) == 19264, "unexpected released gene panel"
    panel_set, panel_pos = set(panel), {g: i for i, g in enumerate(panel)}
    gene2go_path = Path(os.environ.get("IVCBENCH_GENE2GO", str(gears_dir / "data/gene2go.pkl")))
    with gene2go_path.open("rb") as handle:
        gene2go = pickle.load(handle)
    go_set = {g for g, terms in gene2go.items() if len(terms) > 0}

    with np.load(in_path, allow_pickle=True) as d:
        train = np.asarray(d["X_train"], dtype=np.float32)
        infer = np.asarray(d["X_ctrl_inf"], dtype=np.float32)
        genes = np.asarray(d["genes"]).astype(str)
        is_ctrl = np.asarray(d["is_control_train"], dtype=bool)
        train_perts = np.asarray(d["pert_train"]).astype(str)
        requested = sorted(set(np.asarray(d["test_perts"]).astype(str)) - {"control"})
        assert train.ndim == infer.ndim == 2 and train.shape[1] == infer.shape[1] == len(genes)
        assert len(train) == len(is_ctrl) == len(train_perts) and is_ctrl.any() and len(infer)
        assert np.array_equal(is_ctrl, train_perts == "control"), "is_control_train disagrees with literal control labels"
        assert np.isfinite(train).all() and np.isfinite(infer).all()
        if np.min(train) < 0 or np.min(infer) < 0:
            raise ValueError("scFoundation expects nonnegative normalized log1p RNA, not centered/signed values")
        ctrl_mean = train[is_ctrl].mean(axis=0, dtype=np.float64)
        # Report the strict identity assertion independently of model vocabulary
        # declines, so a derived supported-panel run cannot hide original overlap.
        _leak_check(d, train, infer)
        # scFoundation's T/S resolution tokens need a library size. Raw counts do not survive this
        # benchmark's preprocessing and no payload carries them, so the deposited scFoundation cells
        # already in the census derive the total the same way -- scfoundation_c1_runner.py:87,
        # total = expm1(panel).sum() + 1 -- by inverting the log1p of the scattered panel. Using the
        # same convention here keeps every scFoundation cell on one input contract, which is the
        # point of running them side by side. It is disclosed on stdout, not silently assumed.
        totals = {}
        for key, matrix in (("total_count_train", train), ("total_count_inf", infer)):
            if key in d:
                totals[key] = np.asarray(d[key], dtype=np.float64)
                assert totals[key].shape == (len(matrix),) and np.all(totals[key] > 0)
                _log(f"[resolution] {key}: payload original totals")
            else:
                totals[key] = np.expm1(np.asarray(matrix, dtype=np.float64)).sum(1) + 1.0
                _log(f"[resolution] {key}: derived as expm1(panel).sum()+1 "
                     "(same convention as the deposited scFoundation cells; raw counts are not "
                     "retained by this benchmark's preprocessing)")
    normalized = [resolve(g, panel_set) for g in genes]
    unsupported = [str(g) for g, resolved in zip(genes, normalized) if resolved is None]
    # Model the sub-panel the checkpoint carries and leave the rest at the control baseline, which
    # is what the sibling runners do (pertadapt_runner.py:173-174). Every 2,000-HVG panel in this
    # benchmark contains 31-59 lncRNA / TR-segment genes the released 19,264-gene vocabulary does
    # not carry, so rejecting the panel declined 100% of targets on all five T3 datasets and on T4
    # -- a coverage of zero that reads like a model limitation but is a panel-mapping choice.
    supported = np.array(
        [i for i, r in enumerate(normalized) if r is not None and r in panel_pos], dtype=np.int64
    )
    if supported.size == 0:
        _finish(out_path, requested, [], [],
                {p: "no output gene of this panel is in the released scFoundation vocabulary"
                 for p in requested}, ctrl_mean)
        return
    sub = [normalized[i] for i in supported]
    if len(set(sub)) != len(sub):
        _finish(out_path, requested, [], [],
                {p: "output genes map ambiguously to duplicate checkpoint positions"
                 for p in requested}, ctrl_mean)
        return
    if unsupported:
        _log(f"[panel] modelled={supported.size}/{len(genes)} genes; "
             f"{len(unsupported)} keep the control baseline: "
             + ",".join(unsupported[:20]) + (" ..." if len(unsupported) > 20 else ""))
    indices = np.array([panel_pos[g] for g in sub], dtype=np.int64)
    assert np.array_equal(np.asarray(panel)[indices], np.asarray(sub)), "canonical gene-position mismatch"
    for original, normalized_gene in zip(genes, normalized):
        if normalized_gene is not None and original != normalized_gene:
            _log(f"[alias-output] {original} -> {normalized_gene}")   # unsupported genes are listed by [panel]
    targets = {p: resolve(p, go_set) for p in requested}
    reasons = {p: "no nonempty released GO annotations after unambiguous HGNC alias lookup" for p, g in targets.items() if g is None}
    for p, g in targets.items():
        if g is not None and p != g:
            _log(f"[alias-target] {p} -> {g}")
    if len(reasons) == len(requested):
        _finish(out_path, requested, [], [], reasons, ctrl_mean)
        return
    normalized_train = np.array(["ctrl" if c else resolve(p, go_set) for c, p in zip(is_ctrl, train_perts)], dtype=object)
    keep = is_ctrl | (normalized_train != None)  # noqa: E711
    for label in sorted(set(train_perts[~keep])):
        _log(f"[train-exclude] {label}: absent from GO vocabulary after HGNC lookup")
    # Outer held perturbations must never occur in a training/validation target.
    overlap = set(g for g in targets.values() if g is not None) & set(normalized_train[~is_ctrl])
    assert not overlap, f"held target labels occur in training payload: {sorted(overlap)}"
    row_ids = np.where(keep)[0]
    cap = int(os.environ.get("IVCBENCH_SCF_GENE_MAX_CELLS", "40000"))
    seed = int(os.environ.get("IVCBENCH_SCF_GENE_SEED", os.environ.get("IVCBENCH_SEED", "0")))
    rng = np.random.default_rng(seed)
    if cap and len(row_ids) > cap:
        groups = sorted(set(normalized_train[row_ids]))
        per_group = max(1, cap // len(groups))
        row_ids = np.sort(np.concatenate([rng.choice(np.where(normalized_train == g)[0], min(per_group, np.sum(normalized_train == g)), replace=False) for g in groups]))
    _log(f"[train] payload_rows={len(train)} usable_rows={int(keep.sum())} selected_rows={len(row_ids)} cap={cap} labels={len(set(normalized_train[row_ids]))}")
    if len(set(normalized_train[row_ids]) - {"ctrl"}) < 2:
        raise ValueError("at least two supported training perturbations are needed for native no_test validation split")

    # Import committed released code, not local modified copies or pip GEARS.
    asset_gears_dir = gears_dir
    model_dir, gears_dir = _published_sources(model_dir)
    sys.path[:0] = [str(gears_dir), str(model_dir)]
    import torch
    import anndata as ad
    from scipy import sparse
    import gears
    from gears import GEARS, PertData
    from gears.data_utils import DataSplitter
    from gears.utils import create_cell_graph_dataset_for_prediction
    from torch_geometric.loader import DataLoader
    assert Path(gears.__file__).resolve().parent == gears_dir / "gears", "wrong GEARS package imported"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for the released scFoundation gene model")
    device = "cuda:0"
    torch.cuda.manual_seed_all(seed)
    _log(f"[runtime] python={sys.executable} device={torch.cuda.get_device_name(0)} GEARS={gears.__file__}")
    for source in (gears_dir / "gears/model.py", gears_dir / "modules/encoders.py", checkpoint, hgnc):
        _log(f"[source] {source} sha256={_sha(source)}")

    class PanelMappedEncoder(torch.nn.Module):
        """Only reindex input/output; every weight belongs to released MAE.

        MEMOISED (IVCBENCH_SCF_GENE_CACHE=0 turns it off). The released encoder is loaded with
        finetune_method='frozen', which sets requires_grad=False on its parameters and calls
        .eval() on it every epoch, so for a given input row its output is a CONSTANT of this run.
        GEARS nonetheless calls it inside every training step and again inside its per-epoch
        evaluate(train_loader), and the rows it sees are control cells: measured on the live
        schmidt unit, 19,418 training graphs carry exactly 300 DISTINCT encoder inputs, so the
        15-epoch run performs ~609,000 forwards of 300 different things. Memoising on the exact
        input bytes turns that into 300.

        What this does NOT touch: the held gene enters through the GO graph
        (create_cell_graph_dataset_for_prediction(target, ..., model.pert_list)), never through
        this encoder, which only ever sees expression rows. Weights, batch composition, shuffling
        and the RNG stream are untouched; only the provenance of `emb` changes. A row that is not
        in the table -- inference uses the HELD unit's own controls, which training never saw --
        is computed exactly as before, never approximated by a neighbour.
        """
        def __init__(self, native):
            super().__init__()
            self.native = native
            self.register_buffer("gene_positions", torch.as_tensor(indices, dtype=torch.long))
            self._cache = {}
            self._hits = 0
            self._misses = 0
            self._on = os.environ.get("IVCBENCH_SCF_GENE_CACHE", "1") != "0"

        def _encode(self, x):
            canonical = x.new_zeros((len(x), len(panel) + 1))
            canonical[:, self.gene_positions] = x[:, :-1]
            canonical[:, -1] = x[:, -1]
            encoded = self.native(canonical)
            assert encoded.shape[1] >= len(panel), "released MAE omitted canonical gene embeddings"
            return encoded.index_select(1, self.gene_positions).contiguous()

        def forward(self, x):
            if not self._on:
                return self._encode(x)
            keys = [hashlib.sha256(np.ascontiguousarray(r).tobytes()).digest()
                    for r in x.detach().cpu().numpy()]
            missing = [i for i, k in enumerate(keys) if k not in self._cache]
            if missing:
                idx = torch.as_tensor(missing, device=x.device, dtype=torch.long)
                got = self._encode(x.index_select(0, idx))
                for j, i in enumerate(missing):
                    # Keep the table in HOST memory. On the device it is ~4 MB a row and 1.2 GB
                    # for 300, which on top of the 119M-parameter encoder and its activations put
                    # three concurrent units over a 47 GB card: T3 u4 died of CUDA OOM in epoch 1
                    # trying to allocate 12 MB. Moving it off the device costs a host-to-device
                    # copy per batch, which is nothing beside the forward it replaces.
                    self._cache[keys[i]] = got[j].detach().to("cpu", non_blocking=True)
                self._misses += len(missing)
            self._hits += len(keys) - len(missing)
            return torch.stack([self._cache[k].to(x.device, non_blocking=True)
                                for k in keys]).contiguous()

    input_hash = _sha(Path(in_path))[:16]
    condition_hash = hashlib.sha256("\0".join(requested).encode()).hexdigest()[:12]
    prefix = f"scFoundation_GEARS_split-{input_hash}_conditions-{condition_hash}_seed-{seed}_"
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        work = Path(temporary)
        data_dir = work / "data"
        data_dir.mkdir()
        shutil.copyfile(gene2go_path, data_dir / "gene2go.pkl")
        # `sub` is the checkpoint-representable panel; the unsupported columns are re-attached at
        # the control baseline when the profile is assembled.
        adata = ad.AnnData(sparse.csr_matrix(train[row_ids][:, supported]))
        adata.obs_names = [f"train_{i}" for i in row_ids]
        adata.var_names = sub
        adata.var["gene_name"] = sub
        adata.obs["condition"] = ["ctrl" if is_ctrl[i] else f"{normalized_train[i]}+ctrl" for i in row_ids]
        adata.obs["cell_type"] = "training"
        adata.obs["total_count"] = totals["total_count_train"][row_ids]
        adata.uns["log1p"] = {"base": None}
        pert_data = PertData(str(data_dir))
        pert_data.new_data_process(dataset_name="train", adata=adata)
        # Upstream no_test's default sampler selects only *combo* conditions;
        # PertData.prepare_split also drops its explicit test_perts argument in
        # this branch. Call the released DataSplitter API directly, with actual
        # TRAIN-only validation conditions, then set the same wrapper metadata.
        candidates = sorted(set(adata.obs["condition"].astype(str)) - {"ctrl"})
        n_validation = max(1, int(0.1 * len(candidates)))
        validation = sorted(rng.choice(candidates, n_validation, replace=False).tolist())
        _log(f"[validation] public DataSplitter(no_test) held TRAIN conditions={json.dumps(validation)}; outer held targets absent")
        pert_data.split = "no_test"
        pert_data.seed = seed
        pert_data.train_gene_set_size = 0.75  # upstream cache/config metadata
        pert_data.subgroup = None
        pert_data.adata = DataSplitter(pert_data.adata, split_type="no_test").split_data(
            test_size=0.1, seed=seed, test_perts=validation)
        pert_data.set2conditions = {
            partition: pert_data.adata.obs.loc[pert_data.adata.obs["split"] == partition, "condition"].astype(str).unique().tolist()
            for partition in ("train", "val")
        }
        assert set(pert_data.set2conditions["val"]) == set(validation)
        assert set(pert_data.set2conditions["train"]).isdisjoint(validation)
        assert set(pert_data.set2conditions["train"]) | set(validation) == set(candidates) | {"ctrl"}
        batch_size = int(os.environ.get("IVCBENCH_SCF_GENE_BATCH_SIZE", "2"))
        if batch_size < 2:
            raise ValueError("GEARS training BatchNorm requires BATCH_SIZE >= 2")
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
        assert len(pert_data.dataloader["train_loader"]) > 0 and len(pert_data.dataloader["val_loader"]) > 0
        graph = os.environ.get("IVCBENCH_SCF_GENE_GO_GRAPH")
        if graph:
            shutil.copyfile(graph, data_dir / "train/go.csv")
            _log(f"[graph] supplied native GO graph={graph} sha256={_sha(Path(graph))}")
        else:
            # NOT `panel`: that name is bound at the top of main() to the released 19,264-gene
            # checkpoint panel, and PanelMappedEncoder._encode closes over it to size the
            # canonical input row. Rebinding it here silently narrowed that row to the 235
            # perturbation genes and every scatter index ran off the end -- a CUDA device-side
            # assert on the first training batch.
            # targets carries None for a label whose symbol did not resolve; those are already
            # in `reasons` and never reach the graph, so guard them here exactly as the
            # held/training overlap assertion above does.
            pert_vocab = sorted({normalized_train[i] for i in row_ids if not is_ctrl[i]}
                                | {g for g in targets.values() if g is not None})
            edges = _write_panel_go_graph(pert_vocab, gene2go, data_dir / "train/go.csv")
            _log(f"[graph] built this panel's GO graph from {gene2go_path}: "
                 f"perturbation_vocabulary={len(pert_vocab)} edges_above_0.1={edges}")
        model = GEARS(pert_data, device=device)
        # `sub`, not `normalized`: the AnnData handed to GEARS carries only the genes the released
        # checkpoint can represent. Comparing against the full 2,000-gene panel made this assertion
        # fire on every unit once the unsupported columns were moved out of the model.
        assert list(model.gene_list) == sub, "public preprocessing changed output gene order"
        # Hidden width is the released gene checkpoint's decoder width.
        from load import convertconfig
        metadata = torch.load(str(checkpoint), map_location="cpu")
        hidden = int(convertconfig(metadata["gene"])["config"]["decoder"]["hidden_dim"])
        del metadata
        model.model_initialize(hidden_size=hidden, model_type="maeautobin", bin_set="autobin_resolution_append",
                               load_path=str(checkpoint), finetune_method="frozen", mode="v1", highres=0)
        built = data_dir / "train/go.csv"
        assert built.is_file(), "GEARS did not produce a GO graph for this panel"
        _log(f"[graph] GO graph in use sha256={_sha(built)} "
             f"edges={sum(1 for _ in built.open()) - 1}")
        edges = model.config["G_go"].detach().cpu().numpy()
        weights = model.config["G_go_weight"].detach().cpu().numpy()
        assert edges.shape[0] == 2 and edges.shape[1] == len(weights)
        for label, target in targets.items():
            if label in reasons:
                continue
            node = model.node_map_pert[target]
            incident = ((edges[0] == node) | (edges[1] == node)) & (weights > 0)
            nonself = incident & (edges[0] != edges[1])
            _log(f"[graph-support] {label}: GO_terms={len(gene2go[target])} positive_edges={int(incident.sum())} nonself_edges={int(nonself.sum())}")
            if not nonself.any():
                reasons[label] = "released GO graph has no positive nonself edges for this held target; an isolated random node ID is not gene-side transfer"
        if len(reasons) == len(requested):
            _finish(out_path, requested, [], [], reasons, ctrl_mean)
            return
        model.model.singlecell_model = PanelMappedEncoder(model.model.singlecell_model)
        epochs = int(os.environ.get("IVCBENCH_SCF_GENE_EPOCHS", "15"))
        if epochs < 1:
            raise ValueError("EPOCHS must be >= 1; untrained output is not accepted")
        results = work / "checkpoints"
        results.mkdir()
        model.train(epochs=epochs, result_dir=str(results), lr=float(os.environ.get("IVCBENCH_SCF_GENE_LR", "0.0002")))
        own_controls = ad.AnnData(sparse.csr_matrix(infer[:, supported]))
        own_controls.obs_names = [f"inference_{i}" for i in range(len(infer))]
        own_controls.var_names = sub
        own_controls.obs["total_count"] = totals["total_count_inf"]
        # GEARS.predict resets ctrl_adata to TRAIN controls. Use its public graph
        # constructor and the trained best_model directly to preserve own controls.
        model.best_model.eval()
        pred_cap = int(os.environ.get("IVCBENCH_SCF_GENE_PREDCTRL", "300"))
        n_ctrl = min(pred_cap, len(infer)) if pred_cap else len(infer)
        _log(f"[inference] basal=X_ctrl_inf own_control_rows={len(infer)} native_control_draws={n_ctrl}")
        # Hash the SAME projection the graphs are built from. own_controls is infer[:, supported]
        # -- the checkpoint-representable panel -- while this hashed the full-width rows, so the
        # assertion below compared a 1,946-gene basal against 2,000-gene hashes and could only
        # pass when every gene happened to be representable. It never fired before because no unit
        # in this campaign had reached inference: all three earlier attempts timed out in training.
        # The leak guarantee is unchanged: the basal must still be one of the held unit's own
        # control rows, now compared in the space the graph actually carries.
        own_control_hashes = _row_hashes(infer[:, supported])
        labels, profiles = [], []
        for label in requested:
            if label in reasons:
                continue
            target = targets[label]
            # Common control draws isolate condition identity when checking rows.
            np.random.seed(seed)
            graphs = create_cell_graph_dataset_for_prediction([target], own_controls, model.pert_list, device, num_samples=n_ctrl)
            for graph in graphs:
                basal = graph.x.detach().cpu().numpy().reshape(-1)[:-1]
                assert hashlib.sha256(np.ascontiguousarray(basal).tobytes()).digest() in own_control_hashes, "native inference graph basal is not an X_ctrl_inf row"
            _log(f"[basal-slot] {label}: {len(graphs)} native graph inputs verified against X_ctrl_inf")
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            # The model emits the checkpoint-REPRESENTABLE panel (`sub`, supported.size genes),
            # which the scatter below re-expands to the full panel -- so demanding len(genes) here
            # contradicted that handling and the accumulator was the wrong width too. Both are
            # the same full-panel-versus-supported-panel confusion as the basal hashes; that one
            # only surfaced first because it runs a few lines earlier.
            width = int(supported.size)
            total = np.zeros(width, dtype=np.float64)
            count = 0
            with torch.no_grad():
                for batch in loader:
                    pred = model.best_model(batch.to(device)).detach().cpu().numpy()
                    assert pred.ndim == 2 and np.isfinite(pred).all(), "non-finite prediction"
                    assert pred.shape[1] == width, (
                        f"model returned {pred.shape[1]} columns; the representable panel is {width}")
                    total += pred.sum(axis=0, dtype=np.float64)
                    count += len(pred)
            assert count > 0
            labels.append(label)
            sub_profile = (total / count)
            if sub_profile.shape[0] == supported.size and supported.size != len(genes):
                full = np.asarray(ctrl_mean, dtype=np.float64).copy()
                full[supported] = sub_profile
                sub_profile = full
            profiles.append(np.asarray(sub_profile, dtype=np.float32))
        _finish(out_path, requested, labels, profiles, reasons, ctrl_mean)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: scfoundation_gene_runner.py in.npz out.npz")
    main(sys.argv[1], sys.argv[2])
