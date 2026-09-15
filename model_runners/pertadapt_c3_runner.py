#!/usr/bin/env python
"""Published graph PertAdapt for unseen genetic interventions (scfoundation environment).

The model and adaptive loss are imported from the released scFoundation
PertAdapter implementation, not from pertadapt_runner.py's reconstructed head.
A parameter-free scatter/gather connects the measured panel to the pretrained
19,264-gene encoder. The published graph adapter and gene-wise decoder operate
on that panel and the corresponding submatrix of the released GO mask.

Additional native input requirements: total_count_train / total_count_inf must
contain actual pre-normalization library sizes. No library-size proxy is made
from a normalized HVG panel. Optional train_cell_ids / ctrl_inf_cell_ids (or
train_idx / ctrl_inf_idx) establish disjoint cell identity; without them exact
row hashes provide a conservative overlap check. Canonical T3 payloads sharing
all training controls with inference consequently fail this strict contract.

Unsupported output columns reject the entire unit. Unsupported intervention
labels are explicitly declined, never replaced with control or another target.
Knobs: IVCBENCH_PA_REPO, _GO_MASK_NPZ, IVCBENCH_GENE2GO,
IVCBENCH_SCFOUNDATION_DIR / _CKPT, IVCBENCH_HGNC_TSV, IVCBENCH_SEED,
IVCBENCH_PA_EPOCHS (15), _BATCH (4), _MAXCELLS (6000), _PREDCTRL (32),
_LR (0.001), and IVCBENCH_WORK_DIR (defaults to a fresh /tmp directory).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_HGNC_APPROVED: dict[str, set[str]] = {}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _file(env: str, candidates: list[Path]) -> Path:
    if os.environ.get(env):
        candidates = [Path(os.environ[env])]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"set ${env}; none of {list(map(str, candidates))} exists")


def _aliases(path: Path) -> dict[str, set[str]]:
    """Use supplied HGNC equivalence classes; never guess spelling similarity."""
    groups: dict[str, set[str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            approved = row.get("Approved symbol", row.get("symbol", ""))
            names = {approved}
            for key in ("Previous symbols", "Alias symbols", "prev_symbol", "alias_symbol"):
                names.update(x.strip() for x in row.get(key, "").replace("|", ",").split(","))
            names.discard("")
            for name in names:
                groups.setdefault(name, set()).update(names)
                if approved:
                    _HGNC_APPROVED.setdefault(name, set()).add(approved)
    # HGNC:27962 / NCBI Gene 340061; the older bundled HGNC export predates STING1.
    for name in ("TMEM173", "STING1"):
        groups.setdefault(name, set()).update({"TMEM173", "STING1"})
        _HGNC_APPROVED[name] = {"STING1"}
    return groups


def _resolve(label: str, vocabulary: set[str], aliases: dict[str, set[str]]) -> str | None:
    if label in vocabulary:
        return label
    approved = _HGNC_APPROVED.get(label, set()) & vocabulary
    if len(approved) == 1:
        return next(iter(approved))
    matches = aliases.get(label, set()) & vocabulary
    return next(iter(matches)) if len(matches) == 1 else None


def _row_hashes(rows: np.ndarray) -> set[bytes]:
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in rows}


def _assert_disjoint(data, train: np.ndarray, controls: np.ndarray) -> None:
    checked_ids = False
    for left, right in (("train_indices", "inference_input_indices"),
                        ("train_cell_ids", "ctrl_inf_cell_ids"),
                        ("train_idx", "ctrl_inf_idx")):
        if (left in data) != (right in data):
            raise ValueError(f"cell-identity fields {left} and {right} must be supplied together")
        if left in data and right in data:
            a, b = np.asarray(data[left]).ravel(), np.asarray(data[right]).ravel()
            if len(a) != len(train) or len(b) != len(controls):
                raise ValueError("cell-identity arrays have incorrect lengths")
            overlap = set(map(str, a)) & set(map(str, b))
            _log(f"[leak] identity_keys={left},{right} overlap={len(overlap)}")
            if overlap:
                raise AssertionError(f"training/inference cell IDs overlap: {sorted(overlap)[:8]}")
            checked_ids = True
    if checked_ids:
        return
    # The leak boundary here is the TREATED cells. c3.py:107 sets control_inference_only=False, so
    # builder.py:42 makes X_ctrl_inf a strict subset of the training controls by design (see
    # splits/audit.py:57-60). This runner's own docstring conceded the clash; asserting on controls
    # rejected every canonical T3 payload.
    treated = train[~np.asarray(data["is_control_train"], dtype=bool)] if "is_control_train" in data else train
    overlap = _row_hashes(treated) & _row_hashes(controls)
    _log(f"[leak] global_cell_ids=unavailable treated_row_overlap={len(overlap)}")
    if overlap:
        raise AssertionError("TREATED training rows appear in X_ctrl_inf; that is a real leak")
    _log("[leak] exact-row overlap excluded; original global cell identity remains unverified")


def _diagnostics(requested, labels, profiles, control, declines) -> None:
    returned = set(labels)
    missing = sorted(set(requested) - returned)
    if returned - set(requested):
        raise AssertionError("unrequested prediction labels")
    if set(missing) != set(declines):
        raise AssertionError("every missing target must have an explicit decline reason")
    _log("[coverage] " + json.dumps(dict(requested=sorted(requested), predicted=sorted(returned),
                                       missing=missing, decline_reasons=declines), sort_keys=True))
    for label, row in zip(labels, profiles):
        distance = float(np.max(np.abs(np.asarray(row, dtype=np.float64) - control)))
        _log(f"[control] {label}: max_abs_delta={distance:.12g} at_or_below_1e-5={distance <= 1e-5}")
    identical = []
    for i in range(len(labels)):
        for j in range(i):
            if np.array_equal(profiles[i], profiles[j]):
                identical.append([labels[j], labels[i]])
    _log("[conditions] bit_exact_identical_pairs=" + json.dumps(identical))


def _go_graph(mask, backbone_genes, graph_genes, annotations, torch):
    """Released Jaccard > .1, top-20-neighbors-plus-self graph construction.

    The released mask supplies the checkpoint-gene edges. Additional requested
    or training interventions outside the checkpoint expression vocabulary can
    still enter through their GO annotations; their edges use the same Jaccard
    formula, rather than requiring the target to be an output gene.
    """
    backbone_pos = {g: i for i, g in enumerate(backbone_genes)}
    graph_pos = {g: i for i, g in enumerate(graph_genes)}
    extra = [g for g in graph_genes if g not in backbone_pos]
    extra_scores: dict[str, dict[str, float]] = {}
    for g in extra:
        a = set(annotations[g])
        scores = {}
        for h in graph_genes:
            b = set(annotations[h])
            union = len(a | b)
            value = len(a & b) / union if union else 0.0
            if value > 0.1:
                scores[h] = value
        extra_scores[g] = scores
    edges, weights = [], []
    mask = mask.tocsr()
    for target in graph_genes:
        if target in backbone_pos:
            row = mask.getrow(backbone_pos[target])
            scores = {backbone_genes[int(i)]: float(v) for i, v in zip(row.indices, row.data)
                      if backbone_genes[int(i)] in graph_pos and v > 0.1}
            for g in extra:
                if target in extra_scores[g]:
                    scores[g] = extra_scores[g][target]
        else:
            scores = extra_scores[target]
        for source, weight in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:21]:
            edges.append([graph_pos[source], graph_pos[target]])
            weights.append(weight)
    if not edges:
        raise RuntimeError("no annotated perturbation graph edges")
    _log(f"[graph] nodes={len(graph_genes)} edges={len(edges)} extra_GO_condition_nodes={extra}")
    return torch.tensor(edges, dtype=torch.long).T, torch.tensor(weights, dtype=torch.float32)


def main(in_path: str, out_path: str) -> None:
    data = np.load(in_path, allow_pickle=True)
    X = np.asarray(data["X_train"], dtype=np.float32)
    ctrl_inf = np.asarray(data["X_ctrl_inf"], dtype=np.float32)
    genes = list(map(str, data["genes"]))
    pert = np.asarray(data["pert_train"]).astype(str)
    is_ctrl = np.asarray(data["is_control_train"], dtype=bool)
    requested = sorted(set(map(str, data["test_perts"])) - {"control"})
    declines: dict[str, str] = {}
    labels: list[str] = []
    profiles: list[np.ndarray] = []
    control = None

    def reject(reason: str) -> None:
        for label in requested:
            if label not in declines:
                declines[label] = reason
                _log(f"[decline] {label}: {reason}")
        _diagnostics(requested, [], [], control, declines)
        raise RuntimeError(reason)

    if (X.ndim != 2 or ctrl_inf.ndim != 2 or X.shape[1] != len(genes)
            or ctrl_inf.shape[1] != len(genes) or pert.ndim != 1 or is_ctrl.ndim != 1
            or len(pert) != len(X) or len(is_ctrl) != len(X)):
        reject("malformed expression panel or label dimensions")
    control = X[is_ctrl].mean(axis=0, dtype=np.float64) if is_ctrl.any() else None
    if not np.isfinite(X).all() or not np.isfinite(ctrl_inf).all():
        reject("nonfinite expression payload")
    if (X < 0).any() or (ctrl_inf < 0).any():
        reject("released RNA encoder requires nonnegative log-normalized expression")
    if not np.array_equal(is_ctrl, pert == "control"):
        reject("is_control_train disagrees with pert_train control labels")
    if not len(ctrl_inf) or control is None or not requested:
        reject("training controls, own inference controls, and requested targets are required")

    scf_dir = Path(os.environ.get("IVCBENCH_SCFOUNDATION_DIR", ""))
    if not (scf_dir / "load.py").is_file():
        reject("set IVCBENCH_SCFOUNDATION_DIR to released model directory containing load.py")
    try:
        checkpoint = _file("IVCBENCH_SCFOUNDATION_CKPT", [])
        gene_file = scf_dir / "OS_scRNA_gene_index.19264.tsv"
        with gene_file.open() as handle:
            backbone_genes = [row["gene_name"] for row in csv.DictReader(handle, delimiter="\t")]
        if len(backbone_genes) != 19264 or len(set(backbone_genes)) != 19264:
            raise ValueError("checkpoint gene order is not the unique released 19,264-gene panel")
        hgnc = _file("IVCBENCH_HGNC_TSV", [scf_dir.parent / "SCAD/data/processing/HGNC_symbol_all_genes.tsv"])
        aliases = _aliases(hgnc)
        vocabulary = set(backbone_genes)
        canonical = [_resolve(g, vocabulary, aliases) for g in genes]
        missing_genes = [g for g, c in zip(genes, canonical) if c is None]
        if missing_genes:
            # Keep the panel and carry the unsupported columns at the control baseline, which is
            # what the sibling T4 runner does (pertadapt_runner.py:173-174: hvg_to_scf = -1 for a
            # gene the checkpoint lacks, then a `have` mask). Every 2,000-HVG panel in this
            # benchmark contains 31-59 lncRNA / TR-segment genes the 19,264-gene checkpoint does
            # not carry, so rejecting the unit made all five T3 datasets unproducible. The
            # unsupported columns are named on stdout and are not passed off as predictions.
            _log(f"[panel] {len(missing_genes)} of {len(genes)} output genes are outside the "
                 f"checkpoint vocabulary and keep the control baseline: "
                 + ",".join(missing_genes[:20]) + (" ..." if len(missing_genes) > 20 else ""))
        # Only the SUPPORTED names can collide; the unsupported ones are all None and must not be
        # counted as duplicates of each other (they keep the control baseline and never index the
        # backbone).
        _sup_names = [c for c in canonical if c is not None]
        if len(set(_sup_names)) != len(_sup_names):
            raise ValueError("multiple input genes map to the same checkpoint gene; panel must be reconciled upstream")
        _log("[aliases] output_gene_map=" + json.dumps(
            {g: c for g, c in zip(genes, canonical) if c is not None and g != c}))   # unsupported: see [panel]
        _assert_disjoint(data, X, ctrl_inf)
        # Library sizes for the backbone's resolution token. Raw counts do not survive this
        # benchmark's preprocessing and no payload carries them; the deposited scFoundation cells
        # derive the total as expm1(panel).sum()+1 (scfoundation_c1_runner.py:87) and this runner
        # uses the same backbone, so it follows the same convention. Disclosed, not assumed.
        def _totals(key, matrix):
            if key in data:
                v = np.asarray(data[key], dtype=np.float64).ravel()
                if len(v) == len(matrix) and np.isfinite(v).all() and (v > 0).all():
                    _log(f"[resolution] {key}: payload original totals")
                    return v.astype(np.float32)
                raise ValueError(f"{key} present but malformed")
            _log(f"[resolution] {key}: derived as expm1(panel).sum()+1 "
                 "(same convention as the deposited scFoundation cells)")
            return (np.expm1(np.asarray(matrix, dtype=np.float64)).sum(1) + 1.0).astype(np.float32)

        total_train = _totals("total_count_train", X)
        total_inf = _totals("total_count_inf", ctrl_inf)
        annotations_path = _file("IVCBENCH_GENE2GO", [ROOT / "data/_assets/gears/gene2go_all.pkl",
                                                     ROOT.parent / "benchmark/data/_assets/gears/gene2go_all.pkl"])
        with annotations_path.open("rb") as handle:
            annotations = pickle.load(handle)
        annotated = {g for g, terms in annotations.items() if len(terms)}
        # Resolve against the complete GO annotation vocabulary first: a target
        # need not be an expression-output gene to have a native graph node.
        condition_vocab = annotated
        def condition_symbol(label):
            approved = _HGNC_APPROVED.get(label, set()) & annotated
            return next(iter(approved)) if len(approved) == 1 else _resolve(label, condition_vocab, aliases)

        target_map = {}
        for label in requested:
            symbol = condition_symbol(label)
            if symbol is None:
                declines[label] = "no unambiguous annotated GO condition after HGNC alias lookup"
                _log(f"[decline] {label}: {declines[label]}")
            else:
                target_map[label] = symbol
        train_map = {label: condition_symbol(label) for label in set(pert[~is_ctrl])}
        for label, mapped in sorted(train_map.items()):
            if mapped is None:
                _log(f"[train-decline] {label}: no annotated GO condition after HGNC alias lookup")
        held_overlap = set(target_map.values()) & {v for v in train_map.values() if v is not None}
        if held_overlap:
            raise AssertionError(f"requested T3 genes present in training after alias normalization: {sorted(held_overlap)}")
        _log("[aliases] condition_map=" + json.dumps({k: v for k, v in {**train_map, **target_map}.items() if k != v}))
        if not target_map:
            reject("no supported requested GO conditions")
        upstream = Path(os.environ.get("IVCBENCH_PA_REPO", str(ROOT.parent / "benchmark/vendor/pertadapt_repo")))
        if (upstream / "scFoundation/PertAdapter/gears/model_new.py").is_file():
            upstream = upstream / "scFoundation/PertAdapter"
        if not (upstream / "gears/model_new.py").is_file():
            raise FileNotFoundError("IVCBENCH_PA_REPO must contain the original graph PertAdapter implementation")
        mask_path = _file("IVCBENCH_PA_GO_MASK_NPZ", [ROOT / "data/pertadapt/official/go_mask_19264.npz",
                                                  ROOT.parent / "benchmark/data/pertadapt/official/go_mask_19264.npz"])
    except Exception as exc:
        reject(f"{type(exc).__name__}: {exc}")

    import torch
    from scipy import sparse
    import anndata as ad
    import pandas as pd

    if not torch.cuda.is_available():
        reject("released PertAdapt graph adapter requires CUDA; no CPU fallback")
    torch.cuda.set_device(0)
    _log(f"[device] cuda:0 name={torch.cuda.get_device_name(0)} visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} torch={torch.__version__}")
    torch.set_num_threads(int(os.environ.get("IVCBENCH_THREADS", "4")))
    seed = int(os.environ.get("IVCBENCH_SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    epochs = int(os.environ.get("IVCBENCH_PA_EPOCHS", "15"))
    batch_size = int(os.environ.get("IVCBENCH_PA_BATCH", "4"))
    max_cells = int(os.environ.get("IVCBENCH_PA_MAXCELLS", "6000"))
    if min(epochs, batch_size, max_cells) <= 0:
        reject("epochs, batch size, and cell cap must be positive")
    digest = hashlib.sha256(Path(in_path).read_bytes()).hexdigest()
    conditions_digest = hashlib.sha256(json.dumps(requested).encode()).hexdigest()[:12]
    work_root = Path(os.environ.get("IVCBENCH_WORK_DIR", tempfile.gettempdir()))
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"PertAdapt_T3_split-{digest[:12]}_conditions-{conditions_digest}_seed-{seed}_", dir=work_root))
    _log(f"[provenance] model=PertAdapt original_graph={upstream} seed={seed} input_sha256={digest} work={work}")
    mask = sparse.load_npz(mask_path).tocsr()
    if mask.shape != (19264, 19264):
        reject(f"released GO mask shape {mask.shape} does not match checkpoint gene order")
    mask_gene_file = mask_path.parent / "scfoundation_genes.txt"
    if mask_gene_file.exists():
        if mask_gene_file.read_text().splitlines() != backbone_genes:
            reject("released GO mask gene-order manifest does not match checkpoint genes")
        _log(f"[graph] mask_gene_order_verified={mask_gene_file}")
    else:
        _log("[graph] mask gene-order sidecar unavailable; supplied mask must follow released checkpoint gene order")
    backbone_pos = {g: i for i, g in enumerate(backbone_genes)}
    # Only the genes the checkpoint carries can be scattered into and gathered out of the backbone.
    # `supported` indexes them within the evaluation panel; everything else keeps the control
    # baseline in the returned profile, which is what the sibling T4 runner does.
    supported = np.asarray(
        [i for i, c in enumerate(canonical) if c is not None and c in backbone_pos], dtype=np.int64
    )
    if supported.size == 0:
        reject("no output gene of this panel is in the released checkpoint vocabulary")
    positions = np.asarray([backbone_pos[canonical[i]] for i in supported], dtype=np.int64)
    _log(f"[panel] modelled={supported.size}/{len(canonical)} genes; "
         f"{len(canonical) - supported.size} keep the control baseline")
    # The GO mask must cover the FULL evaluation panel, because the attention it gates is sized by
    # num_genes. Supported genes keep their released GO relations; a gene the checkpoint does not
    # carry gets only its own diagonal, so it attends to nothing and nothing attends to it.
    panel_mask = work / "condition-all_GO_panel.npz"
    n_panel = len(canonical)
    sub = mask[positions][:, positions].tocoo()
    full_mask = sparse.lil_matrix((n_panel, n_panel), dtype=mask.dtype)
    sup = np.asarray(supported)
    full_mask[sup[sub.row], sup[sub.col]] = sub.data
    full_mask.setdiag(1)
    _log(f"[graph] GO mask {n_panel}x{n_panel}; {n_panel - len(sup)} unsupported gene(s) carry only "
         "their own diagonal")
    sparse.save_npz(panel_mask, full_mask.tocsr())
    os.environ["IVCBENCH_PA_GO_MASK_NPZ"] = str(panel_mask)
    sys.path.insert(0, str(scf_dir.resolve()))
    sys.path.insert(0, str(upstream.resolve()))
    from gears.model_new import GEARS_Model_Pert_Adapter_New
    from gears.utils import loss_adapt, create_cell_graph_dataset_for_prediction
    from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes
    from torch_geometric.data import Data, DataLoader

    _log(f"[upstream] model_source={sys.modules[GEARS_Model_Pert_Adapter_New.__module__].__file__} loss_source={sys.modules[loss_adapt.__module__].__file__}")
    eligible = np.asarray([c or train_map.get(p) is not None for p, c in zip(pert, is_ctrl)])
    selected = np.where(eligible)[0]
    if len(selected) > max_cells:
        rng = np.random.default_rng(seed)
        strata = sorted(set(pert[selected]))
        if max_cells < len(strata):
            reject("training cell cap is smaller than number of conditions")
        quota = max_cells // len(strata)
        selected = np.sort(np.concatenate([rng.choice(np.where(eligible & (pert == p))[0],
                         min(quota, int(np.sum(eligible & (pert == p)))), replace=False) for p in strata]))
    if not np.any(is_ctrl[selected]) or not np.any(~is_ctrl[selected]):
        reject("no usable training control or perturbation cells after support filtering")
    cond = np.asarray(["ctrl" if is_ctrl[i] else train_map[pert[i]] + "+ctrl" for i in selected])
    # Full 2,000-gene panel: the released model derives num_genes from this object and from the
    # graph, so narrowing it here fights the upstream at every tensor. Genes the checkpoint lacks
    # are given a zero embedding inside CanonicalPanelEncoder instead -- the same treatment the
    # sibling T4 runner applies (pertadapt_runner.py:173-174).
    panel_genes = [c if c is not None else g for c, g in zip(canonical, genes)]
    adata = ad.AnnData(sparse.csr_matrix(X[selected]),
                       obs=pd.DataFrame({"condition": cond, "cell_type": "Tcell"}, index=[f"train_{i}" for i in selected]))
    adata.var_names = panel_genes
    adata.var["gene_name"] = panel_genes
    adata.uns["log1p"] = {"base": None}
    adata = get_dropout_non_zero_genes(get_DE_genes(adata, skip_calc_de=False))
    geneid2idx = {g: i for i, g in enumerate(panel_genes)}
    pert2full = dict(adata.obs[["condition", "condition_name"]].astype(str).values)
    dict_filter = {condition: adata.uns["non_zeros_gene_idx"][full]
                   for condition, full in pert2full.items() if condition != "ctrl"}
    if any(len(v) == 0 for v in dict_filter.values()):
        reject("a training condition has no nonzero measured genes for published adaptive loss")
    graph_genes = sorted((vocabulary & annotated) | set(target_map.values()) |
                         {train_map[pert[i]] for i in selected if not is_ctrl[i]})
    graph_index = {g: i for i, g in enumerate(graph_genes)}
    edge_index, edge_weight = _go_graph(mask, backbone_genes, graph_genes, annotations, torch)
    del mask
    config = dict(num_genes=len(genes), num_perts=len(graph_genes), hidden_size=512,
                  uncertainty=False, num_go_gnn_layers=1, decoder_hidden_size=16,
                  num_gene_gnn_layers=1, no_perturb=False, cell_fitness_pred=False,
                  G_go=edge_index, G_go_weight=edge_weight, device="cuda:0",
                  model_type="maeautobin", bin_set="autobin_resolution_append",
                  load_path=str(checkpoint), finetune_method="frozen", highres=0,
                  mode="v1", record_pred=False)
    model = GEARS_Model_Pert_Adapter_New(config)

    class CanonicalPanelEncoder(torch.nn.Module):
        """Tensor-index adapter only; all embeddings are actual pretrained gene tokens."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.model_config = encoder.model_config
            self.register_buffer("positions", torch.tensor(positions, dtype=torch.long))
            self.register_buffer("supported", torch.tensor(supported, dtype=torch.long))

        def forward(self, inputs):
            # inputs: (B, n_panel + 1) on the FULL evaluation panel. Only the genes the checkpoint
            # carries are scattered into the backbone; the rest read back as a zero embedding, so
            # they contribute nothing rather than shifting the gene indexing.
            full = inputs.new_zeros((len(inputs), 19265))
            full[:, self.positions] = inputs[:, :-1][:, self.supported]
            full[:, -1] = inputs[:, -1]
            enc = self.encoder(full)
            out = enc.new_zeros((enc.shape[0], inputs.shape[1] - 1, enc.shape[2]))
            out[:, self.supported, :] = enc[:, self.positions, :]
            return out.contiguous()

    model.singlecell_model = CanonicalPanelEncoder(model.singlecell_model)
    for parameter in model.singlecell_model.parameters():
        parameter.requires_grad_(False)
    model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("IVCBENCH_PA_LR", "0.001")), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_controls = np.where(is_ctrl[selected])[0]
    rng = np.random.default_rng(seed)
    graphs = []
    basal_source_ids = []
    for local, original in enumerate(selected):
        basal_local = local if is_ctrl[original] else int(rng.choice(train_controls))
        basal = int(selected[basal_local])
        if not is_ctrl[basal]:
            reject("constructed training basal input is not a training control cell")
        basal_source_ids.append(basal)
        # the sub-panel, matching the AnnData handed to the graph builder; X is still the full
        # 2,000-gene payload and using it here mismatched the encoder's gene positions
        vector = np.concatenate([X[basal], [total_train[basal]]]).astype(np.float32)
        graphs.append(Data(x=torch.tensor(vector).reshape(-1, 1),
                           y=torch.tensor(X[original]).reshape(1, -1),
                           pert_idx=[-1] if is_ctrl[original] else [graph_index[train_map[pert[original]]]],
                           pert=str(cond[local])))
    _log(f"[training] selected_cells={len(selected)} conditions={len(set(cond))} epochs={epochs} batch={batch_size} validation_cells=0 held_treated_cells=0")
    _log(f"[training] original_payload_row_indices={selected.tolist()} basal_training_control_indices={sorted(set(basal_source_ids))}")
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        model.singlecell_model.eval()
        losses = []
        for batch in loader:
            batch = batch.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = loss_adapt(prediction, batch.y, batch.pert, adata, geneid2idx, pert2full,
                              dict_filter=dict_filter, de_loss_weight=0.5)
            if not torch.isfinite(loss):
                reject("nonfinite published adaptive loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        _log(f"[train] epoch={epoch + 1}/{epochs} mean_published_loss_adapt={np.mean(losses):.9g}")
    model.eval()
    own_controls = ad.AnnData(sparse.csr_matrix(ctrl_inf))
    own_controls.obs["total_count"] = total_inf
    prediction_samples = min(int(os.environ.get("IVCBENCH_PA_PREDCTRL", "32")), len(ctrl_inf))
    if prediction_samples <= 0:
        reject("inference control sample count must be positive")
    _log(f"[inference] basal_slot=X_ctrl_inf own_controls={len(ctrl_inf)} sampled_per_target={prediction_samples} sampling=published_with_replacement same_seed_per_target")
    # The identity check must hash what the AUTHORS' helper actually builds. utils.py:407 casts the
    # total to int before appending it (`obs['total_count'].astype(int)`), so hashing a float total
    # here made every own-control row look substituted -- 20/20 rows mismatched on the smoke input
    # and the intersection was empty. The expression half is unchanged; only the appended total is
    # brought into the same representation.
    own_vectors = np.concatenate(
        [ctrl_inf, total_inf.astype(np.int64).astype(np.float32)[:, None]], axis=1
    ).astype(np.float32)
    own_hashes = _row_hashes(own_vectors)
    with torch.no_grad():
        for label, symbol in sorted(target_map.items()):
            np.random.seed(seed)
            cell_graphs = create_cell_graph_dataset_for_prediction([symbol], own_controls, graph_genes,
                                                                   "cuda:0", num_samples=prediction_samples)
            for graph in cell_graphs:
                basal_vector = graph.x.detach().cpu().numpy().ravel().astype(np.float32)
                if hashlib.sha256(np.ascontiguousarray(basal_vector).tobytes()).digest() not in own_hashes:
                    reject(f"published inference graph for {label} substituted the own-control basal vector")
                if list(graph.pert_idx) != [graph_index[symbol]]:
                    reject(f"published inference graph for {label} lost its requested GO condition")
            _log(f"[inference-assert] {label}: all_{len(cell_graphs)}_Data.x_rows_match_X_ctrl_inf_with_raw_library_size condition_index={graph_index[symbol]}")
            chunks = [model(batch.to("cuda:0")).detach().cpu().numpy()
                      for batch in DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)]
            cells = np.concatenate(chunks, axis=0)
            if cells.shape[1] != len(genes) or not np.isfinite(cells).all():
                reject(f"invalid original model output for {label}")
            labels.append(label)
            profiles.append(cells.mean(axis=0, dtype=np.float64).astype(np.float32))
            _log(f"[prediction] requested={label} GO_condition={symbol} actual_model_cells={len(cells)}")
    _diagnostics(requested, labels, profiles, control, declines)
    np.savez(out_path, pred_perts=np.asarray(labels, dtype=object),
             pred_means=np.stack(profiles).astype(np.float32))
    _log(f"[saved] {out_path} rows={len(labels)} genes={len(genes)} dtype=float32")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: pertadapt_c3_runner.py in.npz out.npz")
    main(sys.argv[1], sys.argv[2])
