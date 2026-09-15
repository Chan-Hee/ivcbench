#!/usr/bin/env python
"""SE-600M input construction for the MAP runner (`ivc-map` env).

MAP feeds the frozen Arc SE-600M encoder a "cell sentence": a padded list of gene tokens ordered by
expression, plus per-token expression weights.  The released repository builds these only for
Tahoe-100M -- preprocess/data/ds_tahoe_se.py::CellDatasetCollatorSEOnly.sample_cell_sentences_fast,
driven by preprocess/G_extract_se_input_for_lora.py -- and it reads a `gene_vocab_hgnc.json` that
maps Tahoe's local gene index to an HGNC symbol; neither the OP3 branch nor that json is shipped.

This module reproduces that routine exactly, but starting from HGNC gene symbols (which is what a
CellSet carries in `var_names`), so the same tokenisation is available for OP3:

  * a gene is valid iff it is a key of the ESM2 gene-embedding dict the SE config points at, and
    its token id is that key's position in the dict -- the same `global_pos` the released
    `_load_gene_mapping` builds;
  * counts are log1p'd iff they look like raw integer counts (the released heuristic);
  * weights = counts / counts.sum(), scaled by 100, exactly as released;
  * genes are ranked by expression after a random tie-shuffle, truncated to pad_length - 1, and
    position 0 carries cfg.dataset.cls_token_idx.

Any deviation from the released routine would change what the frozen encoder sees, so this file is
deliberately a transcription rather than a reimplementation.
"""
from __future__ import annotations

import numpy as np
import torch

RAW_COUNT_HEURISTIC_THRESHOLD = 20.0  # released constant (ds_tahoe_se.py)
EXPONENTIATED_UMIS_LIMIT = 1e6  # released constant


def _is_raw_integer_counts(counts: torch.Tensor) -> bool:
    if float(torch.max(counts)) > RAW_COUNT_HEURISTIC_THRESHOLD:
        return True
    return int(torch.expm1(counts).sum()) > EXPONENTIATED_UMIS_LIMIT


def gene_token_ids(genes: list[str], cfg) -> np.ndarray:
    """token id per benchmark gene, -1 where the gene has no ESM2 embedding."""
    from data.utils import get_embedding_cfg  # MAP's own accessor

    tok = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    keys = list(tok.keys())
    pos = {g: i for i, g in enumerate(keys)}
    return np.asarray([pos.get(g, -1) for g in genes], dtype=np.int64)


def se_tokens_from_expression(X: np.ndarray, genes: list[str], cfg, seed: int = 0):
    """(n_cells, n_genes) expression -> (gene_ids[int32], expr[float32]) of width pad_length."""
    pad_length = int(cfg.dataset.pad_length)
    cls_idx = int(cfg.dataset.cls_token_idx)
    ids = gene_token_ids(genes, cfg)
    valid = np.where(ids >= 0)[0]
    if valid.size == 0:
        raise RuntimeError(
            "MAP-SE: no benchmark gene has an ESM2 embedding; SE input undefined"
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    out_ids = np.zeros((n, pad_length), dtype=np.int32)
    out_expr = np.zeros((n, pad_length), dtype=np.float32)
    out_ids[:, 0] = cls_idx

    for r in range(n):
        counts = torch.as_tensor(X[r, valid], dtype=torch.float32)
        if _is_raw_integer_counts(counts):
            counts = torch.log1p(counts)
        s = float(counts.sum())
        w = (counts / s) if s > 0 else torch.softmax(counts, dim=0)
        shuffled = rng.permutation(valid.size)
        order = shuffled[np.argsort(-counts.numpy()[shuffled], kind="stable")]
        k = min(valid.size, pad_length - 1)
        sel = order[:k]
        out_ids[r, 1 : k + 1] = ids[valid[sel]].astype(np.int32)
        out_expr[r, 1 : k + 1] = 100.0 * w.numpy()[sel]
    return out_ids, out_expr
