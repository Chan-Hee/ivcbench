"""Genes an evaluation panel carries that the released scFoundation checkpoint cannot represent.

Two runners on the unseen-gene and unseen-knockout cells read the released vocabulary
`OS_scRNA_gene_index.19264.tsv`: scfoundation_gene_runner and pertadapt_c3_runner. A gene in the
2,000-gene panel that is absent from that vocabulary, even after HGNC alias resolution, has no
input embedding and no output column in either model, so neither can predict it. Each runner
invented its own way to fill the hole -- scFoundation writes the control mean, PertAdapt ships
whatever a zero embedding decodes to -- and both printed the same sentence about it, which is how
the difference went unnoticed for so long.

NATIVE_102_FINAL_REVIEW.md section 247 rules on exactly those cells and says the policy of
filling unpredicted output genes with the control has to go; its allowed column offers the
remedy in its place, "a common evaluation mask over the genes actually output". This module is
that mask.

It is common in the strong sense. Both runners resolve against the same file, so "outside the
checkpoint vocabulary" is a property of the PANEL, not of a model: the same gene set is removed
for every model scored on the cell and for both floor members, which is what keeps the
comparison symmetric. Nothing here changes how any model trains or predicts -- the mask is a
metric argument, written into each bundle as `exclude_gene_idx`, so a reviewer rescoring a
deposited bundle reproduces the census number without knowing this file exists.

Resolution follows model_runners/pertadapt_c3_runner.py:54-84 exactly: an exact hit in the
vocabulary, else the unique HGNC approved symbol that is in the vocabulary, else the unique
alias-class member that is. `tests/test_panel_mask.py` pins the counts the runners themselves
logged for six real units.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

__all__ = ["unrepresentable", "vocabulary_path"]

_HGNC_DEFAULT = "SCAD/data/processing/HGNC_symbol_all_genes.tsv"
_VOCAB_NAME = "OS_scRNA_gene_index.19264.tsv"


def vocabulary_path() -> Path:
    root = os.environ.get("IVCBENCH_SCFOUNDATION_DIR")
    if not root:
        raise RuntimeError(
            "IVCBENCH_SCFOUNDATION_DIR is unset, so the panel mask cannot be built. "
            "runs/env.sh sets it; a run that reaches scoring without it would silently "
            "score genes no checkpoint-based model can predict."
        )
    path = Path(root) / _VOCAB_NAME
    if not path.is_file():
        raise FileNotFoundError(f"released vocabulary missing: {path}")
    return path


def _hgnc_path(vocab: Path) -> Path:
    env = os.environ.get("IVCBENCH_HGNC_TSV")
    path = Path(env) if env else vocab.parent.parent / _HGNC_DEFAULT
    if not path.is_file():
        raise FileNotFoundError(f"set $IVCBENCH_HGNC_TSV; {path} does not exist")
    return path


@lru_cache(maxsize=1)
def _tables() -> tuple[frozenset[str], dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """(vocabulary, alias classes, approved-symbol map), read once per process."""
    vocab_path = vocabulary_path()
    vocab = frozenset(
        line.split("\t")[0]
        for line in vocab_path.read_text().splitlines()[1:]
        if line.strip()
    )
    groups: dict[str, set[str]] = {}
    approved_for: dict[str, set[str]] = {}
    with _hgnc_path(vocab_path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            approved = row.get("Approved symbol", row.get("symbol", ""))
            names = {approved}
            for key in ("Previous symbols", "Alias symbols", "prev_symbol", "alias_symbol"):
                names.update(x.strip() for x in row.get(key, "").replace("|", ",").split(","))
            names.discard("")
            for name in names:
                groups.setdefault(name, set()).update(names)
                if approved:
                    approved_for.setdefault(name, set()).add(approved)
    # HGNC:27962 / NCBI Gene 340061. The bundled HGNC export predates the STING1 rename, and the
    # runners carry the same repair, so the mask must not disagree with them on this one symbol.
    for name in ("TMEM173", "STING1"):
        groups.setdefault(name, set()).update({"TMEM173", "STING1"})
        approved_for[name] = {"STING1"}
    return (
        vocab,
        {k: frozenset(v) for k, v in groups.items()},
        {k: frozenset(v) for k, v in approved_for.items()},
    )


def _resolves(label: str) -> bool:
    vocab, groups, approved_for = _tables()
    if label in vocab:
        return True
    approved = approved_for.get(label, frozenset()) & vocab
    if len(approved) == 1:
        return True
    return len(groups.get(label, frozenset()) & vocab) == 1


def unrepresentable(genes: Sequence[str] | Iterable[str]) -> list[str]:
    """The panel's genes that no checkpoint-based model on this cell can predict.

    Returns NAMES, in panel order, so the result composes with the `exclude_genes` argument
    that runner.run already takes.
    """
    return [g for g in genes if not _resolves(str(g))]
