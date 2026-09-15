"""One source of truth for withdrawn HGNC gene symbols, shared by every runner.

Why this exists. A runner that looks a target up literally in a model's vocabulary DECLINES a gene
the model actually supports whenever the benchmark panel carries the withdrawn symbol. The audit
found this on scGPT x T4: the local scGPT_human vocab has STING1 but not TMEM173, so the TMEM173
row fell back to the control mean and was scored as a prediction (+0.0223). PerturbNet's gene runner
had already worked around the same thing with its own private dict -- two copies of a fact is how
they drift apart, so both now read this table.

SCOPE, deliberately narrow. Only unambiguous HGNC renames (previous symbol -> approved symbol) are
listed. Paralogues and lncRNAs that a model genuinely does not annotate (BOLA2B, NEAT1, GAS5,
*-AS1) are NOT here: they must be declined by name, visibly. ATP5MD is likewise absent -- neither
ATP5MD nor its alias USMG5 is in the scGPT vocab, so renaming it would not help and asserting an
equivalence we have not checked would be worse than declining.

resolve() never invents a mapping: it returns a replacement only when the caller's own vocabulary
contains it, so a model that really lacks the gene still declines.
"""

from __future__ import annotations

from typing import Container, Iterable

# previous (withdrawn) symbol -> approved symbol, with the HGNC id and the panel it appears in
ALIASES: dict[str, str] = {
    "C10orf54": "VSIR",     # HGNC:30107, previous symbol C10orf54   (shifrut panel)
    "TCEB2": "ELOB",        # HGNC:11619, previous symbol TCEB2      (shifrut panel)
    "TMEM173": "STING1",    # HGNC:27962, previous symbol TMEM173    (frangieh panel)
}


def resolve(symbol: str, vocabulary: Container[str] | None = None) -> str | None:
    """The name to use for `symbol`, or None when the vocabulary supports neither spelling.

    With no vocabulary, returns the approved symbol if one is known, else the input unchanged.
    """
    if vocabulary is None:
        return ALIASES.get(symbol, symbol)
    if symbol in vocabulary:
        return symbol
    alt = ALIASES.get(symbol)
    if alt is not None and alt in vocabulary:
        return alt
    return None


def report(symbols: Iterable[str], vocabulary: Container[str]) -> tuple[dict[str, str], list[str]]:
    """-> ({input symbol: name actually used} for renames only, [symbols with no usable name]).

    Callers should print both: a silent rename is as hard to audit as a silent decline.
    """
    renamed: dict[str, str] = {}
    missing: list[str] = []
    for s in symbols:
        r = resolve(s, vocabulary)
        if r is None:
            missing.append(s)
        elif r != s:
            renamed[s] = r
    return renamed, missing
