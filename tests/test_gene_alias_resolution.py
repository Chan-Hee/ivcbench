"""Withdrawn HGNC symbols must resolve, and nothing else may be invented — S1 regression.

scGPT declined TMEM173 because the local scGPT_human vocab spells it STING1; that row fell back to
the control mean and was scored as a prediction. PerturbNet's gene runner already carried the same
mapping in a private dict, which is how the two drifted apart.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_runners"))
from gene_alias import ALIASES, report, resolve  # noqa: E402

VOCAB = Path("/data1/home/chlee/projects/single_cell_fm/models/scGPT_human/vocab.json")


def test_only_documented_renames_are_listed():
    # each entry must be previous-symbol -> approved-symbol, never the reverse and never identity
    for prev, approved in ALIASES.items():
        assert prev != approved
        assert approved not in ALIASES, f"{approved} is both a target and a source"


def test_nothing_is_invented_without_the_vocabulary():
    v = {"STING1", "TP53"}
    assert resolve("TMEM173", v) == "STING1"
    assert resolve("TP53", v) == "TP53"
    assert resolve("NEAT1", v) is None          # a real miss stays a miss
    assert resolve("ATP5MD", v) is None         # neither spelling is supported; do not guess
    # a rename is only applied when the vocabulary actually has the approved symbol
    assert resolve("TMEM173", {"TP53"}) is None


@pytest.mark.skipif(not VOCAB.exists(), reason="scGPT_human vocab is not present")
def test_the_audited_case_resolves_against_the_real_vocab():
    vocab = set(json.load(open(VOCAB)))
    assert "TMEM173" not in vocab and "STING1" in vocab   # the condition that caused the decline
    renamed, missing = report(["TMEM173", "C10orf54", "TCEB2", "ATP5MD"], vocab)
    assert renamed == {"TMEM173": "STING1", "C10orf54": "VSIR", "TCEB2": "ELOB"}
    assert missing == ["ATP5MD"]


def test_the_runners_share_one_table():
    src = (ROOT / "model_runners" / "perturbnet_c3_runner.py").read_text()
    assert "from gene_alias import" in src, "PerturbNet must not keep a private copy of the table"
    assert "from gene_alias import" in (ROOT / "model_runners" / "scgpt_runner.py").read_text()
