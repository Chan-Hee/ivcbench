"""Regression checks for operation attribution and reference/formatting metadata."""

import ast
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assemble_cross_cluster import census_metadata_rows


def submission_tables():
    source = ROOT / "submission/tables.py"
    if not source.is_file():
        source = ROOT.parent / "revision_BIB-26-1553/03_etc/11_final/tables.py"
    spec = importlib.util.spec_from_file_location("tested_submission_tables", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pertadapt_is_an_explicit_study_written_donor_adaptation():
    rows = census_metadata_rows()
    counts = {
        state: sum(r["status"] == state for r in rows)
        for state in ("native", "adapted", "diagnostic")
    }
    assert counts == {"native": 34, "adapted": 7, "diagnostic": 6}
    retained = [r for r in rows if r["model"] == "PertAdapt"]
    assert len(retained) == 1
    assert retained[0]["task_id"] == "T2"
    assert retained[0]["status"] == "adapted"
    explanation = retained[0]["author_written_interface"]
    for required in (
        "stimulated training cells",
        "held-donor control embeddings",
        "not the published",
        "binary GO",
        "shared decoder",
    ):
        assert required in explanation


def test_executed_references_are_not_unevaluated_model_panel_entries():
    survey = submission_tables().method_survey().set_index("Entry")
    for name in (
        "cell-mean shift",
        "linear-PCA shift",
        "control-as-prediction",
        "donor shift",
    ):
        assert survey.loc[name, "Use in this study"].startswith("Executed")
        assert "not evaluated" not in survey.loc[name, "Use in this study"]
    assert "training-mean shift" not in survey.index
    assert (
        "training-mean protein"
        in survey.loc["cell-mean shift", "Inputs / prediction operation"]
    )
    assert survey.loc["MAP", "Use in this study"] == "Surveyed; not evaluated"
    assert survey.loc["PertAdapt", "Use in this study"] == "T2 (adapted)"


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0075, "+0.008"),
        (-0.0075, "−0.008"),
        (0.0005, "+0.001"),
        (-0.0005, "−0.001"),
    ],
)
def test_decimal_half_ties_have_one_table_rounding_rule(value, expected):
    assert submission_tables().number(value, signed=True) == expected


def test_study_specific_pertadapt_import_surface_is_shipped():
    vendor = ROOT / "vendor/pertadapt"
    trees = {
        name: ast.parse((vendor / name).read_text())
        for name in ("__init__.py", "pertadapt_modules.py", "build_go_mask.py")
    }
    exports = {
        alias.name
        for node in ast.walk(trees["__init__.py"])
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    required = {"GOMaskedPertAdapter", "loss_adapt", "additive_go_mask", "load_gene2go"}
    assert required <= exports
    definitions = {
        node.name
        for tree in trees.values()
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert required <= definitions
    assert (
        vendor / "UPSTREAM_COMMIT.txt"
    ).read_text().strip() == "53bb7f05b5b21837d5851b7aea51ee29eae5fa3b"


def test_inventory_cannot_carry_a_second_stale_execution_matrix():
    table = submission_tables()
    assert table.META["method_inventory"][0] == ["Entry", "Approach", "Surveyed inputs"]
    assert all(len(row) == 3 for row in table.META["method_inventory"])
    refs = table.META["main_references"]
    assert "10.1093/bioinformatics/btag307" in refs["44"]
