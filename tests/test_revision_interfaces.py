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
    # 34/7/6 was the submitted 47-cell panel. Read against the assembler's own declaration so a
    # census change moves both together and this keeps guarding the status rule.
    from assemble_cross_cluster import EXPECTED_STATUS_COUNTS

    assert counts == EXPECTED_STATUS_COUNTS
    # PertAdapt was an adapted donor cell at T2 in the submitted panel. The re-run put it at T3
    # and T4 as native, and T2 now carries a Table S15b reason instead. What the test is really
    # for is that every ADAPTED cell states the interface written for it, so assert that of
    # whatever is adapted, and pin PertAdapt's placement to the census rather than to T2.
    adapted = [r for r in rows if r["status"] == "adapted"]
    assert adapted, "the panel reports no adapted cell; the status rule has nothing to check"
    for cell in adapted:
        assert cell["author_written_interface"].strip(), f"{cell['model']} @ {cell['task_id']}"
    retained = [r for r in rows if r["model"] == "PertAdapt"]
    assert {r["task_id"] for r in retained} == {"T3", "T4"}
    assert {r["status"] for r in retained} == {"native"}
    # The five phrases that used to be checked here were PertAdapt's T2 adapter, which no longer
    # exists as a cell. The invariant the test is for survives it: an adapted cell must SAY what
    # was written for this study, in enough detail that a reader can tell it from the published
    # interface, because that is what lets a shortfall be read as bounding our interface rather
    # than the architecture.
    for cell in adapted:
        text = cell["author_written_interface"]
        where = f"{cell['model']} @ {cell['task_id']}"
        assert text.startswith("Yes"), where
        assert "author-written" in text or "compound-conditioned head" in text, where
        # A stub-check, not a quality bar: enough words that the entry cannot be a bare "Yes".
        assert len(text.split()) >= 12, f"{where}: too short to identify the interface"


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
    # method_survey() derives this string live from census_metadata_rows(); the literal froze the
    # submitted panel. Build the expectation from the same census the survey reads.
    from assemble_cross_cluster import census_metadata_rows

    pert = sorted(
        (r["task_id"], r["status"]) for r in census_metadata_rows() if r["model"] == "PertAdapt"
    )
    assert survey.loc["PertAdapt", "Use in this study"] == "; ".join(
        f"{task} ({status})" for task, status in pert
    )


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
