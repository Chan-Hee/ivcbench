"""Keep the test suite out of the prediction deposit.

run_job deposits a bundle whenever IVCBENCH_PRED_DUMP is set, and runs/env.sh sets it to
predictions/v2_native -- which is the live deposit the census reads. Sourcing env.sh is the normal
way to run anything in this repository, so `. runs/env.sh && make test` wrote eight synthetic
bundles (15 held strata over 200 genes, compounds named cpd01) straight into the store. Four of
them landed on the unseen-compound floor and four on the OP3 NK lineage floor, and
assemble_cross_cluster.py then refused to assemble, correctly, because each floor member had two
visible bundles with different values.

No test needs its bundle to survive the test. Point the deposit at a temporary directory for the
whole session so the suite cannot reach the store no matter what the environment says.
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_prediction_deposit():
    with tempfile.TemporaryDirectory(prefix="ivcbench-test-deposit-") as tmp:
        keep = {k: os.environ.get(k) for k in ("IVCBENCH_PRED_DUMP", "IVCBENCH_PRED_DUMP_MEANS")}
        os.environ["IVCBENCH_PRED_DUMP"] = tmp
        try:
            yield tmp
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
