"""Reproducibility metadata for one cluster cycle.

Writes results/<cluster>/manifest.json capturing everything needed to reproduce the run: git commit,
env, seeds, per-(split,baseline,seed) status + leak-audit reports, data provenance, and metric
config. This is the "메타데이터 정리" half of one paper cycle — every figure/draft number is
traceable back to a manifest entry.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> dict:
    out = {}
    for p in ("numpy", "scipy", "sklearn", "pandas", "matplotlib"):
        try:
            out[p] = __import__(p).__version__
        except Exception:
            out[p] = "absent"
    return out


def write_manifest(cluster: str, results_dir: Path, *, data_source: str, seeds: list[int],
                   splits: list[dict], rows: list[dict]) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ran = [r for r in rows if r.get("ran")]
    manifest = {
        "cluster": cluster,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "data_source": data_source,        # "synthetic_c1_like" or real accession + sha256
        "seeds": seeds,
        "splits": splits,                  # name, registry_task, leak audit summary
        "n_jobs": len(rows),
        "n_ran": len(ran),
        "n_leak_free": sum(1 for r in ran if r.get("leak_free")),
        "all_leak_free": all(r.get("leak_free") for r in ran) if ran else False,
        "jobs": [{k: r.get(k) for k in ("split", "baseline", "seed", "action",
                                        "headline_eligible", "ran", "leak_free")} for r in rows],
    }
    path = results_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
