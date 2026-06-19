"""A unit of work = (cluster, split, baseline, seed). The scheduler (Phase 2) fills a 2-GPU queue
with these, tracks queued/running/done/failed, and is resumable. Results are written as one row per
Job so partial progress survives interruptions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    cluster: str
    split_name: str
    baseline: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.cluster}/{self.split_name}/{self.baseline}/seed{self.seed}"
