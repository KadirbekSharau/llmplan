"""Throughput and latency model (M3): `estimate()` over registered `PerfBackend`s.

Importing this package registers the backends (`roofline`, `table`).
"""

from __future__ import annotations

from llmplan.perf import roofline, table
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import PerfBackend, PerfEstimate, StatsLike, estimate, get, register

__all__ = [
    "PerfBackend",
    "PerfEstimate",
    "ReplicaConfig",
    "StatsLike",
    "estimate",
    "get",
    "register",
    "roofline",
    "table",
]
