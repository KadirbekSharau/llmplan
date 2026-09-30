"""Throughput and latency model (M3): `estimate()` over registered `PerfBackend`s."""

from __future__ import annotations

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
]
