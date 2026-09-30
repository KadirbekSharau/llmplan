"""Output renderers, one per `--format` value (ARCHITECTURE.md section 6).

A renderer turns library results into a string; it computes nothing that the library does
not already expose. Adding a format is a new module plus an import line below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from llmplan.errors import UnknownRegistryKey

if TYPE_CHECKING:
    from llmplan.catalog.hardware import GPUSpec
    from llmplan.catalog.models import ModelSpec
    from llmplan.memory.fit import FitRequest, FitResult
    from llmplan.perf.benchmarks import BenchmarkRow
    from llmplan.perf.config import ReplicaConfig
    from llmplan.perf.estimate import PerfEstimate, StatsLike
    from llmplan.planner.request import PlanRequest
    from llmplan.planner.result import PlanResult
    from llmplan.workload import Workload, WorkloadStats


class Renderer(Protocol):
    """Formats the results of each command (`fit`, `model-info`, `gpus`, `workload stats`,
    `perf ...`, `plan`)."""

    def fit(self, request: FitRequest, result: FitResult) -> str: ...

    def model_info(self, spec: ModelSpec) -> str: ...

    def gpus(self, gpus: Mapping[str, GPUSpec]) -> str: ...

    def workload_stats(self, workload: Workload, stats: WorkloadStats) -> str: ...

    def perf_estimate(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        config: ReplicaConfig,
        stats: StatsLike,
        result: PerfEstimate,
    ) -> str: ...

    def benchmarks(self, rows: Sequence[BenchmarkRow]) -> str: ...

    def plan(self, request: PlanRequest, result: PlanResult) -> str: ...


_REGISTRY: dict[str, Renderer] = {}


def register(key: str) -> Callable[[type[Renderer]], type[Renderer]]:
    """Class decorator: instantiate the class and register it under `key`."""

    def decorator(cls: type[Renderer]) -> type[Renderer]:
        _REGISTRY[key] = cls()
        return cls

    return decorator


def get(key: str) -> Renderer:
    """Return the renderer registered under `key`."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRegistryKey(f"unknown output format {key!r}; known: {known}") from None


from llmplan.render import json_render, text, vllm_cmd  # noqa: E402

__all__ = ["Renderer", "get", "json_render", "register", "text", "vllm_cmd"]
