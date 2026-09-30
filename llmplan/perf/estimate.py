"""Performance-model contract: `PerfEstimate`, the `PerfBackend` protocol, and `estimate()`.

Backends register themselves under a string key (ARCHITECTURE.md section 6). `estimate()`
validates inputs once, then asks one named backend or, for `"auto"`, each backend in
`AUTO_ORDER` until one answers. Section references are to docs/milestones/M3_DESIGN.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.errors import PerfError, UnknownRegistryKey, ValidationError
from llmplan.perf.config import ReplicaConfig

AUTO_ORDER = ("table", "roofline")
INPUT_STAT_FIELDS = ("input_tokens_mean", "input_tokens_p50", "input_tokens_p95")
OUTPUT_STAT_FIELDS = ("output_tokens_mean", "output_tokens_p50", "output_tokens_p95")


@runtime_checkable
class StatsLike(Protocol):
    """The `WorkloadStats` fields (M2_DESIGN.md section 3) the performance model reads.

    Declared here so M3 does not import `llmplan.workload` (built in parallel by M2); M2's
    `WorkloadStats` satisfies this protocol structurally. Token counts are per request.
    """

    @property
    def input_tokens_mean(self) -> float: ...

    @property
    def input_tokens_p50(self) -> float: ...

    @property
    def input_tokens_p95(self) -> float: ...

    @property
    def output_tokens_mean(self) -> float: ...

    @property
    def output_tokens_p50(self) -> float: ...

    @property
    def output_tokens_p95(self) -> float: ...


class PerfEstimate(BaseModel):
    """Throughput and service-time latency of one replica under a workload's token shape.

    Rates are aggregate for the replica; latencies are service times in milliseconds with
    no queueing. `effective_batch` is the concurrency the estimate assumes. `confidence` is
    `"roofline"` (first-principles bound), `"interpolated"`, or `"measured"` (an exact
    benchmark row). `source_urls` lists the benchmark pages used (empty for roofline).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["roofline", "table"]
    confidence: Literal["roofline", "interpolated", "measured"]
    effective_batch: int = Field(ge=1)
    decode_tokens_per_s: float = Field(gt=0)
    prefill_tokens_per_s: float = Field(gt=0)
    requests_per_s_capacity: float = Field(gt=0)
    ttft_ms_p50: float = Field(gt=0)
    ttft_ms_p95: float = Field(gt=0)
    tpot_ms_p50: float = Field(gt=0)
    tpot_ms_p95: float = Field(gt=0)
    assumptions: tuple[str, ...]
    source_urls: tuple[str, ...]


class PerfBackend(Protocol):
    """A performance model. `estimate` returns `None` when it cannot answer; `explain`
    then returns a one-line reason (called only after `estimate` returned `None`)."""

    name: str

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None: ...

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str: ...


_REGISTRY: dict[str, PerfBackend] = {}


def register(key: str) -> Callable[[type[PerfBackend]], type[PerfBackend]]:
    """Class decorator: instantiate the class and register it under `key`."""

    def decorator(cls: type[PerfBackend]) -> type[PerfBackend]:
        _REGISTRY[key] = cls()
        return cls

    return decorator


def get(key: str, backends: Mapping[str, PerfBackend] | None = None) -> PerfBackend:
    """Return the backend for `key`, from `backends` first, then the registry."""
    if backends is not None and key in backends:
        return backends[key]
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted({*_REGISTRY, *(backends or {})}))
        raise UnknownRegistryKey(f"unknown perf backend {key!r}; known: auto, {known}") from None


def _validate(model: ModelSpec, config: ReplicaConfig, stats: StatsLike) -> None:
    if config.max_model_len > model.max_position_embeddings:
        raise ValidationError(
            f"max_model_len {config.max_model_len} exceeds max_position_embeddings "
            f"{model.max_position_embeddings} of {model.id}"
        )
    if not isinstance(stats, StatsLike):
        raise ValidationError("stats must provide input/output token mean, p50, and p95")
    for name in (*INPUT_STAT_FIELDS, *OUTPUT_STAT_FIELDS):
        value = getattr(stats, name)
        low = 1.0 if name in INPUT_STAT_FIELDS else 0.0
        if not isinstance(value, int | float) or not math.isfinite(value) or value < low:
            raise ValidationError(f"stats.{name} must be a finite number >= {low:g}, got {value}")


def estimate(
    model: ModelSpec,
    gpu: GPUSpec,
    config: ReplicaConfig,
    stats: StatsLike,
    *,
    backend: str = "auto",
    backends: Mapping[str, PerfBackend] | None = None,
) -> PerfEstimate:
    """Estimate one replica's throughput and latency (ARCHITECTURE.md section 5, M3).

    `backend="auto"` tries `AUTO_ORDER` (table, then roofline) and returns the first answer;
    a named backend is used alone. `backends` overrides registry entries by key for this
    call (tests pass a table built from their own rows). Raises `ValidationError` for bad
    inputs, `UnknownRegistryKey` for an unknown backend, and `PerfError` naming every
    backend tried and why it could not answer.
    """
    _validate(model, config, stats)
    keys = AUTO_ORDER if backend == "auto" else (backend,)
    reasons: list[str] = []
    for key in keys:
        chosen = get(key, backends)
        result = chosen.estimate(model, gpu, config, stats)
        if result is not None:
            return result
        reasons.append(f"{key}: {chosen.explain(model, gpu, config, stats)}")
    raise PerfError(f"no performance estimate for {model.id} on {gpu.id} ({'; '.join(reasons)})")
