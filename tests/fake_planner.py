"""Test doubles for the planner (M4_DESIGN.md section 10).

`FakeBackend` is registered in the M3 perf registry under `"fake"`. It returns fixed
capacities per `(gpu_id, tensor_parallel)` from `FAKE_PERF`, which tests fill through the
`fake_perf` fixture (tests/conftest.py) so entries never leak between tests. Fake GPUs
have 10 TB of VRAM so every candidate fits (fit itself is the real M1 code). `sim_plan`
builds the M5 test plans: exact service times and, optionally, a small KV cache.

M7 adds two class-aware fakes. `"fake_shape"` derives every figure from the token
statistics it is given (a request-size class or the whole workload) and a fixed prefill
rate and time per output token per GPU (`SHAPE_PERF`). `"fake_class"` returns fixed
capacities per `(gpu_id, tensor_parallel)` and class index from `FAKE_CLASS_PERF` (filled
by the `fake_class_perf` fixture); a class it marks None fails a 500 ms TTFT SLO.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.memory.engine import EngineProfile
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, register
from llmplan.planner import plan
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.planner.result import PlanResult
from llmplan.workload import Workload, WorkloadStats

AS_OF = date(2026, 9, 30)
SOURCE = "https://example.com/test-only"
MODEL = load_model("fixture:llama3-8b")


@dataclass(frozen=True)
class FakePerf:
    rps: float
    tokens_per_s: float = 1000.0
    ttft_ms_p95: float = 100.0
    tpot_ms_p95: float = 10.0
    prefill_tokens_per_s: float = 10_000.0


FAKE_PERF: dict[tuple[str, int], FakePerf] = {}


@register("fake")
class FakeBackend:
    name = "fake"

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        spec = FAKE_PERF.get((gpu.id, config.tensor_parallel))
        if spec is None:
            return None
        return PerfEstimate(
            backend="table",
            confidence="measured",
            effective_batch=config.max_num_seqs,
            decode_tokens_per_s=spec.tokens_per_s,
            prefill_tokens_per_s=spec.prefill_tokens_per_s,
            requests_per_s_capacity=spec.rps,
            ttft_ms_p50=spec.ttft_ms_p95 / 2,
            ttft_ms_p95=spec.ttft_ms_p95,
            tpot_ms_p50=spec.tpot_ms_p95 / 2,
            tpot_ms_p95=spec.tpot_ms_p95,
            assumptions=("fake backend",),
            source_urls=(),
        )

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        return f"no fake capacity for ({gpu.id}, tp {config.tensor_parallel})"


@dataclass(frozen=True)
class ShapePerf:
    prefill_tokens_per_s: float
    tpot_s: float


SHAPE_PERF: dict[str, ShapePerf] = {
    "shape-a": ShapePerf(prefill_tokens_per_s=2_000.0, tpot_s=0.01),
    "shape-b": ShapePerf(prefill_tokens_per_s=20_000.0, tpot_s=0.01),
}


@register("fake_shape")
class ShapeBackend:
    """capacity = max_num_seqs / (input_mean / prefill + output_mean * tpot); decode tokens/s
    = max_num_seqs / tpot; TTFT p50/p95 = input p50/p95 / prefill; TPOT p50 = p95 = tpot."""

    name = "fake_shape"

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        spec = SHAPE_PERF.get(gpu.id)
        if spec is None:
            return None
        batch = config.max_num_seqs
        service_s = (
            stats.input_tokens_mean / spec.prefill_tokens_per_s
            + stats.output_tokens_mean * spec.tpot_s
        )
        return PerfEstimate(
            backend="table",
            confidence="measured",
            effective_batch=batch,
            decode_tokens_per_s=batch / spec.tpot_s,
            prefill_tokens_per_s=spec.prefill_tokens_per_s,
            requests_per_s_capacity=batch / service_s,
            ttft_ms_p50=stats.input_tokens_p50 / spec.prefill_tokens_per_s * 1000,
            ttft_ms_p95=stats.input_tokens_p95 / spec.prefill_tokens_per_s * 1000,
            tpot_ms_p50=spec.tpot_s * 1000,
            tpot_ms_p95=spec.tpot_s * 1000,
            assumptions=("fake shape backend",),
            source_urls=(),
        )

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        return f"no fake shape for {gpu.id}"


ClassCapacity = tuple[float, float] | None  # (req/s, output tokens/s) or ineligible
FAKE_CLASS_PERF: dict[tuple[str, int], tuple[ClassCapacity, ...]] = {}
INELIGIBLE_TTFT_MS = 10_000.0  # fails the 500 ms SLO of the class-aware tests


@register("fake_class")
class ClassBackend:
    """Fixed capacities per (gpu_id, tensor_parallel) and class index (`stats.index`); the
    whole workload (no `index`) gets the first eligible class's figures."""

    name = "fake_class"

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        per_class = FAKE_CLASS_PERF.get((gpu.id, config.tensor_parallel))
        if per_class is None:
            return None
        index = getattr(stats, "index", None)
        if index is None:
            entry = next((c for c in per_class if c is not None), None)
        else:
            entry = per_class[index]
        rps, tps = (1.0, 1.0) if entry is None else entry
        return PerfEstimate(
            backend="table",
            confidence="measured",
            effective_batch=config.max_num_seqs,
            decode_tokens_per_s=tps,
            prefill_tokens_per_s=10_000.0,
            requests_per_s_capacity=rps,
            ttft_ms_p50=50.0,
            ttft_ms_p95=INELIGIBLE_TTFT_MS if entry is None else 100.0,
            tpot_ms_p50=5.0,
            tpot_ms_p95=10.0,
            assumptions=("fake class backend",),
            source_urls=(),
        )

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        return f"no fake class capacity for ({gpu.id}, tp {config.tensor_parallel})"


def fake_gpu(gpu_id: str) -> GPUSpec:
    return GPUSpec(
        id=gpu_id,
        vendor="nvidia",
        name=f"Fake {gpu_id}",
        vram_bytes=10**13,
        memory_bandwidth_gbps=1000.0,
        fp16_dense_tflops=100.0,
        fp8_dense_tflops=None,
        nvlink=True,
        source_url=SOURCE,
        as_of=AS_OF,
    )


def fake_row(instance: str, gpu_id: str, gpu_count: int, price: float) -> PriceRow:
    return PriceRow(
        provider="test",
        instance=instance,
        gpu_id=gpu_id,
        gpu_count=gpu_count,
        price_usd_per_hour=price,
        commitment="on_demand",
        region=None,
        source_url=SOURCE,
        as_of=AS_OF,
    )


GPUS = {"fake-a": fake_gpu("fake-a"), "fake-b": fake_gpu("fake-b")}
ROW_A = fake_row("a-1x", "fake-a", 1, 2.0)
ROW_B = fake_row("b-8x", "fake-b", 8, 19.0)
SHAPE_GPUS = {"shape-a": fake_gpu("shape-a"), "shape-b": fake_gpu("shape-b")}
ROW_SA = fake_row("sa-1x", "shape-a", 1, 1.0)
ROW_SB = fake_row("sb-1x", "shape-b", 1, 3.0)


def stats(demand_rps: float, demand_tps: float = 0.0) -> WorkloadStats:
    """Workload statistics whose peak window is the given demand."""
    return WorkloadStats(
        n_requests=1000,
        duration_s=600.0,
        window_s=60.0,
        n_windows=10,
        mean_rps=1000 / 600,
        peak_window_rps=demand_rps,
        peak_window_index=0,
        input_tokens_p50=400,
        input_tokens_p95=1500,
        input_tokens_p99=3000,
        input_tokens_mean=512,
        input_tokens_max=8000,
        output_tokens_p50=200,
        output_tokens_p95=800,
        output_tokens_p99=1200,
        output_tokens_mean=256,
        output_tokens_max=2000,
        peak_input_tokens_per_s=0.0,
        peak_output_tokens_per_s=demand_tps,
        hourly_rps=None,
    )


def request(
    demand_rps: float,
    rows: tuple[PriceRow, ...] = (ROW_A, ROW_B),
    *,
    demand_tps: float = 0.0,
    slo: SLO | None = None,
    **options: object,
) -> PlanRequest:
    """A section 10 request: utilization 1.0, tp (1,), bf16, max_num_seqs (64)."""
    opts = {
        "tensor_parallel_choices": (1,),
        "dtype_choices": ("bf16",),
        "max_num_seqs_choices": (64,),
        "max_model_len": 8192,
        "perf_backend": "fake",
        **options,
    }
    return PlanRequest(
        model=MODEL,
        stats=stats(demand_rps, demand_tps),
        slo=slo or SLO(utilization_target=1.0),
        engine=EngineProfile(),
        options=PlanOptions.model_validate(opts),
        gpus=GPUS,
        prices=rows,
    )


def sim_plan(
    *,
    replicas: int = 1,
    effective_batch: int = 1,
    prefill_tokens_per_s: float = 1000.0,
    tpot_ms: float = 10.0,
    kv_token_capacity: int | None = None,
) -> PlanResult:
    """A planned fleet of `replicas` identical replicas on row A (1 GPU per instance).

    Planned with the fake backend (`effective_batch = max_num_seqs`, `tpot_ms_p50 =
    tpot_ms_p95 / 2`), then the replica count and optionally the KV capacity are set.
    """
    key = ("fake-a", 1)
    previous = FAKE_PERF.get(key)
    FAKE_PERF[key] = FakePerf(
        rps=1.0, tpot_ms_p95=2 * tpot_ms, prefill_tokens_per_s=prefill_tokens_per_s
    )
    try:
        result = plan(request(1.0, (ROW_A,), max_num_seqs_choices=(effective_batch,)))
    finally:
        if previous is None:
            del FAKE_PERF[key]
        else:
            FAKE_PERF[key] = previous
    (replica,) = result.replicas
    candidate = replica.candidate
    if kv_token_capacity is not None:
        assert candidate.fit is not None
        fit = candidate.fit.model_copy(update={"kv_token_capacity": kv_token_capacity})
        candidate = candidate.model_copy(update={"fit": fit})
    (item,) = result.fleet
    return result.model_copy(
        update={
            "replicas": (
                replica.model_copy(
                    update={"candidate": candidate, "count": replicas, "instances": replicas}
                ),
            ),
            "fleet": (
                item.model_copy(
                    update={"instances": replicas, "usd_per_day": item.usd_per_day * replicas}
                ),
            ),
            "cost_usd_per_day": result.cost_usd_per_day * replicas,
            "capacity_rps": result.capacity_rps * replicas,
        }
    )


def workload(arrivals: Sequence[float], input_tokens: int, output_tokens: int) -> Workload:
    """A trace of identical requests at the given arrival times (first must be 0.0)."""
    n = len(arrivals)
    return Workload(
        source="test",
        format="synthetic",
        frame=pd.DataFrame(
            {
                "arrival_s": np.asarray(arrivals, dtype=np.float64),
                "input_tokens": np.full(n, input_tokens, dtype=np.int64),
                "output_tokens": np.full(n, output_tokens, dtype=np.int64),
                "model": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
                "tenant": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
            }
        ),
        dropped_rows=0,
    )
