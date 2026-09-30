"""Test doubles for the planner (M4_DESIGN.md section 10).

`FakeBackend` is registered in the M3 perf registry under `"fake"`. It returns fixed
capacities per `(gpu_id, tensor_parallel)` from `FAKE_PERF`, which tests fill through the
`fake_perf` fixture (tests/conftest.py) so entries never leak between tests. Fake GPUs
have 10 TB of VRAM so every candidate fits (fit itself is the real M1 code).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.memory.engine import EngineProfile
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, register
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.workload import WorkloadStats

AS_OF = date(2026, 9, 30)
SOURCE = "https://example.com/test-only"
MODEL = load_model("fixture:llama3-8b")


@dataclass(frozen=True)
class FakePerf:
    rps: float
    tokens_per_s: float = 1000.0
    ttft_ms_p95: float = 100.0
    tpot_ms_p95: float = 10.0


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
            prefill_tokens_per_s=10_000.0,
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
