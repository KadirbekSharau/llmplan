"""M3 acceptance tests: M3_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import load_model
from llmplan.errors import PerfError
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, estimate
from llmplan.perf.roofline import RooflineBackend


class WorkloadStats(BaseModel):
    """Field-for-field copy of M2's `WorkloadStats` (M2_DESIGN.md section 3).

    TODO(M2): import `WorkloadStats` from `llmplan.workload` once M2 is merged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_requests: int
    duration_s: float
    window_s: float
    n_windows: int
    mean_rps: float
    peak_window_rps: float
    peak_window_index: int
    input_tokens_p50: float
    input_tokens_p95: float
    input_tokens_p99: float
    input_tokens_mean: float
    input_tokens_max: int
    output_tokens_p50: float
    output_tokens_p95: float
    output_tokens_p99: float
    output_tokens_mean: float
    output_tokens_max: int
    peak_input_tokens_per_s: float
    peak_output_tokens_per_s: float
    hourly_rps: tuple[float, ...] | None


STATS = WorkloadStats(
    n_requests=1000,
    duration_s=600.0,
    window_s=60.0,
    n_windows=10,
    mean_rps=1000 / 600,
    peak_window_rps=3.0,
    peak_window_index=4,
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
    peak_input_tokens_per_s=1536.0,
    peak_output_tokens_per_s=768.0,
    hourly_rps=None,
)

LLAMA70 = load_model("fixture:llama3-70b")
H100 = load_gpus()["h100-sxm-80gb"]


def _config(tp: int, max_num_seqs: int = 256, max_model_len: int = 8192) -> ReplicaConfig:
    return ReplicaConfig(tensor_parallel=tp, max_num_seqs=max_num_seqs, max_model_len=max_model_len)


def test_stats_stand_in_satisfies_protocol() -> None:
    assert isinstance(STATS, StatsLike)


# 9.1 Roofline, llama3-70b bf16, h100-sxm-80gb, tp=4
def test_9_1_roofline_llama70b_h100_tp4() -> None:
    est = RooflineBackend().estimate(LLAMA70, H100, _config(4), STATS)
    assert isinstance(est, PerfEstimate)
    assert est.backend == "roofline"
    assert est.confidence == "roofline"
    assert est.source_urls == ()
    assert est.effective_batch == 256
    rel = 2e-2
    assert est.tpot_ms_p50 == pytest.approx(30.44, rel=rel)
    assert est.tpot_ms_p95 == pytest.approx(45.66, rel=rel)
    assert est.decode_tokens_per_s == pytest.approx(8411, rel=rel)
    assert est.prefill_tokens_per_s == pytest.approx(14_018, rel=rel)
    assert est.ttft_ms_p50 == pytest.approx(28.5, rel=rel)
    assert est.ttft_ms_p95 == pytest.approx(107.0, rel=rel)
    assert est.requests_per_s_capacity == pytest.approx(32.7, rel=rel)
    assert any("compute-bound" in note for note in est.assumptions)
    # The facade with the roofline backend returns the same estimate.
    assert estimate(LLAMA70, H100, _config(4), STATS, backend="roofline") == est


# 9.2 Roofline, tp=1: does not fit
def test_9_2_roofline_does_not_fit() -> None:
    assert RooflineBackend().estimate(LLAMA70, H100, _config(1), STATS) is None
    with pytest.raises(PerfError, match="does not fit"):
        estimate(LLAMA70, H100, _config(1), STATS, backend="roofline")


# 9.3 Roofline requires catalog fields
def test_9_3_roofline_requires_bandwidth() -> None:
    gpu = GPUSpec.model_validate(
        {**H100.model_dump(), "id": "h100-no-bandwidth", "memory_bandwidth_gbps": None}
    )
    assert RooflineBackend().estimate(LLAMA70, gpu, _config(4), STATS) is None
    with pytest.raises(PerfError, match="memory_bandwidth_gbps null"):
        estimate(LLAMA70, gpu, _config(4), STATS, backend="roofline")
