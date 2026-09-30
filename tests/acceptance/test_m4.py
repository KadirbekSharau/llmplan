"""M4 acceptance tests: M4_DESIGN.md section 10. These define done; do not relax them.

The fake perf backend (tests/fake_planner.py, registered as "fake") returns fixed
capacities per (gpu_id, tp). Fake GPUs have 10 TB of VRAM, so the real M1 fit always
passes. Defaults: utilization_target 1.0, tp (1,), bf16, max_num_seqs (64), token demand 0.
"""

from __future__ import annotations

import pytest

from llmplan import render
from llmplan.errors import InfeasiblePlan
from llmplan.planner import SLO, plan
from llmplan.planner.result import PlanResult
from llmplan.render.vllm_cmd import serve_command
from tests.conftest import UseFakePerf
from tests.fake_planner import ROW_A, ROW_B, FakePerf, request

A_B_CAPACITY = {("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)}


def _fleet(result: PlanResult) -> dict[str, int]:
    return {item.price_row.instance: item.instances for item in result.fleet}


# 10.1 Heterogeneous beats homogeneous.
def test_10_1_heterogeneous_beats_homogeneous(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    result = plan(request(34))
    assert _fleet(result) == {"b-8x": 1, "a-1x": 1}
    assert result.cost_usd_per_day == 21 * 24 == 504.0
    assert result.baseline is not None
    assert _fleet(result.baseline) == {"a-1x": 12}  # 2 x B = 912 is worse
    assert result.baseline.cost_usd_per_day == 576.0
    assert result.baseline_saving_pct == pytest.approx(12.5)
    assert result.binding == "requests"
    assert result.capacity_rps == 35.0
    assert result.solver.status == "optimal"


# 10.2 Homogeneous option.
def test_10_2_homogeneous_option(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    result = plan(request(34, homogeneous=True))
    assert _fleet(result) == {"a-1x": 12}
    assert result.cost_usd_per_day == 576.0
    assert result.baseline is None


# 10.3 Tie stays deterministic.
def test_10_3_tie_is_deterministic(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    row_b20 = ROW_B.model_copy(update={"price_usd_per_hour": 20.0})
    first = plan(request(30, (ROW_A, row_b20), seed=7))
    second = plan(request(30, (ROW_A, row_b20), seed=7))
    assert first.model_dump_json() == second.model_dump_json()
    assert _fleet(first) in ({"a-1x": 10}, {"b-8x": 1})
    assert first.cost_usd_per_day == 480.0


# 10.4 Token demand binds.
def test_10_4_token_demand_binds(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-a", 1): FakePerf(rps=3, tokens_per_s=300), ("fake-b", 1): FakePerf(rps=4)})
    result = plan(request(6, demand_tps=900))
    assert _fleet(result) == {"a-1x": 3}
    assert result.binding == "tokens"
    assert result.cost_usd_per_day == 144.0


# 10.5 SLO rejection.
def test_10_5_slo_rejection(fake_perf: UseFakePerf) -> None:
    fake_perf(
        {
            ("fake-a", 1): FakePerf(rps=3, ttft_ms_p95=100),
            ("fake-b", 1): FakePerf(rps=4, ttft_ms_p95=800),
        }
    )
    result = plan(request(34, slo=SLO(ttft_ms_p95=500, utilization_target=1.0)))
    assert _fleet(result) == {"a-1x": 12}
    (b,) = [c for c in result.candidates if c.price_row.instance == "b-8x"]
    assert b.status == "slo_ttft"
    assert "800" in b.reason
    assert "500" in b.reason


# 10.6 Infeasible.
def test_10_6_infeasible(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    with pytest.raises(InfeasiblePlan) as info:
        plan(request(34, slo=SLO(ttft_ms_p95=10, utilization_target=1.0)))
    assert "0 of" in str(info.value)
    assert "slo_ttft" in str(info.value)


# 10.7 GPU packing.
def test_10_7_gpu_packing(fake_perf: UseFakePerf) -> None:
    fake_perf(
        {
            ("fake-b", 1): FakePerf(rps=4),
            ("fake-b", 2): FakePerf(rps=9),
            ("fake-b", 4): FakePerf(rps=20),
        }
    )
    result = plan(request(40, (ROW_B,), tensor_parallel_choices=(1, 2, 4)))
    assert _fleet(result) == {"b-8x": 1}
    replicas = {r.candidate.config.tensor_parallel: r.count for r in result.replicas}
    assert replicas == {4: 2}
    assert result.cost_usd_per_day == 456.0


# 10.8 CP-SAT agrees with HiGHS.
def test_10_8_cp_sat_agrees_with_highs(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    highs = plan(request(34))
    cp_sat = plan(request(34, solver="cp_sat"))
    assert cp_sat.solver.backend == "cp_sat"
    assert _fleet(cp_sat) == _fleet(highs) == {"b-8x": 1, "a-1x": 1}
    assert cp_sat.cost_usd_per_day == highs.cost_usd_per_day == 504.0


# 10.11 Determinism and renderer.
def test_10_11_determinism_and_renderer(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B_CAPACITY)
    req = request(34)
    first = render.get("json").plan(req, plan(req))
    second = render.get("json").plan(req, plan(req))
    assert first == second
    fp8_request = request(34, dtype_choices=("fp8",))
    fp8 = plan(fp8_request)
    replica = fp8.replicas[0]
    assert replica.candidate.config.dtype == "fp8"
    assert "--quantization fp8" in serve_command(fp8_request.model, replica.candidate.config)
