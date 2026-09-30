"""M4 acceptance tests: M4_DESIGN.md section 10. These define done; do not relax them.

The fake perf backend (tests/fake_planner.py, registered as "fake") returns fixed
capacities per (gpu_id, tp). Fake GPUs have 10 TB of VRAM, so the real M1 fit always
passes. Defaults: utilization_target 1.0, tp (1,), bf16, max_num_seqs (64), token demand 0.
"""

from __future__ import annotations

import pytest

from llmplan.planner import plan
from llmplan.planner.result import PlanResult
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
