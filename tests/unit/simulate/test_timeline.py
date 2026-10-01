from __future__ import annotations

import numpy as np
import pydantic
import pytest

from llmplan.planner.request import SLO
from llmplan.simulate import SimOptions, Timeline, replay
from llmplan.simulate.stepfn import step_windows
from tests.fake_planner import sim_plan, workload


def test_step_windows_integral_and_max() -> None:
    # 0 until 1, then 2 until 3, then 0; a zero-length spike to 5 at t=3; window 2 s.
    time_s = np.array([1.0, 3.0, 3.0])
    value = np.array([2.0, 5.0, 0.0])
    integral, maximum = step_windows(time_s, value, np.array([0.0, 2.0, 4.0]))
    assert integral.tolist() == [2.0, 2.0]
    assert maximum.tolist() == [2.0, 5.0]


def test_step_windows_carries_state_into_a_window() -> None:
    integral, maximum = step_windows(np.array([0.5]), np.array([3.0]), np.array([0.0, 1.0, 2.0]))
    assert integral.tolist() == [1.5, 3.0]
    assert maximum.tolist() == [3.0, 3.0]


def test_step_windows_of_an_idle_replica() -> None:
    integral, maximum = step_windows(np.array([]), np.array([]), np.array([0.0, 1.0, 2.0]))
    assert integral.tolist() == [0.0, 0.0]
    assert maximum.tolist() == [0.0, 0.0]


def test_windows_extend_to_the_last_completion() -> None:
    # Two requests at 0 and 1 s, service 0.5 s, one slot, 1 s windows: the second completes
    # at 1.5 s, inside window 1; nothing arrives or completes later.
    timeline = replay(sim_plan(), workload([0.0, 1.0], 100, 40), options=SimOptions(window_s=1.0))
    assert [w.arrivals for w in timeline.windows] == [1, 1]
    assert [w.completions for w in timeline.windows] == [1, 1]
    assert [w.replicas[0].utilization for w in timeline.windows] == [0.5, 0.5]
    # A trace whose last completion falls past the last arrival window gets extra windows.
    late = replay(sim_plan(), workload([0.0, 0.0, 0.0], 100, 40), options=SimOptions(window_s=1.0))
    assert [w.arrivals for w in late.windows] == [3, 0]
    assert [w.completions for w in late.windows] == [1, 2]
    assert late.windows[1].demand_rps == 0.0
    assert late.windows[0].replicas[0].queue_depth_max == 2
    assert late.windows[0].replicas[0].queue_depth_mean == pytest.approx(1.5)


def test_window_records_and_summary() -> None:
    plan = sim_plan(replicas=2, effective_batch=2)
    timeline = replay(plan, workload([0.0, 0.0, 0.0, 70.0], 100, 40))
    first, second = timeline.windows
    assert first.start_s == 0.0
    assert second.start_s == 60.0
    assert first.capacity_rps == plan.capacity_rps
    assert first.demand_rps == 3 / 60
    assert [r.requests_started for r in first.replicas] == [2, 1]
    assert [r.requests_completed for r in second.replicas] == [1, 0]
    replica = first.replicas[0]
    assert replica.kv_tokens_in_use_max == 280
    assert replica.kv_tokens_in_use_mean == round(280 * 0.5 / 60)
    spec_candidate = plan.replicas[0].candidate
    assert spec_candidate.fit is not None
    assert replica.kv_bytes_in_use_max == 280 * spec_candidate.fit.kv_bytes_per_token_per_gpu
    assert replica.weight_bytes == spec_candidate.fit.per_gpu_weight_bytes
    assert first.ttft_ms_p95 == 100.0
    assert first.e2e_ms_p95 == 500.0
    assert timeline.summary.n_requests == 4
    assert timeline.summary.tpot_ms_p95 == 10.0
    assert timeline.summary.mean_utilization == pytest.approx(
        np.mean([0.5 / 60, 0.5 / 60 / 2, 0.5 / 60 / 2, 0.0])
    )
    assert timeline.plan_cost_usd_per_day == plan.cost_usd_per_day


def test_window_without_completions_has_no_percentiles() -> None:
    timeline = replay(sim_plan(), workload([0.0, 130.0], 100, 40))
    assert [w.completions for w in timeline.windows] == [1, 0, 1]
    assert timeline.windows[1].ttft_ms_p95 is None
    assert timeline.windows[1].e2e_ms_p95 is None


def test_budgets_default_to_the_slo() -> None:
    trace = workload([0.0, 0.0], 100, 40)  # second request waits 0.5 s: TTFT 600 ms
    timeline = replay(sim_plan(), trace, slo=SLO(ttft_ms_p95=200.0, tpot_ms_p95=5.0))
    assert timeline.options.ttft_budget_ms == 200.0
    assert timeline.options.tpot_budget_ms == 5.0
    assert timeline.summary.ttft_violation_pct == 50.0
    assert timeline.summary.tpot_violation_pct == 100.0
    assert timeline.windows[0].ttft_violations == 1
    assert timeline.windows[0].tpot_violations == 2
    explicit = replay(
        sim_plan(), trace, slo=SLO(ttft_ms_p95=200.0), options=SimOptions(ttft_budget_ms=700.0)
    )
    assert explicit.options.ttft_budget_ms == 700.0
    assert explicit.summary.ttft_violation_pct == 0.0
    assert any("no TPOT budget" in note for note in explicit.assumptions)


def test_no_budgets_count_no_violations() -> None:
    timeline = replay(sim_plan(), workload([0.0, 0.0], 100, 40))
    assert timeline.summary.ttft_violation_pct == 0.0
    assert any("no TTFT budget set" in note for note in timeline.assumptions)


def test_kv_capped_requests_are_noted() -> None:
    timeline = replay(sim_plan(effective_batch=4, kv_token_capacity=100), workload([0.0], 100, 40))
    assert any("capped at the replica's KV capacity" in note for note in timeline.assumptions)


def test_timeline_json_round_trips() -> None:
    timeline = replay(sim_plan(replicas=2), workload([0.25 * i for i in range(20)], 100, 40))
    assert Timeline.model_validate_json(timeline.model_dump_json()) == timeline


@pytest.mark.parametrize(
    "field",
    [{"window_s": 0}, {"window_s": float("inf")}, {"max_requests": 0}, {"ttft_budget_ms": -1}],
)
def test_sim_options_validation(field: dict[str, float]) -> None:
    with pytest.raises(pydantic.ValidationError):
        SimOptions.model_validate(field)
