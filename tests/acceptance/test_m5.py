"""M5 acceptance tests: M5_DESIGN.md section 9. These define done; do not relax them.

Plans come from the M4 planner with the fake perf backend (tests/fake_planner.py,
`sim_plan`): fake GPUs have 10 TB of VRAM, `effective_batch = max_num_seqs`, and the
prefill rate and time per output token are set exactly, so every service time is
hand-checkable. Defaults: prefill 1000 tokens/s, tpot 10 ms, one slot, one replica.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import load_model
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.planner.result import PlanResult
from llmplan.render.plots import save_png
from llmplan.simulate import SimOptions, replay, replay_requests
from llmplan.workload import Distribution, compute_stats, generate, load_workload
from tests.fake_planner import sim_plan, workload

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# Token lengths spanning workload_10.csv (inputs 100..1000, outputs 10..100).
INPUTS = Distribution(kind="uniform", lo=100, hi=1000)
OUTPUTS = Distribution(kind="uniform", lo=10, hi=100)

BUDGET_200 = SimOptions(ttft_budget_ms=200.0)


# 9.1 No queueing: service = 100 / 1000 + 40 * 0.010 = 0.5 s, one request per second.
def test_9_1_no_queueing() -> None:
    plan = sim_plan()
    trace = workload([1.0 * i for i in range(100)], 100, 40)
    requests = replay_requests(plan, trace, options=BUDGET_200).frame
    assert (requests["ttft_ms"] == 100.0).all()
    assert (requests["e2e_ms"] == 500.0).all()
    timeline = replay(plan, trace, options=BUDGET_200)
    assert timeline.summary.max_queue_depth == 0
    assert timeline.windows[0].replicas[0].utilization == 0.5  # 30 busy slot-s over 60
    assert timeline.summary.ttft_violation_pct == 0.0


# 9.2 Overload queues: one request every 0.25 s, service 0.5 s. The busy window is read
# with 10 s windows; the replica is busy from 0 to 50 s (M5_NOTES.md).
def test_9_2_overload_queues() -> None:
    plan = sim_plan()
    trace = workload([0.25 * i for i in range(100)], 100, 40)
    options = BUDGET_200.model_copy(update={"window_s": 10.0})
    timeline = replay(plan, trace, options=options)
    requests = replay_requests(plan, trace, options=options).frame
    assert timeline.summary.max_queue_depth >= 45
    assert requests["ttft_ms"].iloc[-1] > 20_000
    assert timeline.summary.ttft_violation_pct > 50
    assert timeline.windows[0].replicas[0].utilization == pytest.approx(1.0, abs=1e-9)


# 9.3 Two replicas, least_outstanding: two slots, service 0.5 s, arrivals every 0.25 s.
def test_9_3_two_replicas_halve_the_wait() -> None:
    plan = sim_plan(replicas=2)
    trace = workload([0.25 * i for i in range(100)], 100, 40)
    options = SimOptions(routing="least_outstanding")
    timeline = replay(plan, trace, options=options)
    requests = replay_requests(plan, trace, options=options).frame
    assert timeline.summary.max_queue_depth == 0
    assert (requests["ttft_ms"] == 100.0).all()


# 9.4 KV limits admission: 300 tokens per request, 1000 KV tokens, so 3 run at once.
# Pinned to M5's full reservation; M7's default incremental accounting reserves 250 tokens
# (input + output / 2), so 4 would run (M7_NOTES.md).
def test_9_4_kv_limits_admission_before_slots() -> None:
    plan = sim_plan(effective_batch=8, kv_token_capacity=1000)
    trace = workload([0.0] * 10, 200, 100)
    full = SimOptions(kv_accounting="full")
    requests = replay_requests(plan, trace, options=full).frame
    starts = requests["start_s"].tolist()
    first_complete = requests["complete_s"].iloc[0]
    assert starts[:3] == [0.0, 0.0, 0.0]
    assert starts[3] == first_complete == 0.2 + 100 * 0.01
    assert sum(start == 0.0 for start in starts) == 3
    timeline = replay(plan, trace, options=full)
    assert timeline.windows[0].replicas[0].queue_depth_max == 7


# 9.5 Routing tie-break determinism.
def test_9_5_round_robin_is_deterministic() -> None:
    plan = sim_plan(replicas=3)
    trace = workload([0.1 * i for i in range(30)], 100, 40)
    options = SimOptions(routing="round_robin")
    requests = replay_requests(plan, trace, options=options).frame
    assert requests["replica_index"].tolist() == [i % 3 for i in range(30)]
    first = replay(plan, trace, options=options).model_dump_json()
    second = replay(plan, trace, options=options).model_dump_json()
    assert first == second


# 9.8 Truncation.
def test_9_8_truncation() -> None:
    timeline = replay(
        sim_plan(),
        workload([1.0 * i for i in range(100)], 100, 40),
        options=SimOptions(max_requests=50),
    )
    assert timeline.summary.n_truncated == 50
    assert timeline.summary.n_requests == 50
    assert any("truncated" in note for note in timeline.assumptions)


# 9.9 Plot: a PNG written headless (Agg canvas, no display).
def test_9_9_plot(tmp_path: Path) -> None:
    timeline = replay(sim_plan(replicas=2), workload([0.25 * i for i in range(100)], 100, 40))
    path = tmp_path / "timeline.png"
    save_png(timeline, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def plan_10_9() -> tuple[PlanResult, float]:
    """The M4 acceptance plan 10.9 (llama3-8b, roofline, shipped catalogs) and its
    workload's peak window rate."""
    stats = compute_stats(load_workload(FIXTURES / "workload_10.csv"))
    request = PlanRequest(
        model=load_model("fixture:llama3-8b"),
        stats=stats,
        slo=SLO(),
        engine=EngineProfile(),
        options=PlanOptions(max_model_len=8192, perf_backend="roofline"),
        gpus=load_gpus(),
        prices=load_prices(),
    )
    return plan(request), stats.peak_window_rps


def ttft_budget(result: PlanResult) -> SimOptions:
    """TTFT budget of twice the largest planned replica's service-time TTFT p95."""
    p95 = max(r.candidate.perf.ttft_ms_p95 for r in result.replicas if r.candidate.perf)
    return SimOptions(ttft_budget_ms=2 * p95)


# 9.6 Real plan, well provisioned: half the plan's peak demand.
def test_9_6_real_plan_well_provisioned() -> None:
    result, peak_rps = plan_10_9()
    trace = generate(
        rate_rps=peak_rps * 0.5, duration_s=3600, input_tokens=INPUTS, output_tokens=OUTPUTS, seed=0
    )
    timeline = replay(result, trace, options=ttft_budget(result))
    assert timeline.summary.ttft_violation_pct < 5
    assert timeline.summary.mean_utilization < 0.8


# 9.7 Real plan, under-provisioned: one replica at 3x its capacity. The design's "3x the
# plan's demand" (0.3 req/s) cannot queue on this plan: its single replica serves 29.3 req/s,
# about 98x that rate. M5_NOTES.md shows the arithmetic.
def test_9_7_real_plan_under_provisioned() -> None:
    result, _ = plan_10_9()
    first = result.replicas[0]
    one = result.model_copy(update={"replicas": (first.model_copy(update={"count": 1}),)})
    assert first.candidate.perf is not None
    capacity_rps = first.candidate.perf.requests_per_s_capacity  # one replica, not derated
    trace = generate(
        rate_rps=3 * capacity_rps,
        duration_s=300,
        input_tokens=INPUTS,
        output_tokens=OUTPUTS,
        seed=0,
    )
    timeline = replay(one, trace, options=ttft_budget(result))
    assert timeline.summary.ttft_violation_pct > 50
    assert timeline.summary.max_queue_depth > 10


# 9.10 Performance: about 200k requests on 4 replicas in under 30 s.
@pytest.mark.slow
def test_9_10_performance() -> None:
    fleet = sim_plan(replicas=4, effective_batch=64, prefill_tokens_per_s=10_000.0, tpot_ms=5.0)
    trace = generate(
        rate_rps=252.0,
        duration_s=800.0,
        input_tokens=Distribution(kind="lognormal", mean=6.2, sigma=0.8),
        output_tokens=Distribution(kind="lognormal", mean=4.8, sigma=0.9),
        seed=0,
    )
    assert len(trace.frame) >= 200_000
    start = time.perf_counter()
    timeline = replay(fleet, trace)
    elapsed = time.perf_counter() - start
    print(f"9.10: replayed {timeline.summary.n_requests:,} requests in {elapsed:.2f} s")
    assert timeline.summary.n_requests == len(trace.frame)
    assert elapsed < 30
