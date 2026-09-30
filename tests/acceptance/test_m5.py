"""M5 acceptance tests: M5_DESIGN.md section 9. These define done; do not relax them.

Plans come from the M4 planner with the fake perf backend (tests/fake_planner.py,
`sim_plan`): fake GPUs have 10 TB of VRAM, `effective_batch = max_num_seqs`, and the
prefill rate and time per output token are set exactly, so every service time is
hand-checkable. Defaults: prefill 1000 tokens/s, tpot 10 ms, one slot, one replica.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmplan.render.plots import save_png
from llmplan.simulate import SimOptions, replay, replay_requests
from tests.fake_planner import sim_plan, workload

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
def test_9_4_kv_limits_admission_before_slots() -> None:
    plan = sim_plan(effective_batch=8, kv_token_capacity=1000)
    trace = workload([0.0] * 10, 200, 100)
    requests = replay_requests(plan, trace).frame
    starts = requests["start_s"].tolist()
    first_complete = requests["complete_s"].iloc[0]
    assert starts[:3] == [0.0, 0.0, 0.0]
    assert starts[3] == first_complete == 0.2 + 100 * 0.01
    assert sum(start == 0.0 for start in starts) == 3
    timeline = replay(plan, trace)
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
