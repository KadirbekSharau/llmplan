"""The run behind the result tabs (M9): window choices, the re-windowed replay of the
Timeline tab, and the stage timings `run_plan` reports for the progress steps."""

from __future__ import annotations

from llmplan.simulate import SimOptions
from llmplan.ui import state
from tests.acceptance.test_m7 import two_class_request, two_class_workload


def test_window_choices_and_replay_window() -> None:
    assert state.window_choices(50.0) == (10.0, 20.0, 50.0, 100.0, 200.0)
    assert state.window_choices(1.0) == (0.2, 0.5, 1.0, 2.0, 5.0)
    assert state.window_choices(0.01) == (0.002, 0.005, 0.01, 0.02, 0.05)
    trace = two_class_workload()  # M7 8.2: 1 x SA + 1 x SB, $96/day, two classes
    request = two_class_request(trace)
    run = state.run_plan(request, trace, SimOptions(window_s=1.0), request.gpus)
    wider = state.replay_window(run, trace, 5.0)
    assert wider.options.window_s == 5.0
    assert len(wider.windows) < len(run.timeline.windows)
    same = ("n_requests", "ttft_ms_p95", "tpot_ms_p95", "ttft_violation_pct")  # per request
    assert {k: getattr(wider.summary, k) for k in same} == {
        k: getattr(run.timeline.summary, k) for k in same
    }


def test_run_plan_reports_each_stage() -> None:
    trace = two_class_workload()
    request = two_class_request(trace)
    stages: dict[str, float] = {}
    state.run_plan(request, trace, SimOptions(), request.gpus, progress=stages.__setitem__)
    assert list(stages) == ["estimate", "solve", "replay"]
    assert all(seconds >= 0 for seconds in stages.values())
