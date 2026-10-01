"""Interactive charts and the Timeline tab's helpers (M9): long-form timeline data, thinning
to 500 windows, the routing bar chart, window choices and the re-windowed replay."""

from __future__ import annotations

import pytest

from llmplan.simulate import SimOptions
from llmplan.ui import charts, state, wording
from tests.acceptance.test_m7 import two_class_request, two_class_workload


@pytest.fixture(scope="module")
def run() -> state.PlanRun:
    trace = two_class_workload()  # M7 8.2: 1 x SA + 1 x SB, $96/day, two classes
    request = two_class_request(trace)
    return state.run_plan(request, trace, SimOptions(window_s=1.0), request.gpus)


def test_timeline_frame_has_every_panel_and_series(run: state.PlanRun) -> None:
    frame = charts.timeline_frame(run.timeline)
    assert set(frame["panel"]) == set(charts.PANELS)
    assert frame.groupby("panel")["series"].nunique().to_dict() == {
        "req/s": 2,
        "utilization": 2,
        "VRAM (GB)": 3,
        "requests": 2,
    }
    first = run.timeline.windows[0]
    demand = frame[(frame["series"] == "demand") & (frame["minute"] == 0.0)]["value"]
    assert demand.tolist() == [first.demand_rps]


def test_long_timelines_are_thinned_to_500_windows(run: state.PlanRun) -> None:
    many = run.timeline.model_copy(update={"windows": run.timeline.windows * 20})
    assert len(many.windows) > charts.MAX_WINDOWS
    frame = charts.timeline_frame(many)
    assert frame[frame["series"] == "demand"].shape[0] <= charts.MAX_WINDOWS
    spec = charts.timeline_chart(run.timeline).to_dict()
    assert len(spec["vconcat"]) == 4
    assert spec["resolve"]["scale"]["x"] == "shared"


def test_routing_chart_has_one_bar_segment_per_rule(run: state.PlanRun) -> None:
    spec = charts.routing_chart(run.result).to_dict()
    (rows,) = spec["datasets"].values()
    assert len(rows) == len(run.result.routing) == 2
    assert {row["weight"] for row in rows} == {1.0}  # short -> SA, long -> SB (M7 8.2)
    assert rows[0]["class"].startswith("class 0: in 1-1,050")


def test_window_choices_and_replay_window(run: state.PlanRun) -> None:
    assert state.window_choices(50.0) == (10.0, 20.0, 50.0, 100.0, 200.0)
    assert state.window_choices(1.0) == (0.2, 0.5, 1.0, 2.0, 5.0)
    assert state.window_choices(0.01) == (0.002, 0.005, 0.01, 0.02, 0.05)
    trace = two_class_workload()
    wider = state.replay_window(run, trace, 5.0, run.request.gpus)
    assert wider.options.window_s == 5.0
    assert len(wider.windows) < len(run.timeline.windows)
    same = ("n_requests", "ttft_ms_p95", "tpot_ms_p95", "ttft_violation_pct")  # per request
    assert {k: getattr(wider.summary, k) for k in same} == {
        k: getattr(run.timeline.summary, k) for k in same
    }


def test_fleet_sentence(run: state.PlanRun) -> None:
    sentence = wording.fleet_sentence(run.result, run.request.gpus)
    assert sentence == (
        "1 x Fake shape-a (test sa-1x) + 1 x Fake shape-b (test sb-1x) serving 2 replicas"
    )
