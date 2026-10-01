"""M9 renderers used by the web UI: the Altair charts (long-form timeline data, thinning to
500 windows, the routing bar chart), the fleet in one sentence, and the model summary."""

from __future__ import annotations

import pytest

from llmplan.catalog.models import load_model
from llmplan.render import charts
from llmplan.render.plan_text import fleet_sentence
from llmplan.render.text import model_summary
from llmplan.simulate import SimOptions
from llmplan.ui import state
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


def test_fleet_sentence(run: state.PlanRun) -> None:
    sentence = fleet_sentence(run.result, run.request.gpus)
    assert sentence == (
        "1 x Fake shape-a (test sa-1x) + 1 x Fake shape-b (test sb-1x) serving 2 replicas"
    )


def test_model_summary_dense_and_mixture_of_experts() -> None:
    assert model_summary(load_model("fixture:llama3-8b")) == (
        "8.03B parameters · GQA · 128 KiB KV cache per token (bf16)"
    )
    assert model_summary(load_model("fixture:qwen3-30b-a3b")) == (
        "30.53B parameters (3.35B active) · GQA · 96 KiB KV cache per token (bf16)"
    )
