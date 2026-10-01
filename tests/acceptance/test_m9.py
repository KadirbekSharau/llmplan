"""M9 acceptance tests: M9_DESIGN.md section 9. These define done; do not relax them.

The page runs headless and offline through `streamlit.testing.v1.AppTest` (marker `ui`;
9.10 is also `slow`). 9.11 counts lines and runs in the default selection. 9.12 is the
M6, M7 and M8 UI suites themselves (tests/acceptance/test_m6.py, test_m8.py,
tests/unit/ui/), which run unchanged apart from the selectors listed in M9_NOTES.md.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Tab

from llmplan import render
from llmplan.ui import presets, state
from tests.acceptance.test_m6 import APP, H100, PLAN_TIMEOUT_S, cost_per_day, plan

SMALLEST = min(presets.SAMPLE_PRESETS, key=lambda p: p.rows).key
STEPS = ("Resolving model", "Loading traffic", "Estimating performance", "Solving", "Replaying")


def app(**params: str) -> AppTest:
    """A fresh session opened with the given query parameters, after its first run."""
    at = AppTest.from_file(str(APP), default_timeout=PLAN_TIMEOUT_S)
    at.query_params.update(params)
    return at.run()


def quick(at: AppTest, *gpu_ids: str) -> AppTest:
    """The smallest bundled sample (and only `gpu_ids`, when given): a plan in a second."""
    at.selectbox(key="preset").set_value(SMALLEST)
    if gpu_ids:
        at.multiselect(key="gpu_ids").set_value(list(gpu_ids))
    return at.run()


def last_run(at: AppTest) -> state.PlanRun:
    kind, run = at.session_state["outcome"]
    assert kind == "run", run
    assert isinstance(run, state.PlanRun)
    return run


def plan_json(at: AppTest) -> str:
    """The document behind "Download plan JSON" (`llmplan plan --format json`)."""
    run = last_run(at)
    return render.get("json").plan(run.request, run.result, run.comparison)


def tab(at: AppTest, label: str) -> Tab:
    (found,) = [t for t in at.tabs if t.label == label]
    return found


def charts_in(block: Tab) -> int:
    return len(block.get("vega_lite_chart"))


# 9.1 Empty state: two example scenarios; the first plans and shows the answer card.
@pytest.mark.ui
def test_9_1_empty_state_and_first_example() -> None:
    at = app()
    examples = [at.button(key=f"example_{i}") for i in range(2)]
    assert [b.label for b in examples] == list(presets.SCENARIOS)
    examples[0].click()
    at.run(timeout=PLAN_TIMEOUT_S)
    assert not at.exception
    assert not at.error
    assert cost_per_day(at) > 0
    assert not [b for b in at.button if (b.key or "").startswith("example_")]


# 9.2 Share link round trip: the link's parameters reproduce the plan JSON byte for byte.
@pytest.mark.ui
def test_9_2_share_link_round_trip() -> None:
    at = plan(quick(app(), H100, "l4-24gb"))
    (link,) = [c.value for c in at.code if "?model=" in c.value]
    assert len(link) < 2000
    params = dict(parse_qsl(urlsplit(link).query))
    assert params["traffic"] == SMALLEST
    assert params["gpu_ids"] == f"{H100},l4-24gb"
    first = plan_json(at)
    st.cache_data.clear()  # the fresh session plans again instead of reusing the cache
    fresh = app(**params)
    assert not fresh.exception
    assert not [w for w in fresh.warning if "Ignored link parameters" in w.value]
    assert plan_json(fresh) == first


# 9.3 Invalid parameters are ignored with a notice; the page still renders.
@pytest.mark.ui
def test_9_3_invalid_query_params_are_ignored_with_a_notice() -> None:
    at = app(gpu_ids=f"{H100},b999-1tb", ttft="-5", nope="1")
    assert not at.exception
    notices = [w.value for w in at.warning if w.value.startswith("Ignored link parameters")]
    assert notices == ["Ignored link parameters (unknown or invalid): gpu_ids, ttft, nope."]
    assert at.number_input(key="ttft").value == presets.DEFAULT_SLO.ttft_ms_p95
    assert H100 in at.multiselect(key="gpu_ids").value  # the default selection
    assert at.button(key="plan").label == "Plan"
    assert "outcome" not in at.session_state  # nothing valid to plan


# 9.4 Stale indicator after a TTFT change; gone after planning again.
@pytest.mark.ui
def test_9_4_stale_indicator() -> None:
    def stale(at: AppTest) -> bool:
        return any("Inputs changed since this plan" in m.value for m in at.markdown)

    at = plan(quick(app(), H100))
    assert not stale(at)
    at.number_input(key="ttft").set_value(400.0)
    at.run()
    assert stale(at)
    assert at.button(key="plan_again").label == "Plan again"
    plan(at)
    assert not stale(at)
    assert not [b for b in at.button if b.key == "plan_again"]


# 9.5 Error hints: "relax" for an infeasible target; the expected columns for a bad upload.
@pytest.mark.ui
def test_9_5_error_hints() -> None:
    at = app()
    at.number_input(key="ttft").set_value(1.0)
    plan(at.run())
    (error,) = at.error
    assert "relax" in error.value
    at.radio(key="traffic_mode").set_value("Upload CSV")
    at.run()
    both = b"arrival_s,timestamp,input_tokens,output_tokens\n0,2024-01-01T00:00:00Z,10,5\n"
    at.file_uploader(key="upload").set_value(("both.csv", both, "text/csv"))
    plan(at.run())
    columns = ("arrival_s", "timestamp", "input_tokens", "output_tokens")
    assert len([e for e in at.error if all(c in e.value for c in columns)]) == 1


# 9.6 Progress: the status container lists all five steps after a plan.
@pytest.mark.ui
def test_9_6_progress_lists_the_five_steps() -> None:
    at = plan(quick(app(), H100))
    (status,) = at.status
    assert status.state == "complete"
    lines = [m.value for m in status.markdown]
    assert [next(line for line in lines if line.startswith(step)) for step in STEPS]
    assert any(re.match(r"Solving \(time limit 10 s\): ", line) for line in lines)


# 9.7 Candidates: at most 15 rows, none rejected; the toggle adds rejected rows and reasons.
@pytest.mark.ui
def test_9_7_candidates_tab() -> None:
    at = plan(quick(app()))
    (table,) = tab(at, "Candidates").dataframe
    assert 0 < len(table.value) <= 15
    assert set(table.value["status"]) == {"eligible"}
    at.toggle(key="show_rejected").set_value(True)
    at.run()
    (table,) = tab(at, "Candidates").dataframe
    rejected = table.value[table.value["status"] != "eligible"]
    assert len(rejected) > 0
    assert rejected["reason"].str.len().gt(0).all()


# 9.8 Routing: the bar chart only with two or more classes.
@pytest.mark.ui
def test_9_8_routing_chart_only_with_classes() -> None:
    at = plan(quick(app(), H100))  # UI default: 2x2 classes
    assert len(last_run(at).result.classes) > 1
    assert charts_in(tab(at, "Routing")) == 1
    at.selectbox(key="classes").set_value("1")
    plan(at.run())
    assert not last_run(at).result.classes
    assert charts_in(tab(at, "Routing")) == 0


# 9.9 Timeline: another window re-renders with a different number of windows; PNG download.
@pytest.mark.ui
def test_9_9_timeline_window_selector() -> None:
    def windows(at: AppTest) -> int:
        (caption,) = [c.value for c in tab(at, "Timeline").caption if "windows of" in c.value]
        return int(re.search(r"in ([\d,]+) windows", caption).group(1).replace(",", ""))  # type: ignore[union-attr]

    at = plan(quick(app(), H100))
    auto = last_run(at).timeline.options.window_s
    before = windows(at)
    (selector,) = tab(at, "Timeline").radio
    assert selector.value == auto
    selector.set_value(state.window_choices(auto)[-1])
    at.run()
    assert windows(at) < before
    assert charts_in(tab(at, "Timeline")) == 1
    labels = [b.label for b in tab(at, "Timeline").get("download_button")]
    assert "Download PNG" in labels


# 9.10 Every preset plans under the default inputs (slow).
@pytest.mark.ui
@pytest.mark.slow
@pytest.mark.parametrize("chosen", presets.PRESETS, ids=lambda p: p.key)
def test_9_10_every_preset_plans_with_default_inputs(chosen: presets.TracePreset) -> None:
    at = app()
    at.selectbox(key="preset").set_value(chosen.key)
    plan(at.run())
    assert not at.exception
    assert not at.error
    assert cost_per_day(at) > 0


# 9.11 Line budget: llmplan/ui/ stays under 1,500 lines.
def test_9_11_ui_line_budget() -> None:
    lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in APP.parent.rglob("*.py"))
    assert lines < 1500, f"llmplan/ui/ has {lines} lines"
