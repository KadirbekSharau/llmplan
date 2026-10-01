"""Paths of the web UI beyond the acceptance tests: synthetic traffic, uploads that parse,
model id errors, the plan brake, and unexpected errors. Headless and offline (AppTest)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import llmplan.ui.state
from llmplan.ui import presets
from tests.acceptance.test_m6 import APP, PLAN_TIMEOUT_S, cost_per_day, plan

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=PLAN_TIMEOUT_S).run()


def test_first_load_shows_the_fixture_model_info_and_preset_stats() -> None:
    at = app()
    assert any("8,030,261,248" in code.value for code in at.code)  # model-info text
    assert any("csv format, 19,999 requests" in caption.value for caption in at.caption)
    (info,) = at.info
    assert info.value == "Choose the inputs in the sidebar, then click Plan."


def test_synthetic_traffic_plans_and_refuses_too_many_requests() -> None:
    at = app()
    at.radio(key="traffic_mode").set_value("Synthetic")
    at.run()
    at.number_input(key="syn_rate").set_value(0.5)
    at.number_input(key="syn_duration").set_value(600.0)
    plan(at.run())
    assert not at.error
    assert cost_per_day(at) > 0
    at.number_input(key="syn_rate").set_value(100.0)
    at.number_input(key="syn_duration").set_value(3600.0)
    plan(at.run())
    (error,) = at.error
    assert "360,000 expected requests; the web UI allows up to 200,000" in error.value


def test_a_valid_upload_shows_its_stats_and_plans() -> None:
    at = app()
    at.radio(key="traffic_mode").set_value("Upload CSV")
    at.run()
    assert at.file_uploader(key="upload").allowed_type == [".csv"]
    plan(at)
    assert "upload a CSV trace" in at.error[0].value
    content = (FIXTURES / "workload_azure2024_50.csv").read_bytes()
    at.file_uploader(key="upload").set_value(("trace.csv", content, "text/csv"))
    at.run()
    assert any("azure2024 format, 50 requests" in c.value for c in at.caption)
    plan(at.run())  # the second run reuses the parsed upload
    assert not at.error
    assert cost_per_day(at) > 0


def test_custom_model_ids_are_validated_before_any_request() -> None:
    at = app()
    at.selectbox(key="model_choice").set_value("Other Hugging Face id")
    at.run()
    plan(at)
    assert at.error[0].value == "enter a Hugging Face model id (org/name)"
    at.text_input(key="model_custom").set_value("not a repo id")
    plan(at.run())
    assert "invalid Hugging Face repo id 'not a repo id'" in at.error[0].value


def test_the_session_brake_stops_after_30_plans_an_hour() -> None:
    at = app()
    at.session_state["plan_times"] = tuple(
        time.time() - i for i in range(presets.MAX_PLANS_PER_HOUR)
    )
    plan(at.run())
    assert not at.error
    assert "30 plans in the last hour" in at.warning[0].value


def test_unexpected_errors_show_a_request_id_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object) -> object:
        raise RuntimeError("internal detail")

    monkeypatch.setattr(llmplan.ui.state, "run_plan", broken)
    at = app()
    at.number_input(key="utilization").set_value(0.65)  # a request no other test caches
    plan(at.run())
    (error,) = at.error
    assert error.value.startswith("Something went wrong (request id ")
    assert "internal detail" not in error.value
    assert not at.exception


@pytest.mark.slow
@pytest.mark.parametrize("chosen", presets.PRESETS, ids=lambda p: p.key)
def test_every_preset_plans_under_60_s_with_default_inputs(chosen: presets.TracePreset) -> None:
    """docs/LAUNCH.md checklist: a plan on each preset completes under 60 s (all GPUs)."""
    at = app()
    at.selectbox(key="preset").set_value(chosen.key)
    at.run()
    start = time.perf_counter()
    plan(at)
    elapsed = time.perf_counter() - start
    print(f"preset {chosen.key}: {elapsed:.2f} s")
    assert not at.error
    assert elapsed < PLAN_TIMEOUT_S
