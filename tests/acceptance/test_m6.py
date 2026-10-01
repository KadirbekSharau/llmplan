"""M6 acceptance tests: M6_DESIGN.md section 9. These define done; do not relax them.

The app runs headless and offline through `streamlit.testing.v1.AppTest`. Inputs are the
`fixture:llama3-8b` model and the bundled trace samples (or the synthetic preset when no
sample is committed); nothing touches the network.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import llmplan.workload
from llmplan.ui import presets

pytestmark = pytest.mark.ui  # M8 9b: the Streamlit AppTest suite runs in the `ui` CI job

APP = Path(__file__).resolve().parents[2] / "llmplan" / "ui" / "app.py"
PLAN_TIMEOUT_S = 60
H100 = "h100-sxm-80gb"


def app() -> AppTest:
    """A fresh session of the app after its first run."""
    return AppTest.from_file(str(APP), default_timeout=PLAN_TIMEOUT_S).run()


def smallest_preset() -> presets.TracePreset:
    """The smallest bundled sample, or the synthetic preset when none is committed."""
    if presets.SAMPLE_PRESETS:
        return min(presets.SAMPLE_PRESETS, key=lambda p: p.rows)
    return presets.SYNTHETIC_PRESET


def texts(at: AppTest) -> Iterator[str]:
    """Every text value on the page (sidebar included)."""
    for kind in ("markdown", "caption", "code", "error", "warning", "info", "text", "exception"):
        for element in at.get(kind):
            yield str(getattr(element, "value", ""))


def cost_per_day(at: AppTest) -> float:
    (metric,) = [m for m in at.metric if m.label == "Cost per day"]
    return float(re.sub(r"[$,]", "", metric.value))


def plan(at: AppTest) -> AppTest:
    at.button(key="plan").click()
    return at.run(timeout=PLAN_TIMEOUT_S)


def preset_session(*gpu_ids: str) -> AppTest:
    at = app()
    at.selectbox(key="model_choice").set_value("fixture:llama3-8b")
    at.selectbox(key="preset").set_value(smallest_preset().key)
    at.multiselect(key="gpu_ids").set_value(list(gpu_ids))
    return at.run()


# 9.1 Loads.
def test_9_1_loads() -> None:
    at = app()
    assert not at.exception
    assert [b.label for b in at.button if b.label == "Plan"] == ["Plan"]


# 9.2 Preset end to end, default SLO, H100 and L4, within 60 s.
def test_9_2_preset_end_to_end() -> None:
    at = preset_session(H100, "l4-24gb")
    start = time.perf_counter()
    plan(at)
    elapsed = time.perf_counter() - start
    print(f"9.2: preset {smallest_preset().key} planned and replayed in {elapsed:.2f} s")
    assert elapsed < PLAN_TIMEOUT_S
    assert not at.exception
    assert not at.error
    assert cost_per_day(at) > 0
    assert any("--tensor-parallel-size" in code.value for code in at.code)
    assert len(at.get("vega_lite_chart")) >= 1  # the timeline figure (M9: Altair, was st.image)


# 9.3 Infeasible surfaces cleanly: one error element, no traceback anywhere.
def test_9_3_infeasible_surfaces_cleanly() -> None:
    at = app()
    at.number_input(key="ttft").set_value(1.0)
    plan(at.run())
    assert len(at.error) == 1
    assert "0 of" in at.error[0].value
    assert not at.exception
    assert not any("Traceback" in text for text in texts(at))


# 9.4 Upload validation: the WorkloadFormatError message, and 51 MB refused before parsing.
def test_9_4_upload_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    at = app()
    at.radio(key="traffic_mode").set_value("Upload CSV")
    at.run()
    both = b"arrival_s,timestamp,input_tokens,output_tokens\n0,2024-01-01T00:00:00Z,10,5\n"
    at.file_uploader(key="upload").set_value(("both.csv", both, "text/csv"))
    at.run()
    assert [e.value for e in at.error] == [
        "uploaded.csv: has both 'arrival_s' and 'timestamp' columns; keep exactly one"
    ]

    calls: list[object] = []
    real = llmplan.workload.load_workload
    monkeypatch.setattr(
        llmplan.workload, "load_workload", lambda *a, **k: calls.append(a) or real(*a, **k)
    )
    big = b"0" * 51_000_000
    at.file_uploader(key="upload").set_value(("big.csv", big, "text/csv"))
    at.run()
    assert calls == []  # refused before any parsing
    (error,) = at.error
    assert error.value.startswith("Upload refused: 51,000,000 bytes is over the 50 MB")
    assert not at.exception


# 9.5 Price edit flows through: H100 price x10 on an H100-only plan. Documented assertion:
# the cost increases (by exactly 10x: every candidate row is scaled alike, so the same
# fleet stays optimal).
def test_9_5_price_edit_flows_through() -> None:
    at = preset_session(H100)
    before = cost_per_day(plan(at))
    table = at.session_state["price_table"].copy()
    table.loc[table["gpu_id"] == H100, "price_usd_per_hour"] *= 10
    at.session_state["price_table"] = table
    after = cost_per_day(plan(at.run()))
    assert not at.error
    assert after > before
    assert after == pytest.approx(10 * before, rel=1e-3)  # rel: metric rounds to cents


# 9.6 Usage log: one plan run appends one JSON line with exactly the section 7 fields.
SECTION_7_FIELDS = {
    "ts",
    "request_id",
    "model_id",
    "gpu_ids",
    "n_requests",
    "peak_rps",
    "slo",
    "cost_usd_per_day",
    "baseline_usd_per_day",
    "solver_status",
    "duration_s",
}


def test_9_6_usage_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("LLMPLAN_USAGE_LOG", str(path))
    at = preset_session(H100, "l4-24gb")
    plan(at)
    assert not at.error
    (line,) = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert set(record) == SECTION_7_FIELDS
    assert record["model_id"] == "fixture:llama3-8b"
    assert record["gpu_ids"] == [H100, "l4-24gb"]
    assert record["cost_usd_per_day"] == pytest.approx(cost_per_day(at), abs=0.005)
    assert record["solver_status"] == "optimal"


# 9.7 Docker build. Excluded by default (minutes, and it downloads the base image and the
# locked packages); run with `uv run pytest -m docker --no-cov`. Skipped without Docker.
@pytest.mark.docker
@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is not installed")
def test_9_7_docker_build() -> None:
    root = APP.parents[2]
    build = subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["docker", "build", "-t", "llmplan:test", str(root)],  # noqa: S607  # docker on PATH
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert build.returncode == 0, build.stderr[-2000:]
