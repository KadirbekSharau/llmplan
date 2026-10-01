"""M8 acceptance tests: M8_DESIGN.md section 9. These define done; do not relax them.

9.11 (live Hugging Face checks) is manual: tests/live/test_hf_configs.py, marked `network`,
with the recorded run in docs/milestones/M8_NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llmplan import render
from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import load_model
from llmplan.cli import app
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.workload import compute_stats, load_workload
from tests.conftest import UseFakePerf
from tests.fake_planner import FakePerf, request

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ROOFLINE_SENTENCE = "roofline (uncalibrated first-principles model; expect ±30% on throughput)"


def confidence_line(text: str) -> str:
    (line,) = [line for line in text.splitlines() if line.startswith("Confidence")]
    return line


# 9.1 Confidence everywhere.
def test_9_1_fake_measured_plan_reports_measured(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)})
    req = request(34)
    result = plan(req)
    assert result.perf_confidence == "measured"
    assert result.perf_sources == ()
    line = confidence_line(render.get("text").plan(req, result))
    assert line.startswith("Confidence  measured (")


def test_9_1_real_catalog_plan_is_roofline() -> None:
    req = PlanRequest(  # M4 test 10.9
        model=load_model("fixture:llama3-8b"),
        stats=compute_stats(load_workload(FIXTURES / "workload_10.csv")),
        slo=SLO(),
        engine=EngineProfile(),
        options=PlanOptions(max_model_len=8192, perf_backend="roofline"),
        gpus=load_gpus(),
        prices=load_prices(),
    )
    result = plan(req)
    assert result.perf_confidence == "roofline"
    line = confidence_line(render.get("text").plan(req, result))
    assert line == f"Confidence  {ROOFLINE_SENTENCE}"


def test_9_1_cli_plan_text_has_the_confidence_line() -> None:
    out = CliRunner().invoke(
        app,
        [
            "plan",
            "--model",
            "fixture:llama3-8b",
            "--trace",
            str(FIXTURES / "workload_10.csv"),
            "--max-model-len",
            "8192",
            "--perf-backend",
            "roofline",
        ],
    )
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Cost"))
    assert lines[index + 1] == f"Confidence  {ROOFLINE_SENTENCE}"  # directly under Cost


def test_9_1_two_confidences_are_mixed(fake_perf: UseFakePerf) -> None:
    fake_perf(
        {
            ("fake-a", 1): FakePerf(rps=3, confidence="measured"),
            ("fake-b", 1): FakePerf(rps=4, confidence="roofline"),
        }
    )
    req = request(34)
    result = plan(req)  # 1 x B (8 GPUs) + 1 x A, as in M4 test 10.1
    assert {r.candidate.perf.confidence for r in result.replicas if r.candidate.perf} == {
        "measured",
        "roofline",
    }
    assert result.perf_confidence == "mixed"
    assert confidence_line(render.get("text").plan(req, result)).startswith("Confidence  mixed (")
