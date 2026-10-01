from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from llmplan.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TEN = str(FIXTURES / "workload_10.csv")
FIFTY = str(FIXTURES / "workload_csv_50.csv")
PLAN = ["plan", "--model", "fixture:llama3-8b", "--max-model-len", "8192", "--trace", TEN]
PLAN_JSON = [*PLAN, "--perf-backend", "roofline", "--format", "json"]
runner = CliRunner()


def _plan_file(tmp_path: Path, *extra: str) -> str:
    result = runner.invoke(app, [*PLAN_JSON, *extra])
    assert result.exit_code == 0, result.stderr
    path = tmp_path / "plan.json"
    path.write_text(result.stdout, encoding="utf-8")
    return str(path)


def test_help_lists_simulate() -> None:
    assert "simulate" in runner.invoke(app, ["--help"]).stdout
    assert "--routing" in runner.invoke(app, ["simulate", "--help"]).stdout


def test_simulate_text_uses_the_plan_slo(tmp_path: Path) -> None:
    plan = _plan_file(tmp_path, "--ttft-p95-ms", "500")
    result = runner.invoke(app, ["simulate", "--plan", plan, "--trace", FIFTY])
    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert out.startswith("Requests  50 simulated (0 truncated)\n")
    assert "budget 500 ms" in out
    assert "(no budget)" in out
    assert "1 replica," in out
    assert "Assumptions" in out


def test_simulate_json_flags_override_and_png(tmp_path: Path) -> None:
    plan = _plan_file(tmp_path, "--ttft-p95-ms", "500")
    png = tmp_path / "timeline.png"
    args = ["simulate", "--plan", plan, "--trace", FIFTY, "--format", "json", "--png", str(png)]
    flags = ["--window", "30", "--routing", "round_robin", "--ttft-p95-ms", "900"]
    first = runner.invoke(app, [*args, *flags, "--tpot-p95-ms", "50"])
    second = runner.invoke(app, [*args, *flags, "--tpot-p95-ms", "50"])
    assert first.exit_code == 0, first.stderr
    assert first.stdout == second.stdout
    doc = json.loads(first.stdout)
    assert doc["options"]["window_s"] == 30.0
    assert doc["options"]["routing"] == "round_robin"
    assert doc["options"]["ttft_budget_ms"] == 900.0
    assert doc["options"]["tpot_budget_ms"] == 50.0
    assert doc["summary"]["n_requests"] == 50
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_simulate_errors_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["simulate", "--plan", TEN, "--trace", TEN])
    assert result.exit_code == 2
    assert "is not valid JSON" in result.stderr
    plan = _plan_file(tmp_path)
    bad_window = runner.invoke(app, ["simulate", "--plan", plan, "--trace", TEN, "--window", "0"])
    assert bad_window.exit_code == 2
    assert "window_s" in bad_window.stderr
