from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmplan.catalog.hardware import DEFAULT_GPUS_PATH, DEFAULT_PRICES_PATH
from llmplan.cli import app
from llmplan.planner import solve as solve_module

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
TEN = str(FIXTURES / "workload_10.csv")
BASE = ["plan", "--model", "fixture:llama3-8b", "--max-model-len", "8192"]
ROOFLINE = [*BASE, "--trace", TEN, "--perf-backend", "roofline"]
runner = CliRunner()


def test_help_lists_plan() -> None:
    assert "plan" in runner.invoke(app, ["--help"]).stdout
    assert "--stats-json" in runner.invoke(app, ["plan", "--help"]).stdout


def test_plan_text_on_shipped_catalogs() -> None:
    result = runner.invoke(app, ROOFLINE)
    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert out.startswith("Model     fixture:llama3-8b  (max_model_len 8,192)\n")
    assert "Fleet" in out
    assert "vllm serve <MODEL_ID> --tensor-parallel-size" in out
    assert "Candidates (top 10 of 256 by $/hour per req/s)" in out


def test_plan_json_is_deterministic() -> None:
    first = runner.invoke(app, [*ROOFLINE, "--format", "json"])
    second = runner.invoke(app, [*ROOFLINE, "--format", "json"])
    assert first.exit_code == 0
    assert first.stdout == second.stdout
    doc = json.loads(first.stdout)
    assert doc["request"]["options"]["tensor_parallel_choices"] == [1, 2, 4, 8]
    assert doc["cost_usd_per_day"] > 0


def test_plan_vllm_format_and_filters() -> None:
    args = [*ROOFLINE, "--gpus", "h100-sxm-80gb", "--providers", "lambda", "--tp", "1"]
    result = runner.invoke(
        app, [*args, "--dtypes", "bf16", "--max-num-seqs", "64", "--format", "vllm"]
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == [
        "# 1 replica(s) on 1 x lambda gpu_1x_h100_sxm5 (1 x h100-sxm-80gb per instance)",
        "vllm serve <MODEL_ID> --tensor-parallel-size 1 --max-num-seqs 64 --max-model-len 8192 "
        "--gpu-memory-utilization 0.9 --dtype bfloat16",
    ]


def test_plan_from_stats_json(tmp_path: Path) -> None:
    stats = runner.invoke(app, ["workload", "stats", "--trace", TEN, "--format-out", "json"])
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(stats.stdout)
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(json.loads(stats.stdout)["stats"]))
    outputs = [
        runner.invoke(app, [*BASE, "--stats-json", str(path), "--perf-backend", "roofline"])
        for path in (wrapped, bare)
    ]
    assert [o.exit_code for o in outputs] == [0, 0]
    trace_output = runner.invoke(app, ROOFLINE).stdout
    strip = [line for line in trace_output.splitlines() if not line.startswith("Solver")]
    for output in outputs:
        assert [
            line for line in output.stdout.splitlines() if not line.startswith("Solver")
        ] == strip


def test_plan_custom_catalogs(tmp_path: Path) -> None:
    gpus = [g for g in yaml.safe_load(DEFAULT_GPUS_PATH.read_text()) if g["id"] == "l40s-48gb"]
    prices = [
        p for p in yaml.safe_load(DEFAULT_PRICES_PATH.read_text()) if p["gpu_id"] == "l40s-48gb"
    ]
    gpu_file, price_file = tmp_path / "gpus.yaml", tmp_path / "prices.yaml"
    gpu_file.write_text(yaml.safe_dump(gpus))
    price_file.write_text(yaml.safe_dump(prices))
    args = [
        *ROOFLINE,
        "--gpu-catalog",
        str(gpu_file),
        "--prices",
        str(price_file),
        "--format",
        "json",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert {item["price_row"]["gpu_id"] for item in json.loads(result.stdout)["fleet"]} == {
        "l40s-48gb"
    }


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        ([*BASE], 2, "pass exactly one of --trace or --stats-json"),
        ([*ROOFLINE, "--stats-json", TEN], 2, "pass exactly one of --trace or --stats-json"),
        ([*ROOFLINE, "--tp", "1,x"], 2, "--tp must be a comma-separated list"),
        ([*ROOFLINE, "--dtypes", "bf17"], 2, "invalid dtype_choices"),
        ([*ROOFLINE, "--utilization", "0"], 2, "invalid utilization_target"),
        ([*ROOFLINE, "--gpus", "b200"], 3, "unknown gpu id 'b200'"),
        ([*ROOFLINE, "--ttft-p95-ms", "0.001"], 4, "0 of 256 candidates eligible"),
    ],
)
def test_plan_errors(args: list[str], code: int, message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == code
    assert message in result.stderr


def test_plan_unavailable_solver_exits_5(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solve_module, "available", lambda backend: False)
    result = runner.invoke(app, [*ROOFLINE, "--solver", "gurobi"])
    assert result.exit_code == 5
    assert "backend not available" in result.stderr


def test_stats_json_errors(tmp_path: Path) -> None:
    missing = runner.invoke(app, [*BASE, "--stats-json", str(tmp_path / "nope.json")])
    assert missing.exit_code == 2
    assert "is not readable" in missing.stderr
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert "is not valid JSON" in runner.invoke(app, [*BASE, "--stats-json", str(bad)]).stderr
    big = tmp_path / "big.json"
    big.write_text(" " * 1_000_001)
    assert "larger than" in runner.invoke(app, [*BASE, "--stats-json", str(big)]).stderr
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"n_requests": 1}))
    invalid = runner.invoke(app, [*BASE, "--stats-json", str(wrong)])
    assert invalid.exit_code == 2
    assert "error: invalid" in invalid.stderr
