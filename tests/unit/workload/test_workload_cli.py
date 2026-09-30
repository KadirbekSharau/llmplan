from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from llmplan.cli import app
from llmplan.cli_workload import stats_text
from llmplan.workload import Distribution, compute_stats, generate

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
TEN = str(FIXTURES / "workload_10.csv")
runner = CliRunner()


def test_help_lists_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "workload" in result.stdout
    assert "traces" in result.stdout
    sub = runner.invoke(app, ["workload", "--help"])
    assert "stats" in sub.stdout
    assert "synth" in sub.stdout


def test_stats_text() -> None:
    result = runner.invoke(app, ["workload", "stats", "--trace", TEN])
    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        f"trace           {TEN}  (format csv)",
        "requests        10  (0 rows dropped)",
        "duration        90 s",
        "windows         2 x 60 s",
        "mean rate       0.111 req/s",
        "peak rate       0.1 req/s  (window 0)",
        "input tokens    p50 550  p95 955  p99 991  mean 550  max 1,000",
        "output tokens   p50 55  p95 95.5  p99 99.1  mean 55  max 100",
        "peak input      56.67 tokens/s",
        "peak output     5.67 tokens/s",
        "hourly req/s    - (trace covers fewer than 24 hour windows)",
    ]


def test_stats_json_is_deterministic_and_complete() -> None:
    args = ["workload", "stats", "--trace", TEN, "--format", "csv", "--window", "30"]
    first = runner.invoke(app, [*args, "--format-out", "json"])
    second = runner.invoke(app, [*args, "--format-out", "json"])
    assert first.exit_code == 0
    assert first.stdout == second.stdout
    doc = json.loads(first.stdout)
    assert doc["format"] == "csv"
    assert doc["dropped_rows"] == 0
    assert doc["notes"] == []
    assert doc["stats"]["n_windows"] == 4  # the request at 90 s opens window [90, 120)
    assert doc["stats"]["peak_input_tokens_per_s"] == 2400 / 30  # 700 + 800 + 900


def test_stats_text_shows_hourly_profile_and_notes() -> None:
    fixed = Distribution(kind="fixed", value=10)
    workload = generate(
        rate_rps=0.5, duration_s=86_400, input_tokens=fixed, output_tokens=fixed, seed=3
    )
    lines = stats_text(workload, compute_stats(workload)).splitlines()
    assert lines[10].startswith("hourly req/s    00-05h  0.")
    assert lines[13].startswith("                18-23h  0.")
    assert lines[14].startswith("note            synthetic Poisson arrivals")


def test_stats_errors_exit_2(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["workload", "stats", "--trace", str(tmp_path / "x.csv")])
    assert missing.exit_code == 2
    assert "not a readable file" in missing.stderr
    zero = runner.invoke(app, ["workload", "stats", "--trace", TEN, "--window", "0"])
    assert zero.exit_code == 2
    assert "window_s" in zero.stderr


def test_synth_writes_loadable_csv(tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    args = ["workload", "synth", "--rps", "5", "--duration", "60", "--seed", "1"]
    result = runner.invoke(app, [*args, "--in-tokens", "fixed:100", "--out", str(out)])
    assert result.exit_code == 0
    first = out.read_bytes()
    assert result.stdout.startswith("wrote ")
    assert result.stdout.endswith(f" requests to {out} (seed 1)\n")
    runner.invoke(app, [*args, "--in-tokens", "fixed:100", "--out", str(out)])
    assert out.read_bytes() == first
    stats = runner.invoke(app, ["workload", "stats", "--trace", str(out), "--format-out", "json"])
    assert json.loads(stats.stdout)["stats"]["input_tokens_max"] == 100


def test_synth_errors_exit_2(tmp_path: Path) -> None:
    args = ["workload", "synth", "--rps", "5", "--duration", "60"]
    bad = runner.invoke(app, [*args, "--out-tokens", "gauss:1", "--out", str(tmp_path / "a.csv")])
    assert bad.exit_code == 2
    assert "invalid distribution 'gauss:1'" in bad.stderr
    unwritable = runner.invoke(app, [*args, "--out", str(tmp_path / "no" / "a.csv")])
    assert unwritable.exit_code == 2
    assert "cannot write" in unwritable.stderr
