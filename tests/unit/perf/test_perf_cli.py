from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmplan.cli import app
from llmplan.cli_perf import STAT_FLAGS

runner = CliRunner()
TEN = str(Path(__file__).resolve().parents[2] / "fixtures" / "workload_10.csv")
STATS = [
    "--in-mean", "512", "--in-p50", "400", "--in-p95", "1500",
    "--out-mean", "256", "--out-p50", "200", "--out-p95", "800",
]  # fmt: skip
CHAT_1K = [
    "--in-mean", "1000", "--in-p50", "900", "--in-p95", "2000",
    "--out-mean", "1000", "--out-p50", "900", "--out-p95", "2000",
]  # fmt: skip
LLAMA70_TP4 = ["perf", "estimate", "--model", "fixture:llama3-70b", "--gpu", "h100-sxm-80gb"]


def test_help_lists_perf_commands() -> None:
    assert "perf" in runner.invoke(app, ["--help"]).stdout
    out = runner.invoke(app, ["perf", "--help"]).stdout
    assert "estimate" in out
    assert "benchmarks" in out


def test_estimate_text_roofline() -> None:
    result = runner.invoke(app, [*LLAMA70_TP4, "--tp", "4", *STATS])
    assert result.exit_code == 0
    out = result.stdout
    assert "Backend   roofline  (confidence: roofline)" in out
    assert "Decode    8,411 tokens/s   TPOT p50 30.44 ms, p95 45.66 ms" in out
    assert "Prefill   14,018 tokens/s   TTFT p50 28.54 ms, p95 107.01 ms" in out
    assert "Capacity  32.70 requests/s" in out
    assert "Sources" not in out


def test_estimate_json_table_is_deterministic() -> None:
    args = [
        "perf", "estimate", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb",
        "--dtype", "fp8", "--max-num-seqs", "64", *CHAT_1K, "--format", "json",
    ]  # fmt: skip
    first = runner.invoke(app, args)
    assert first.exit_code == 0
    assert first.stdout == runner.invoke(app, args).stdout
    doc = json.loads(first.stdout)
    assert doc["backend"] == "table"
    assert doc["confidence"] == "interpolated"
    assert doc["model"] == "fixture:llama3-8b"
    assert doc["config"]["max_num_seqs"] == 64
    assert doc["stats"]["output_tokens_p95"] == 2000
    assert doc["source_urls"] == [
        "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html"
    ]


def test_estimate_text_table_lists_sources() -> None:
    args = ["perf", "estimate", "--model", "fixture:llama3-8b", "--gpu", "l40s-48gb", *CHAT_1K]
    out = runner.invoke(app, [*args, "--backend", "table"]).stdout
    assert "Sources   https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html" in out


@pytest.mark.parametrize(
    ("extra", "code", "message"),
    [
        (["--tp", "4", "--trace", TEN, *STATS], 2, "pass either --trace or the token statistics"),
        (["--tp", "4", "--trace", "missing.csv"], 2, "not a readable file"),
        (["--tp", "4", "--in-mean", "512"], 2, "missing workload statistics: --in-p50"),
        (["--tp", "4", *STATS[:-1], "-1"], 2, "output_tokens_p95"),
        (["--tp", "4", "--max-model-len", "9000", *STATS], 2, "max_model_len 9000 exceeds"),
        (["--tp", "4", "--backend", "table", *STATS], 1, "table: no benchmark rows"),
        (STATS, 1, "does not fit"),
    ],
)
def test_estimate_errors(extra: list[str], code: int, message: str) -> None:
    result = runner.invoke(app, [*LLAMA70_TP4, *extra])
    assert result.exit_code == code
    assert message in result.stderr


def test_estimate_from_trace_matches_explicit_stats() -> None:
    from_trace = runner.invoke(app, [*LLAMA70_TP4, "--tp", "4", "--trace", TEN, "--format", "json"])
    assert from_trace.exit_code == 0, from_trace.stderr
    doc = json.loads(from_trace.stdout)
    expected = {
        "input_tokens_mean": 550.0,
        "input_tokens_p50": 550.0,
        "input_tokens_p95": 955.0,
        "output_tokens_mean": 55.0,
        "output_tokens_p50": 55.0,
        "output_tokens_p95": 95.5,
    }
    assert doc["stats"] == pytest.approx(expected)
    # The same statistics given as flags (repr round-trips floats exactly) give the same JSON.
    explicit = [
        arg for name, flag in STAT_FLAGS.items() for arg in (flag, repr(doc["stats"][name]))
    ]
    same = runner.invoke(app, [*LLAMA70_TP4, "--tp", "4", *explicit, "--format", "json"])
    assert same.stdout == from_trace.stdout


def test_estimate_unknown_gpu_exits_3() -> None:
    args = ["perf", "estimate", "--model", "fixture:llama3-8b", "--gpu", "b200", *STATS]
    result = runner.invoke(app, args)
    assert result.exit_code == 3
    assert "unknown gpu id 'b200'" in result.stderr


def test_benchmarks_text_filters_with_aliases() -> None:
    result = runner.invoke(
        app, ["perf", "benchmarks", "--gpu", "h100-sxm-80gb", "--model", "fixture:llama3-70b"]
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0].startswith("gpu")
    assert "32 rows" in lines
    assert all("Llama-3.1-70B" in line for line in lines[1:33])
    assert lines[-1].startswith("source: https://docs.nvidia.com/nim/")


def test_benchmarks_json_and_unknown_gpu() -> None:
    doc = json.loads(runner.invoke(app, ["perf", "benchmarks", "--format", "json"]).stdout)
    assert len(doc) == 105
    assert {row["gpu_id"] for row in doc} == {"h100-sxm-80gb", "l40s-48gb"}
    result = runner.invoke(app, ["perf", "benchmarks", "--gpu", "b200"])
    assert result.exit_code == 3
