from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmplan.cli import app, exit_code
from llmplan.errors import (
    CatalogError,
    FetchError,
    InfeasiblePlan,
    LLMPlanError,
    SolverError,
    UnknownRegistryKey,
    UnsupportedArchitecture,
    ValidationError,
)

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fit", "model-info", "gpus"):
        assert command in result.stdout


def test_fit_text_output_content() -> None:
    result = runner.invoke(
        app, ["fit", "--model", "fixture:llama3-70b", "--gpu", "h100-sxm-80gb", "--tp", "2"]
    )
    assert result.exit_code == 0
    out = result.stdout
    assert "fixture:llama3-70b  (llama_like, 70.55B params, GQA 64/8 heads)" in out
    assert "usable 76.97 GB/GPU (90% of 85.52 GB)" in out
    assert "141.11 GB total   70.55 GB/GPU" in out
    assert "3.22 GB/GPU  (1 GiB fixed + 2.15 GB activations)" in out
    assert "163,840 B/token/GPU" in out
    assert "capacity 19,493 tokens" in out
    assert "max 2 concurrent sequences" in out
    assert "FITS  (binding: ok" in out
    assert "tensor parallel: even weight split assumed" in out


def test_fit_json_includes_resolved_inputs() -> None:
    args = ["fit", "--model", "fixture:llama3-8b", "--gpu", "l4-24gb", "--format", "json"]
    result = runner.invoke(
        app, [*args, "--kv-dtype", "fp8", "--context", "4096", "--gpu-mem-util", "0.95"]
    )
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["model"]["id"] == "fixture:llama3-8b"
    assert doc["gpu"]["id"] == "l4-24gb"
    assert doc["engine"]["kv_dtype"] == "fp8"
    assert doc["engine"]["gpu_memory_utilization"] == 0.95
    assert doc["request"] == {
        "tensor_parallel": 1,
        "dtype": "bf16",
        "quantize_embeddings": False,
        "context_len": 4096,
    }
    assert doc["kv_bytes_per_token_per_gpu"] == 65_536


def test_fit_json_is_byte_identical_across_runs() -> None:
    args = ["fit", "--model", "fixture:qwen2.5-7b", "--gpu", "a10g-24gb", "--format", "json"]
    assert runner.invoke(app, args).stdout == runner.invoke(app, args).stdout


def test_fit_param_count_override() -> None:
    args = ["fit", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb", "--format", "json"]
    doc = json.loads(runner.invoke(app, [*args, "--param-count-override", "1000"]).stdout)
    assert doc["confidence"] == "estimated"
    assert doc["per_gpu_weight_bytes"] == 2000


def test_fit_text_does_not_fit() -> None:
    result = runner.invoke(app, ["fit", "--model", "fixture:llama3-70b", "--gpu", "l4-24gb"])
    assert result.exit_code == 0
    assert "DOES NOT FIT  (binding: weights" in result.stdout


@pytest.mark.parametrize(
    ("extra", "code", "message"),
    [
        (["--tp", "3"], 2, "tensor_parallel 3 must divide num_attention_heads 32"),
        (["--context", "9000"], 2, "context_len 9000 exceeds"),
        (["--gpu-mem-util", "1.5"], 2, "gpu_memory_utilization"),
        (["--param-count-override", "0"], 2, "param_count_override"),
        (["--dtype", "fp4"], 2, ""),
    ],
)
def test_fit_usage_errors(extra: list[str], code: int, message: str) -> None:
    args = ["fit", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb", *extra]
    result = runner.invoke(app, args)
    assert result.exit_code == code
    assert message in result.stderr


def test_catalog_errors_exit_3(tmp_path: Path) -> None:
    result = runner.invoke(app, ["model-info", "--model", "fixture:gpt2"])
    assert result.exit_code == 3
    assert "GPT2LMHeadModel" in result.stderr
    result = runner.invoke(app, ["model-info", "--model", "fixture:missing"])
    assert result.exit_code == 3
    result = runner.invoke(app, ["gpus", "--gpus", str(tmp_path / "none.yaml")])
    assert result.exit_code == 3


def test_model_info_text_and_json() -> None:
    text = runner.invoke(app, ["model-info", "--model", "fixture:mistral-7b-v0.1"])
    assert text.exit_code == 0
    assert "7,241,732,096 (7.24B)" in text.stdout
    assert "sliding_window 4,096" in text.stdout
    assert "131,072 B/token (bf16)" in text.stdout
    raw = runner.invoke(app, ["model-info", "--model", "fixture:llama3-70b", "--format", "json"])
    doc = json.loads(raw.stdout)
    assert doc["derived"]["param_count"] == 70_553_706_496
    assert doc["derived"]["weight_bytes_by_dtype"]["bf16"] == 141_107_412_992
    assert doc["kv_bytes_per_token_total"] == {"bf16": 327_680, "fp16": 327_680, "fp8": 163_840}


def test_gpus_text_and_json() -> None:
    text = runner.invoke(app, ["gpus"])
    assert text.exit_code == 0
    assert "h100-sxm-80gb" in text.stdout
    assert "a10g-24gb" in text.stdout
    doc = json.loads(runner.invoke(app, ["gpus", "--format", "json"]).stdout)
    assert [g["id"] for g in doc][:2] == ["h100-sxm-80gb", "h200-sxm-141gb"]
    assert doc[0]["as_of"] == "2026-09-30"


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (ValidationError("x"), 2),
        (UnknownRegistryKey("x"), 2),
        (CatalogError("x"), 3),
        (FetchError("x"), 3),
        (UnsupportedArchitecture("x", field="f"), 3),
        (InfeasiblePlan("x", reason="r"), 4),
        (SolverError("x"), 5),
        (LLMPlanError("x"), 1),
    ],
)
def test_exit_code_mapping(exc: LLMPlanError, code: int) -> None:
    assert exit_code(exc) == code
