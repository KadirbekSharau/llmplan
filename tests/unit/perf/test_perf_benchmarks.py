from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import load_model
from llmplan.errors import BenchmarkError, CatalogError
from llmplan.perf.benchmarks import (
    BenchmarkRow,
    default_table,
    load_benchmarks,
    physical_floor_s,
)

GPUS = load_gpus()
LLAMA8 = load_model("fixture:llama3-8b")


def row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_id": "fixture:llama3-8b",
        "gpu_id": "h100-sxm-80gb",
        "engine": "vllm",
        "engine_version": "test",
        "tensor_parallel": 1,
        "dtype": "bf16",
        "concurrency": 8,
        "input_len": 512,
        "output_len": 256,
        "output_tokens_per_s": 500.0,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "tpot_ms_p50": None,
        "tpot_ms_p95": None,
        "source_url": "https://example.com/test",
        "as_of": "2026-09-30",
    }
    return {**base, **overrides}


def write(tmp: Path, rows: Any, name: str = "h100-sxm-80gb.yaml", aliases: Any = None) -> Path:
    (tmp / name).write_text(yaml.safe_dump(rows), encoding="utf-8")
    if aliases is not None:
        (tmp / "aliases.yaml").write_text(yaml.safe_dump(aliases), encoding="utf-8")
    return tmp


def test_benchmark_error_is_a_catalog_error() -> None:
    assert issubclass(BenchmarkError, CatalogError)


def test_valid_row_and_alias_resolution(tmp_path: Path) -> None:
    aliases = {"fixture:llama3-8b": "org/llama-8b", "org/old-name": "org/llama-8b"}
    table = load_benchmarks(write(tmp_path, [row(model_id="org/old-name")], aliases=aliases))
    assert table.rows[0].model_id == "org/old-name"
    assert table.canonical("fixture:llama3-8b") == "org/llama-8b"
    assert table.canonical("org/unknown") == "org/unknown"


@pytest.mark.parametrize(
    ("rows", "name", "message"),
    [
        ({"not": "a list"}, "h100-sxm-80gb.yaml", "expected a list"),
        ([row(concurrency=0)], "h100-sxm-80gb.yaml", "field 'concurrency'"),
        (["scalar"], "h100-sxm-80gb.yaml", "row index 0 (model_id None)"),
        ([row(gpu_id="b200")], "b200.yaml", "unknown gpu_id 'b200'"),
        ([row(gpu_id="l4-24gb")], "h100-sxm-80gb.yaml", "does not match file"),
        ([row(model_id="org/nofixture")], "h100-sxm-80gb.yaml", "no fixture for this model"),
        ([row(model_id="fixture:missing")], "h100-sxm-80gb.yaml", "fixture 'missing' not found"),
        ([row(tensor_parallel=3)], "h100-sxm-80gb.yaml", "tensor_parallel 3 does not divide"),
        ([row(output_tokens_per_s=1e6)], "h100-sxm-80gb.yaml", "below the physical floor"),
    ],
)
def test_bad_rows_rejected(tmp_path: Path, rows: Any, name: str, message: str) -> None:
    with pytest.raises(BenchmarkError) as info:
        load_benchmarks(write(tmp_path, rows, name))
    assert message in str(info.value)


def test_gpu_without_bandwidth_or_tflops_rejected(tmp_path: Path) -> None:
    bare = GPUSpec.model_validate(
        {
            **GPUS["h100-sxm-80gb"].model_dump(),
            "memory_bandwidth_gbps": None,
            "fp16_dense_tflops": None,
        }
    )
    with pytest.raises(BenchmarkError, match="neither memory_bandwidth_gbps nor dense TFLOPS"):
        load_benchmarks(write(tmp_path, [row()]), gpus={"h100-sxm-80gb": bare})


def test_floor_uses_available_terms() -> None:
    parsed = BenchmarkRow.model_validate(row())
    both = physical_floor_s(parsed, LLAMA8, GPUS["h100-sxm-80gb"])
    no_bw = GPUSpec.model_validate(
        {**GPUS["h100-sxm-80gb"].model_dump(), "memory_bandwidth_gbps": None}
    )
    compute_only = physical_floor_s(parsed, LLAMA8, no_bw)
    assert both is not None and compute_only is not None
    assert compute_only <= both


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        (["a", "list"], "expected a mapping"),
        ({"a": "b", "b": "c"}, "chains to another alias"),
    ],
)
def test_bad_aliases_rejected(tmp_path: Path, aliases: Any, message: str) -> None:
    with pytest.raises(BenchmarkError, match=message):
        load_benchmarks(write(tmp_path, [], aliases=aliases))


def test_unreadable_and_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "h100-sxm-80gb.yaml").write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="invalid YAML"):
        load_benchmarks(tmp_path)
    (tmp_path / "h100-sxm-80gb.yaml").unlink()
    (tmp_path / "aliases.yaml").mkdir()
    with pytest.raises(BenchmarkError, match="cannot read"):
        load_benchmarks(tmp_path)


def test_default_table_is_cached() -> None:
    assert default_table() is default_table()
