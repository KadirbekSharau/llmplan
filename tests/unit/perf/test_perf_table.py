from __future__ import annotations

import math
from typing import Any

import pytest

from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import load_model
from llmplan.perf import ReplicaConfig
from llmplan.perf.benchmarks import BenchmarkRow, BenchmarkTable
from llmplan.perf.table import TableBackend, log_interp
from tests.unit.perf.stats import FakeStats

H100 = load_gpus()["h100-sxm-80gb"]
LLAMA8 = load_model("fixture:llama3-8b")
LLAMA70 = load_model("fixture:llama3-70b")
STATS = FakeStats()


def row(concurrency: int, tokens_per_s: float, **overrides: Any) -> BenchmarkRow:
    base: dict[str, Any] = {
        "model_id": "org/llama-8b",
        "gpu_id": "h100-sxm-80gb",
        "engine": "vllm",
        "engine_version": "test",
        "tensor_parallel": 1,
        "dtype": "bf16",
        "concurrency": concurrency,
        "input_len": 512,
        "output_len": 256,
        "output_tokens_per_s": tokens_per_s,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "tpot_ms_p50": None,
        "tpot_ms_p95": None,
        "source_url": "https://example.com/a",
        "as_of": "2026-09-30",
    }
    return BenchmarkRow.model_validate({**base, **overrides})


def backend(*rows: BenchmarkRow) -> TableBackend:
    table = BenchmarkTable(rows=rows, aliases={"fixture:llama3-8b": "org/llama-8b"})
    return TableBackend(table)


def config(max_num_seqs: int = 256, **kw: Any) -> ReplicaConfig:
    return ReplicaConfig(max_model_len=8192, max_num_seqs=max_num_seqs, **kw)


def test_log_interp() -> None:
    assert log_interp(4, 2, 8, 10.0, 20.0) == pytest.approx(15.0)


def test_latencies_interpolated_and_used() -> None:
    lat = {"ttft_ms_p50": 100.0, "ttft_ms_p95": 200.0, "tpot_ms_p50": 10.0, "tpot_ms_p95": 20.0}
    hi = {k: 2 * v for k, v in lat.items()}
    b = backend(row(8, 800.0, source_url="https://example.com/b", **lat), row(32, 2000.0, **hi))
    est = b.estimate(LLAMA8, H100, config(16), STATS)
    assert est is not None
    assert est.tpot_ms_p50 == pytest.approx(15.0)
    assert est.tpot_ms_p95 == pytest.approx(30.0)
    assert est.prefill_tokens_per_s == pytest.approx(512 / 0.150)
    assert est.ttft_ms_p50 == pytest.approx(400 / (512 / 0.150) * 1e3)
    assert est.source_urls == ("https://example.com/a", "https://example.com/b")
    service_s = 512 / est.prefill_tokens_per_s + 256 * 0.015
    assert est.requests_per_s_capacity == pytest.approx(16 / service_s)


def test_measured_row_latencies_taken_verbatim() -> None:
    b = backend(row(8, 800.0, tpot_ms_p50=9.0, tpot_ms_p95=12.0))
    est = b.estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.confidence == "measured"
    assert (est.tpot_ms_p50, est.tpot_ms_p95) == (9.0, 12.0)


def test_derived_tpot_without_latencies() -> None:
    est = backend(row(8, 800.0), row(64, 3200.0)).estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.tpot_ms_p50 == pytest.approx(10.0)
    assert est.tpot_ms_p95 == pytest.approx(15.0)
    assert any("derived" in note for note in est.assumptions)


def test_below_smallest_concurrency_clamps_per_sequence_rate() -> None:
    est = backend(row(8, 800.0), row(64, 3200.0)).estimate(LLAMA8, H100, config(2), STATS)
    assert est is not None
    assert est.effective_batch == 2
    assert est.confidence == "interpolated"
    assert est.decode_tokens_per_s == pytest.approx(200.0)
    assert any("clamped" in note for note in est.assumptions)


def test_above_largest_concurrency_caps_batch() -> None:
    est = backend(row(8, 800.0), row(64, 3200.0)).estimate(LLAMA8, H100, config(256), STATS)
    assert est is not None
    assert est.effective_batch == 64
    assert est.confidence == "measured"
    assert any("capped at the largest benchmarked 64" in note for note in est.assumptions)


def test_duplicate_concurrency_keeps_lowest_throughput() -> None:
    b = backend(row(8, 900.0), row(8, 800.0, source_url="https://example.com/b"))
    est = b.estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.decode_tokens_per_s == 800.0
    assert est.source_urls == ("https://example.com/b",)


def test_vllm_rows_preferred_over_other_engines() -> None:
    b = backend(row(8, 800.0), row(8, 700.0, engine="nim", engine_version="1.0"))
    est = b.estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.decode_tokens_per_s == 800.0
    assert any("nim, vllm; vllm preferred" in note for note in est.assumptions)
    only_nim = backend(row(8, 700.0, engine="nim", engine_version="1.0"))
    est = only_nim.estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.decode_tokens_per_s == 700.0


def test_nearest_shape_chosen_in_log_space() -> None:
    b = backend(row(8, 800.0, input_len=1000, output_len=1000), row(8, 700.0, input_len=600))
    est = b.estimate(LLAMA8, H100, config(8), STATS)
    assert est is not None
    assert est.decode_tokens_per_s == 700.0
    assert math.isclose(est.effective_batch, 8)


@pytest.mark.parametrize(
    ("model", "cfg", "stats", "message"),
    [
        (LLAMA70, config(8), STATS, "no benchmark rows for fixture:llama3-70b"),
        (LLAMA8, config(8, dtype="fp8"), STATS, "dtype fp8"),
        (LLAMA8, config(8), FakeStats(output_tokens_mean=4096), "more than 2x"),
        (LLAMA8, config(8, gpu_memory_utilization=0.2), STATS, "does not fit"),
    ],
)
def test_no_answer_reasons(model: Any, cfg: ReplicaConfig, stats: FakeStats, message: str) -> None:
    b = backend(row(8, 800.0))
    assert b.estimate(model, H100, cfg, stats) is None
    assert message in b.explain(model, H100, cfg, stats)


def test_no_ttft_and_no_tflops_cannot_answer() -> None:
    gpu = GPUSpec.model_validate({**H100.model_dump(), "fp16_dense_tflops": None})
    b = backend(row(8, 800.0))
    assert b.estimate(LLAMA8, gpu, config(8), STATS) is None
    assert "fp16_dense_tflops null" in b.explain(LLAMA8, gpu, config(8), STATS)


def test_explain_when_answer_exists() -> None:
    assert backend(row(8, 800.0)).explain(LLAMA8, H100, config(8), STATS) == (
        "an estimate is available"
    )


def test_default_table_is_shipped_table() -> None:
    assert TableBackend().table.aliases["fixture:llama3-8b"] == "meta-llama/Llama-3.1-8B-Instruct"
