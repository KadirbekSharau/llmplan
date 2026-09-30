"""M3 acceptance tests: M3_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import load_model
from llmplan.errors import BenchmarkError, PerfError
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, estimate
from llmplan.perf.benchmarks import BenchmarkRow, load_benchmarks, physical_floor_s
from llmplan.perf.roofline import RooflineBackend
from llmplan.perf.table import TableBackend
from llmplan.workload.schema import WorkloadStats

STATS = WorkloadStats(
    n_requests=1000,
    duration_s=600.0,
    window_s=60.0,
    n_windows=10,
    mean_rps=1000 / 600,
    peak_window_rps=3.0,
    peak_window_index=4,
    input_tokens_p50=400,
    input_tokens_p95=1500,
    input_tokens_p99=3000,
    input_tokens_mean=512,
    input_tokens_max=8000,
    output_tokens_p50=200,
    output_tokens_p95=800,
    output_tokens_p99=1200,
    output_tokens_mean=256,
    output_tokens_max=2000,
    peak_input_tokens_per_s=1536.0,
    peak_output_tokens_per_s=768.0,
    hourly_rps=None,
)

LLAMA70 = load_model("fixture:llama3-70b")
LLAMA8 = load_model("fixture:llama3-8b")
H100 = load_gpus()["h100-sxm-80gb"]


def _config(tp: int, max_num_seqs: int = 256, max_model_len: int = 8192) -> ReplicaConfig:
    return ReplicaConfig(tensor_parallel=tp, max_num_seqs=max_num_seqs, max_model_len=max_model_len)


def test_stats_stand_in_satisfies_protocol() -> None:
    assert isinstance(STATS, StatsLike)


# 9.1 Roofline, llama3-70b bf16, h100-sxm-80gb, tp=4
def test_9_1_roofline_llama70b_h100_tp4() -> None:
    est = RooflineBackend().estimate(LLAMA70, H100, _config(4), STATS)
    assert isinstance(est, PerfEstimate)
    assert est.backend == "roofline"
    assert est.confidence == "roofline"
    assert est.source_urls == ()
    assert est.effective_batch == 256
    rel = 2e-2
    assert est.tpot_ms_p50 == pytest.approx(30.44, rel=rel)
    assert est.tpot_ms_p95 == pytest.approx(45.66, rel=rel)
    assert est.decode_tokens_per_s == pytest.approx(8411, rel=rel)
    assert est.prefill_tokens_per_s == pytest.approx(14_018, rel=rel)
    assert est.ttft_ms_p50 == pytest.approx(28.5, rel=rel)
    assert est.ttft_ms_p95 == pytest.approx(107.0, rel=rel)
    assert est.requests_per_s_capacity == pytest.approx(32.7, rel=rel)
    assert any("compute-bound" in note for note in est.assumptions)
    # The facade with the roofline backend returns the same estimate.
    assert estimate(LLAMA70, H100, _config(4), STATS, backend="roofline") == est


# 9.2 Roofline, tp=1: does not fit
def test_9_2_roofline_does_not_fit() -> None:
    assert RooflineBackend().estimate(LLAMA70, H100, _config(1), STATS) is None
    with pytest.raises(PerfError, match="does not fit"):
        estimate(LLAMA70, H100, _config(1), STATS, backend="roofline")


# 9.3 Roofline requires catalog fields
def test_9_3_roofline_requires_bandwidth() -> None:
    gpu = GPUSpec.model_validate(
        {**H100.model_dump(), "id": "h100-no-bandwidth", "memory_bandwidth_gbps": None}
    )
    assert RooflineBackend().estimate(LLAMA70, gpu, _config(4), STATS) is None
    with pytest.raises(PerfError, match="memory_bandwidth_gbps null"):
        estimate(LLAMA70, gpu, _config(4), STATS, backend="roofline")
    with pytest.raises(PerfError) as info:
        estimate(LLAMA70, gpu, _config(4), STATS, backend="auto")
    message = str(info.value)
    assert "table: no benchmark rows" in message
    assert "roofline: gpu h100-no-bandwidth has memory_bandwidth_gbps null" in message


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": "fixture:llama3-70b",
        "gpu_id": "h100-sxm-80gb",
        "engine": "vllm",
        "engine_version": "0.0.0-test",
        "tensor_parallel": 4,
        "dtype": "bf16",
        "concurrency": 8,
        "input_len": 512,
        "output_len": 256,
        "output_tokens_per_s": 100.0,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "tpot_ms_p50": None,
        "tpot_ms_p95": None,
        "source_url": "https://example.com/test-only",
        "as_of": "2026-09-30",
    }
    return {**row, **overrides}


def _write_table(directory: Path, rows: list[dict[str, Any]]) -> Path:
    (directory / "h100-sxm-80gb.yaml").write_text(yaml.safe_dump(rows), encoding="utf-8")
    return directory


# 9.4 Table validation: a row 10x faster than the physical floor is rejected
def test_9_4_row_above_physical_bound_rejected(tmp_path: Path) -> None:
    floor_s = physical_floor_s(BenchmarkRow.model_validate(_row()), LLAMA70, H100)
    assert floor_s is not None
    too_fast = 10 * 8 / floor_s  # 10x the highest physically possible output tokens/s
    with pytest.raises(BenchmarkError) as info:
        load_benchmarks(_write_table(tmp_path, [_row(output_tokens_per_s=too_fast)]))
    assert "row index 0" in str(info.value)
    assert "fixture:llama3-70b" in str(info.value)
    # The same row at a physically possible rate loads.
    ok = load_benchmarks(_write_table(tmp_path, [_row(output_tokens_per_s=0.5 * 8 / floor_s)]))
    assert len(ok.rows) == 1


# 9.6 Seed data sanity
def test_9_6_shipped_rows_are_valid_and_sourced() -> None:
    table = load_benchmarks()  # raises BenchmarkError if any row fails the physical bound
    assert table.rows
    for row in table.rows:
        assert row.source_url.startswith("https://")
        assert row.as_of is not None
    # Section 5.4 row groups: Llama-3.1-70B on H100 at tp 4 or 8 with two or more
    # concurrencies at one shape, and Llama-3.1-8B on H100 at tp 1.
    llama70 = {
        (r.input_len, r.output_len, r.concurrency)
        for r in table.rows
        if r.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        and r.gpu_id == "h100-sxm-80gb"
        and r.tensor_parallel in (4, 8)
        and r.dtype in ("bf16", "fp8")
    }
    shapes = {(i, o) for i, o, _ in llama70}
    assert any(sum((i, o) == shape for i, o, _ in llama70) >= 2 for shape in shapes)
    assert any(
        r.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        and r.gpu_id == "h100-sxm-80gb"
        and r.tensor_parallel == 1
        for r in table.rows
    )


# 9.5 Table interpolation
def test_9_5_table_interpolation(tmp_path: Path) -> None:
    rows = [
        _row(
            model_id="fixture:llama3-8b",
            tensor_parallel=1,
            concurrency=8,
            output_tokens_per_s=1000.0,
        ),
        _row(
            model_id="fixture:llama3-8b",
            tensor_parallel=1,
            concurrency=64,
            output_tokens_per_s=6000.0,
        ),
    ]
    backend = TableBackend(load_benchmarks(_write_table(tmp_path, rows)))
    backends = {"table": backend}

    est = backend.estimate(LLAMA8, H100, _config(1, max_num_seqs=32), STATS)
    assert est is not None
    assert est.backend == "table"
    assert est.confidence == "interpolated"
    assert est.effective_batch == 32
    assert 1000.0 < est.decode_tokens_per_s < 6000.0
    weight = math.log(32 / 8) / math.log(64 / 8)
    assert est.decode_tokens_per_s == pytest.approx(1000.0 + weight * 5000.0, rel=1e-6)
    assert est.source_urls == ("https://example.com/test-only",)

    measured = backend.estimate(LLAMA8, H100, _config(1, max_num_seqs=64), STATS)
    assert measured is not None
    assert measured.confidence == "measured"
    assert measured.decode_tokens_per_s == 6000.0

    long_inputs = STATS.model_copy(update={"input_tokens_mean": 8192.0})
    assert backend.estimate(LLAMA8, H100, _config(1, max_num_seqs=32), long_inputs) is None
    fallback = estimate(LLAMA8, H100, _config(1, max_num_seqs=32), long_inputs, backends=backends)
    assert fallback.backend == "roofline"
    assert fallback.confidence == "roofline"


# 9.7 Determinism
def test_9_7_identical_inputs_give_identical_json(tmp_path: Path) -> None:
    for backend in ("auto", "roofline"):
        first = estimate(LLAMA70, H100, _config(4), STATS, backend=backend)
        second = estimate(LLAMA70, H100, _config(4), STATS, backend=backend)
        assert first.model_dump_json() == second.model_dump_json()
    rows = [
        _row(model_id="fixture:llama3-8b", tensor_parallel=1, concurrency=c, output_tokens_per_s=t)
        for c, t in ((8, 1000.0), (64, 6000.0))
    ]
    backends = {"table": TableBackend(load_benchmarks(_write_table(tmp_path, rows)))}
    config = _config(1, max_num_seqs=32)
    first = estimate(LLAMA8, H100, config, STATS, backends=backends)
    second = estimate(LLAMA8, H100, config, STATS, backends=backends)
    assert first.backend == "table"
    assert first.model_dump_json() == second.model_dump_json()
