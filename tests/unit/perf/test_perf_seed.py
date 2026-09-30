from __future__ import annotations

import math

import pytest

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.perf import ReplicaConfig, estimate
from llmplan.perf.benchmarks import default_table
from tests.unit.perf.stats import FakeStats

GPUS = load_gpus()
CHAT_1K = FakeStats(
    input_tokens_mean=1000,
    input_tokens_p50=900,
    input_tokens_p95=2000,
    output_tokens_mean=1000,
    output_tokens_p50=900,
    output_tokens_p95=2000,
)


def test_shipped_rows_answer_a_matching_workload() -> None:
    model = load_model("fixture:llama3-70b")
    config = ReplicaConfig(tensor_parallel=4, dtype="fp8", max_model_len=8192, max_num_seqs=100)
    est = estimate(model, GPUS["h100-sxm-80gb"], config, CHAT_1K)
    assert est.backend == "table"
    assert est.confidence == "measured"
    assert est.effective_batch == 100
    assert est.decode_tokens_per_s == 3794.76  # NIM page, 1000/1000, concurrency 100
    assert est.source_urls == (
        "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html",
    )


def test_shipped_rows_interpolate_between_concurrencies() -> None:
    model = load_model("fixture:llama3-8b")
    config = ReplicaConfig(max_model_len=8192, max_num_seqs=64)
    est = estimate(model, GPUS["l40s-48gb"], config, CHAT_1K)
    assert est.backend == "table"
    assert est.confidence == "interpolated"
    # bf16 1000/1000 rows at concurrency 50 (1226.46 tok/s) and 100 (1613.38 tok/s)
    weight = math.log(64 / 50) / math.log(100 / 50)
    assert est.effective_batch == 64
    assert est.decode_tokens_per_s == pytest.approx(1226.46 + weight * (1613.38 - 1226.46))


def test_every_alias_target_has_rows() -> None:
    table = default_table()
    for target in table.aliases.values():
        assert any(row.model_id == target for row in table.rows)
