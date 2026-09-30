from __future__ import annotations

import pytest

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.perf import ReplicaConfig
from llmplan.perf.roofline import (
    RooflineBackend,
    decode_compute_s,
    decode_memory_s,
    effective_batch,
    prefill_tokens_per_s,
)
from tests.unit.perf.stats import FakeStats

GPUS = load_gpus()
LLAMA8 = load_model("fixture:llama3-8b")
BACKEND = RooflineBackend()


def test_effective_batch_is_capped_by_kv_capacity_and_floor_of_one() -> None:
    assert effective_batch(469_612, 256, 640.0) == 256
    assert effective_batch(469_612, 1024, 640.0) == 733
    assert effective_batch(100, 256, 640.0) == 1
    assert effective_batch(10, 256, 0.0) == 10


def test_step_time_formulas() -> None:
    assert decode_memory_s(1000, 10, 2, 50.0, 1.0, efficiency=1.0) == pytest.approx(2000 / 1e9)
    assert decode_compute_s(10**12, 4, 2, 1.0, mfu=1.0) == pytest.approx(4.0)
    assert prefill_tokens_per_s(1.0, 1, 10**12) == pytest.approx(0.25)


def test_small_batch_is_memory_bound() -> None:
    config = ReplicaConfig(max_model_len=8192, max_num_seqs=1)
    est = BACKEND.estimate(LLAMA8, GPUS["h100-sxm-80gb"], config, FakeStats())
    assert est is not None
    assert est.effective_batch == 1
    assert any("memory-bound" in note for note in est.assumptions)
    assert not any("tensor parallel" in note for note in est.assumptions)
    assert est.tpot_ms_p95 == pytest.approx(est.tpot_ms_p50 * 1.5)
    assert BACKEND.explain(LLAMA8, GPUS["h100-sxm-80gb"], config, FakeStats()) == (
        "an estimate is available"
    )


def test_fp8_uses_fp8_tflops() -> None:
    bf16 = BACKEND.estimate(
        LLAMA8, GPUS["h100-sxm-80gb"], ReplicaConfig(max_model_len=8192), FakeStats()
    )
    fp8 = BACKEND.estimate(
        LLAMA8, GPUS["h100-sxm-80gb"], ReplicaConfig(max_model_len=8192, dtype="fp8"), FakeStats()
    )
    assert bf16 is not None and fp8 is not None
    assert fp8.prefill_tokens_per_s == pytest.approx(2 * bf16.prefill_tokens_per_s, rel=1e-3)


def test_fp8_on_ampere_has_no_tflops() -> None:
    config = ReplicaConfig(max_model_len=8192, dtype="fp8")
    gpu = GPUS["a100-sxm-80gb"]
    assert BACKEND.estimate(LLAMA8, gpu, config, FakeStats()) is None
    assert "fp8_dense_tflops null" in BACKEND.explain(LLAMA8, gpu, config, FakeStats())


def test_int4_notes_dequantization() -> None:
    config = ReplicaConfig(max_model_len=8192, dtype="int4")
    est = BACKEND.estimate(LLAMA8, GPUS["l4-24gb"], config, FakeStats())
    assert est is not None
    assert any("dequantization not modeled" in note for note in est.assumptions)
