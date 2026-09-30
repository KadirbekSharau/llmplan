from __future__ import annotations

import pytest

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.memory.dtypes import bytes_for, bytes_per_element
from llmplan.memory.engine import (
    EngineProfile,
    activation_bytes,
    overhead_bytes,
    overhead_note,
    usable_bytes,
)
from llmplan.memory.fit import FitRequest, fit
from llmplan.memory.kv_cache import kv_bytes_per_token_per_gpu, kv_bytes_per_token_total
from llmplan.memory.weights import model_info, per_gpu_weight_bytes, weight_bytes

H100 = load_gpus()["h100-sxm-80gb"]
DEFAULT_NOTE = (
    "overhead model: 1 GiB fixed + activations(max_num_batched_tokens x hidden x 2 B x 16)"
)


def _spec(name: str, **overrides: object) -> ModelSpec:
    spec = load_model(f"fixture:{name}")
    return spec.model_copy(update=overrides) if overrides else spec


@pytest.mark.parametrize(
    ("dtype", "bpe"),
    [("fp32", 4.0), ("bf16", 2.0), ("fp16", 2.0), ("fp8", 1.0), ("int8", 1.0), ("int4", 0.5)],
)
def test_bytes_per_element(dtype: str, bpe: float) -> None:
    assert bytes_per_element(dtype) == bpe  # type: ignore[arg-type]  # parametrized DType


def test_bytes_for_rounds_up() -> None:
    assert bytes_for(3, "int4") == 2
    assert bytes_for(4, "int4") == 2
    assert bytes_for(3, "bf16") == 6


def test_weight_bytes_variants() -> None:
    spec = _spec("llama3-8b")
    assert weight_bytes(spec, "fp32") == 4 * 8_030_261_248
    assert weight_bytes(spec, "fp8") == 8_030_261_248
    assert weight_bytes(spec, "int4", quantize_embeddings=True) == 8_030_261_248 // 2
    assert weight_bytes(spec, "int8") == 8_030_261_248 + 2 * 525_336_576


def test_weight_bytes_override() -> None:
    spec = _spec("llama3-8b", param_count_override=1_000_000_000)
    assert weight_bytes(spec, "bf16") == 2_000_000_000
    tiny = _spec("llama3-8b", param_count_override=10)  # smaller than the embeddings
    assert weight_bytes(tiny, "int4") == 20


def test_per_gpu_weight_bytes_rounds_up() -> None:
    spec = _spec("llama3-8b", param_count_override=3)
    assert per_gpu_weight_bytes(spec, "bf16", 4) == 2  # ceil(6 / 4)


def test_model_info() -> None:
    info = model_info(_spec("llama3-70b"))
    assert info.param_count == 70_553_706_496
    assert info.attention == "gqa"
    assert info.weight_bytes_by_dtype["bf16"] == 141_107_412_992
    assert set(info.weight_bytes_by_dtype) == {"fp32", "bf16", "fp16", "fp8", "int8", "int4"}


def test_kv_per_gpu() -> None:
    spec = _spec("llama3-70b")
    assert kv_bytes_per_token_per_gpu(spec, "bf16", 1) == kv_bytes_per_token_total(spec, "bf16")
    assert kv_bytes_per_token_per_gpu(spec, "bf16", 8) == 327_680 // 8
    assert kv_bytes_per_token_per_gpu(spec, "bf16", 16) == 327_680 // 8  # replicated


def test_engine_defaults_and_overhead() -> None:
    engine = EngineProfile()
    spec = _spec("llama3-70b")
    assert usable_bytes(engine, H100) == 76_968_728_985
    assert activation_bytes(engine, spec) == 2_147_483_648
    assert overhead_bytes(engine, spec) == 3_221_225_472
    assert overhead_note(engine) == DEFAULT_NOTE


def test_overhead_note_reflects_settings() -> None:
    note = overhead_note(EngineProfile(fixed_overhead_bytes=123, activation_multiplier=4.5))
    assert note == (
        "overhead model: 123 B fixed + activations(max_num_batched_tokens x hidden x 2 B x 4.5)"
    )


@pytest.mark.parametrize("util", [0.0, 1.5])
def test_engine_rejects_bad_utilization(util: float) -> None:
    with pytest.raises(Exception, match="gpu_memory_utilization"):
        EngineProfile(gpu_memory_utilization=util)


def _req(spec: ModelSpec, **kwargs: object) -> FitRequest:
    return FitRequest.model_validate({"model": spec, "gpu": H100, "context_len": 8192, **kwargs})


def test_fit_default_notes_and_confidence() -> None:
    result = fit(_req(_spec("llama3-8b")))
    assert result.notes == (DEFAULT_NOTE,)
    assert result.confidence == "exact"


def test_fit_overhead_binding() -> None:
    result = fit(
        _req(
            _spec("llama3-70b"),
            tensor_parallel=2,
            engine=EngineProfile(gpu_memory_utilization=0.87),
        )
    )
    assert result.fits is False
    assert result.binding == "overhead"
    assert 0 < result.kv_token_capacity < 8192


def test_fit_negative_budget_clamps_capacity() -> None:
    result = fit(_req(_spec("llama3-70b")))
    assert result.per_gpu_kv_budget_bytes < 0
    assert result.kv_token_capacity == 0


def test_fit_notes_for_assumptions() -> None:
    spec = _spec("mistral-7b-v0.1", head_dim=64, param_count_override=7_000_000_000)
    result = fit(_req(spec, tensor_parallel=16, dtype="int4", context_len=8192))
    joined = "\n".join(result.notes)
    assert "tensor parallel: even weight split" in joined
    assert "KV heads replicated" in joined
    assert "int4/int8: scales/zeros not counted (typically +2-5%)" in joined
    assert "embeddings and lm_head kept in 16 bits" in joined
    assert "sliding window 4096" in joined
    assert "context_len 8192 > sliding_window 4096" in joined
    assert "explicit head_dim 64" in joined
    assert "param_count_override" in joined
    assert result.confidence == "estimated"


def test_fit_quantized_embeddings_note_absent() -> None:
    result = fit(_req(_spec("llama3-8b"), dtype="int8", quantize_embeddings=True))
    assert not any("kept in 16 bits" in n for n in result.notes)


def test_fit_kv_dtype_fp8_doubles_capacity() -> None:
    bf16 = fit(_req(_spec("llama3-8b")))
    fp8 = fit(_req(_spec("llama3-8b"), engine=EngineProfile(kv_dtype="fp8")))
    assert fp8.kv_bytes_per_token_per_gpu * 2 == bf16.kv_bytes_per_token_per_gpu
    assert fp8.kv_token_capacity in (2 * bf16.kv_token_capacity, 2 * bf16.kv_token_capacity + 1)


def test_fit_result_is_deterministic_json() -> None:
    req = _req(_spec("qwen2.5-7b"))
    assert fit(req).model_dump_json() == fit(req).model_dump_json()
