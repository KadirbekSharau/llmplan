from __future__ import annotations

from typing import Any

import pytest

from llmplan import render
from llmplan.catalog import architectures
from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import ModelSpec, _spec_from_config, load_model
from llmplan.errors import CatalogError, UnsupportedArchitecture
from llmplan.memory.weights import expert_weight_bytes, model_info, per_gpu_weight_bytes
from llmplan.perf import ReplicaConfig, estimate
from llmplan.perf.benchmarks import BenchmarkRow, physical_floor_s
from llmplan.perf.roofline import decode_weight_bytes
from tests.unit.perf.stats import FakeStats

# A tiny Qwen2-MoE: qkv bias, a shared expert, MoE on every 2nd layer except layer 3.
QWEN2_MOE: dict[str, Any] = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "hidden_size": 16,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 12,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 2,
    "mlp_only_layers": [3],
    "vocab_size": 10,
    "tie_word_embeddings": False,
    "max_position_embeddings": 128,
}


def spec(**changes: Any) -> ModelSpec:
    return _spec_from_config("test/qwen2-moe", {**QWEN2_MOE, **changes}, "huggingface")


def test_qwen2_moe_by_hand() -> None:
    model = spec()
    assert model.architecture == "qwen_moe"
    assert model.attention_bias  # Qwen2-MoE default
    assert model.moe_layer_indices == (1,)  # layers 1 and 3 are on the step; 3 is dense
    # attention 16*16 + 2*16*8 + 16*16 + bias (16 + 8 + 8) = 800; norms 32; dense MLP
    # 3*16*32 = 1536; expert 3*16*8 = 384; router 16*4 = 64; shared 3*16*12 + 16 = 592.
    moe_all = 800 + 32 + 4 * 384 + 64 + 592  # 3024
    moe_active = 800 + 32 + 2 * 384 + 64 + 592  # 2256
    dense = 800 + 32 + 1536  # 2368
    tail = 2 * 10 * 16 + 16  # embed + lm_head + final norm
    info = model_info(model)
    assert info.param_count == moe_all + 3 * dense + tail == 10_464
    assert info.active_param_count == moe_active + 3 * dense + tail == 9_696
    arch = architectures.get("qwen_moe")
    assert arch.expert_params(model) == 4 * 384
    assert architectures.get("llama_like").expert_params(load_model("fixture:llama3-8b")) == 0
    assert expert_weight_bytes(model, "int8") == 4 * 384  # experts are not embeddings
    assert "shared expert 12" in render.get("text").model_info(model)


def test_qwen3_moe_ignores_a_shared_expert_key() -> None:
    model = spec(architectures=["Qwen3MoeForCausalLM"], shared_expert_intermediate_size=99)
    assert model.shared_expert_intermediate_size == 0
    assert model.qk_norm


def test_bad_moe_configs() -> None:
    with pytest.raises(UnsupportedArchitecture) as missing:
        spec(num_experts=None)
    assert missing.value.field == "num_experts"
    with pytest.raises(CatalogError, match="'decoder_sparse_step': expected an integer"):
        spec(decoder_sparse_step="2")
    with pytest.raises(CatalogError, match="'decoder_sparse_step': must be >= 1"):
        spec(decoder_sparse_step=0)
    with pytest.raises(CatalogError, match="'mlp_only_layers': expected a list"):
        spec(mlp_only_layers=3)
    with pytest.raises(CatalogError, match="experts_per_token"):
        spec(num_experts_per_tok=5)
    with pytest.raises(CatalogError, match="moe_layer_indices"):
        spec(mlp_only_layers=[1, 3])  # no MoE layer left
    dense = load_model("fixture:llama3-8b").model_dump()
    with pytest.raises(ValueError, match="need num_experts > 0"):
        ModelSpec.model_validate({**dense, "experts_per_token": 2})
    with pytest.raises(ValueError, match="moe_intermediate_size is required"):
        ModelSpec.model_validate(
            {**dense, "num_experts": 2, "experts_per_token": 1, "moe_layer_indices": (0,)}
        )


def test_param_count_override_treats_a_moe_model_as_dense() -> None:
    model = spec(architectures=["MixtralForCausalLM"], num_local_experts=4)
    overridden = model.model_copy(update={"param_count_override": 1_000})
    info = model_info(overridden)
    assert info.param_count == info.active_param_count == 1_000
    per_gpu = per_gpu_weight_bytes(overridden, "bf16", 1)
    assert decode_weight_bytes(overridden, "bf16", 1, 1, per_gpu) == per_gpu


def test_moe_roofline_note_and_physical_floor() -> None:
    mixtral = load_model("fixture:mixtral-8x7b")
    h100 = load_gpus()["h100-sxm-80gb"]
    stats = FakeStats()
    config = ReplicaConfig(dtype="fp8", tensor_parallel=1, max_num_seqs=8, max_model_len=4096)
    result = estimate(mixtral, h100, config, stats, backend="roofline")
    assert any(
        note.startswith("mixture of experts: FLOPs use the 12,879,925,248 active")
        for note in result.assumptions
    )
    row = BenchmarkRow(
        model_id="fixture:mixtral-8x7b",
        gpu_id="h100-sxm-80gb",
        engine="vllm",
        engine_version="0.30.0",
        tensor_parallel=1,
        dtype="fp8",
        concurrency=1,
        input_len=1000,
        output_len=200,
        output_tokens_per_s=100.0,
        ttft_ms_p50=None,
        ttft_ms_p95=None,
        tpot_ms_p50=None,
        tpot_ms_p95=None,
        source_url="user-upload",
        as_of="2026-10-01",
    )
    floor = physical_floor_s(row, mixtral, h100)
    assert floor is not None
    assert h100.memory_bandwidth_gbps is not None
    # batch 1 touches 2 of 8 experts: well under reading all 46.7 GB of fp8 weights.
    assert floor < 46_702_792_704 / (h100.memory_bandwidth_gbps * 1e9)
