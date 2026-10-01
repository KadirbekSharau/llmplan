from __future__ import annotations

import pytest

from llmplan.catalog import architectures
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.errors import UnknownRegistryKey, UnsupportedArchitecture


def _tiny(**overrides: object) -> ModelSpec:
    base: dict[str, object] = {
        "id": "manual:tiny",
        "architecture": "llama_like",
        "hidden_size": 8,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 2,
        "intermediate_size": 16,
        "vocab_size": 10,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "max_position_embeddings": 64,
        "source": "manual",
    }
    return ModelSpec.model_validate({**base, **overrides})


def test_get_unknown_key() -> None:
    with pytest.raises(UnknownRegistryKey, match="nope"):
        architectures.get("nope")


def test_resolve_hf_class() -> None:
    key, defaults = architectures.resolve_hf_class("Qwen3ForCausalLM")
    assert key == "llama_like"
    assert defaults.qk_norm is True
    assert architectures.resolve_hf_class("MixtralForCausalLM")[0] == "mixtral"  # M8
    with pytest.raises(UnsupportedArchitecture) as info:
        architectures.resolve_hf_class("GPT2LMHeadModel")
    assert info.value.field == "architectures"
    assert "GPT2LMHeadModel" in str(info.value)


def test_tiny_param_count_by_hand() -> None:
    # H=8, A=4, K=2, d=2, I=16, V=10, L=2
    # attn = 8*8 + 2*8*4 + 8*8 = 192; mlp = 3*8*16 = 384; norms = 16 -> per_layer 592
    # embed 80 + lm_head 80 + final_norm 8
    arch = architectures.get("llama_like")
    assert arch.count_params(_tiny()) == 2 * 592 + 80 + 80 + 8


def test_bias_and_qk_norm_terms() -> None:
    arch = architectures.get("llama_like")
    base = arch.count_params(_tiny())
    # qkv bias: A*d + 2*K*d = 8 + 8 = 16 per layer
    assert arch.count_params(_tiny(attention_bias=True)) == base + 2 * 16
    # mlp bias: 2*I + H = 40 per layer
    assert arch.count_params(_tiny(mlp_bias=True)) == base + 2 * 40
    # qk_norm: 2*d = 4 per layer
    assert arch.count_params(_tiny(qk_norm=True)) == base + 2 * 4
    # tied: lm_head removed
    assert arch.count_params(_tiny(tie_word_embeddings=True)) == base - 80


def test_override_and_embeddings() -> None:
    arch = architectures.get("llama_like")
    assert arch.count_params(_tiny(param_count_override=123)) == 123
    assert arch.embedding_params(_tiny()) == 160
    assert arch.embedding_params(_tiny(tie_word_embeddings=True)) == 80


@pytest.mark.parametrize(("tp", "expected"), [(1, 8), (2, 4), (4, 2), (8, 1), (16, 1), (64, 1)])
def test_kv_heads_per_gpu_replicates(tp: int, expected: int) -> None:
    spec = load_model("fixture:llama3-70b")
    assert architectures.get("llama_like").kv_heads_per_gpu(spec, tp) == expected
