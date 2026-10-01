"""Dense llama-like decoders: Llama, Mistral, Qwen2, Qwen3 (M1_DESIGN.md section 4.1).

`attention_params`, `norm_params`, `dense_mlp_params` and `embedding_params` are the
per-part formulas, shared with the mixture-of-experts families (M8, `moe.py`), whose
attention is the same dense attention.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING, Any

from llmplan.catalog.architectures import HFClassDefaults, register

if TYPE_CHECKING:
    from llmplan.catalog.models import ModelSpec


def attention_params(spec: ModelSpec) -> int:
    """q, k, v, o projections of one layer, plus the q/k/v bias when the model has one."""
    h, a, k, d = spec.hidden_size, spec.num_attention_heads, spec.num_kv_heads, spec.head_dim
    qkv_bias = (a * d + 2 * k * d) if spec.attention_bias else 0
    return h * (a * d) + 2 * h * (k * d) + qkv_bias + (a * d) * h


def norm_params(spec: ModelSpec) -> int:
    """The two RMSNorms of one layer, plus Qwen3's q_norm/k_norm (`qk_norm`)."""
    return 2 * spec.hidden_size + (2 * spec.head_dim if spec.qk_norm else 0)


def dense_mlp_params(spec: ModelSpec) -> int:
    """SwiGLU MLP of one dense layer: gate, up, down (and their bias when `mlp_bias`)."""
    h, i = spec.hidden_size, spec.intermediate_size
    return 3 * h * i + ((2 * i + h) if spec.mlp_bias else 0)


def embedding_params(spec: ModelSpec) -> int:
    """Input embedding plus lm_head (none when the embeddings are tied)."""
    embed = spec.vocab_size * spec.hidden_size
    return embed + (0 if spec.tie_word_embeddings else embed)


@register("llama_like")
class LlamaLike:
    """Exact parameter count for RMSNorm + GQA attention + SwiGLU MLP decoders."""

    @property
    def hf_classes(self) -> Mapping[str, HFClassDefaults]:
        return {
            "LlamaForCausalLM": HFClassDefaults(),
            "MistralForCausalLM": HFClassDefaults(),
            "Qwen2ForCausalLM": HFClassDefaults(attention_bias=True),
            "Qwen3ForCausalLM": HFClassDefaults(qk_norm=True),
        }

    def config_fields(self, model_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        return {}  # dense: no fields beyond the common ones

    def count_params(self, spec: ModelSpec) -> int:
        if spec.param_count_override is not None:
            return spec.param_count_override
        per_layer = attention_params(spec) + dense_mlp_params(spec) + norm_params(spec)
        final_norm = spec.hidden_size
        return spec.num_layers * per_layer + self.embedding_params(spec) + final_norm

    def active_params(self, spec: ModelSpec) -> int:
        return self.count_params(spec)  # dense: every parameter works on every token

    def expert_params(self, spec: ModelSpec) -> int:
        return 0

    def embedding_params(self, spec: ModelSpec) -> int:
        return embedding_params(spec)

    def kv_heads_per_gpu(self, spec: ModelSpec, tensor_parallel: int) -> int:
        # vLLM replicates KV heads when tensor_parallel > num_kv_heads.
        return ceil(spec.num_kv_heads / tensor_parallel)
