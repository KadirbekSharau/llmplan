"""Dense llama-like decoders: Llama, Mistral, Qwen2, Qwen3 (M1_DESIGN.md section 4.1)."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING

from llmplan.catalog.architectures import HFClassDefaults, register

if TYPE_CHECKING:
    from llmplan.catalog.models import ModelSpec


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

    def count_params(self, spec: ModelSpec) -> int:
        if spec.param_count_override is not None:
            return spec.param_count_override
        h, a, k, d, i = (
            spec.hidden_size,
            spec.num_attention_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.intermediate_size,
        )
        qkv_bias = (a * d + 2 * k * d) if spec.attention_bias else 0
        attention = h * (a * d) + 2 * h * (k * d) + qkv_bias + (a * d) * h  # q, k, v, o
        mlp_bias = (2 * i + h) if spec.mlp_bias else 0
        mlp = 3 * h * i + mlp_bias  # gate, up, down
        norms = 2 * h + (2 * d if spec.qk_norm else 0)
        per_layer = attention + mlp + norms
        final_norm = h
        return spec.num_layers * per_layer + self.embedding_params(spec) + final_norm

    def embedding_params(self, spec: ModelSpec) -> int:
        embed = spec.vocab_size * spec.hidden_size
        lm_head = 0 if spec.tie_word_embeddings else embed
        return embed + lm_head

    def kv_heads_per_gpu(self, spec: ModelSpec, tensor_parallel: int) -> int:
        # vLLM replicates KV heads when tensor_parallel > num_kv_heads.
        return ceil(spec.num_kv_heads / tensor_parallel)
