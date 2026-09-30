"""KV-cache bytes per token, GQA and tensor-parallel aware (M1_DESIGN.md section 4.3)."""

from __future__ import annotations

from llmplan.catalog import architectures
from llmplan.catalog.models import ModelSpec
from llmplan.memory.dtypes import bytes_for
from llmplan.types import KVDType


def kv_bytes_per_token_total(spec: ModelSpec, kv_dtype: KVDType) -> int:
    """Key plus value bytes for one token across all layers and all KV heads."""
    return bytes_for(2 * spec.num_layers * spec.num_kv_heads * spec.head_dim, kv_dtype)


def kv_bytes_per_token_per_gpu(spec: ModelSpec, kv_dtype: KVDType, tensor_parallel: int) -> int:
    """Key plus value bytes one GPU stores per token.

    Each GPU holds `ceil(num_kv_heads / tensor_parallel)` heads; when `tensor_parallel`
    exceeds `num_kv_heads`, vLLM replicates heads, so the per-GPU share stops shrinking.
    """
    heads = architectures.get(spec.architecture).kv_heads_per_gpu(spec, tensor_parallel)
    return bytes_for(2 * spec.num_layers * heads * spec.head_dim, kv_dtype)
