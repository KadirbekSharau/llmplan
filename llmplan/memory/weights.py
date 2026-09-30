"""Weight bytes from the exact parameter count (M1_DESIGN.md sections 4.2 and 4.4)."""

from __future__ import annotations

from typing import get_args

from llmplan.catalog import architectures
from llmplan.catalog.models import DerivedModelInfo, ModelSpec
from llmplan.memory.dtypes import QUANTIZED_INT, bytes_for
from llmplan.types import DType

EMBEDDING_DTYPE: DType = "bf16"  # int8/int4 checkpoints keep embeddings in 16 bits


def weight_bytes(spec: ModelSpec, dtype: DType, *, quantize_embeddings: bool = False) -> int:
    """Total weight bytes of one model copy stored in `dtype`.

    For int8/int4, embeddings and lm_head stay 16-bit unless `quantize_embeddings` is true.
    Quantization scales and zero-points are not counted (callers note this). Uses
    `param_count_override` when set; the embedding share is then capped at the override.
    """
    arch = architectures.get(spec.architecture)
    params = arch.count_params(spec)
    if dtype in QUANTIZED_INT and not quantize_embeddings:
        embedding = min(arch.embedding_params(spec), params)
        return bytes_for(params - embedding, dtype) + bytes_for(embedding, EMBEDDING_DTYPE)
    return bytes_for(params, dtype)


def per_gpu_weight_bytes(
    spec: ModelSpec, dtype: DType, tensor_parallel: int, *, quantize_embeddings: bool = False
) -> int:
    """Weight bytes on each GPU of a replica, assuming an even split across `tensor_parallel`.

    Uneven shards and replicated embeddings are ignored in M1.
    """
    total = weight_bytes(spec, dtype, quantize_embeddings=quantize_embeddings)
    return -(-total // tensor_parallel)


def model_info(spec: ModelSpec) -> DerivedModelInfo:
    """Parameter count, attention kind, and weight bytes for every `DType` of `spec`."""
    return DerivedModelInfo(
        param_count=architectures.get(spec.architecture).count_params(spec),
        attention=spec.attention,
        weight_bytes_by_dtype={d: weight_bytes(spec, d) for d in get_args(DType)},
    )
