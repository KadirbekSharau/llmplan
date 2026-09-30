"""Does a model fit on a GPU, and how many sequences can one replica hold? (M1 public API)

`fit()` combines exact weight and KV arithmetic with the engine overhead assumption from
`llmplan.memory.engine`. Section references are to docs/milestones/M1_DESIGN.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.errors import ValidationError
from llmplan.memory import engine as engine_model
from llmplan.memory.dtypes import QUANTIZED_INT
from llmplan.memory.engine import EngineProfile
from llmplan.memory.kv_cache import kv_bytes_per_token_per_gpu
from llmplan.memory.weights import per_gpu_weight_bytes
from llmplan.types import DType


class FitRequest(BaseModel):
    """One question: model on GPU with this engine, tensor parallelism, dtype, and context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelSpec
    gpu: GPUSpec
    engine: EngineProfile = EngineProfile()
    tensor_parallel: int = Field(default=1, ge=1)
    dtype: DType = "bf16"
    quantize_embeddings: bool = False
    context_len: int = Field(gt=0)


class FitResult(BaseModel):
    """Per-GPU memory split for one replica and the concurrency it allows. Units: bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fits: bool
    per_gpu_usable_bytes: int
    per_gpu_weight_bytes: int
    per_gpu_overhead_bytes: int
    per_gpu_kv_budget_bytes: int
    kv_bytes_per_token_per_gpu: int
    kv_token_capacity: int
    max_concurrent_seqs_at_context: int
    binding: Literal["weights", "overhead", "ok"]
    notes: tuple[str, ...]
    confidence: Literal["exact", "estimated"]


def _validate(req: FitRequest) -> None:
    spec = req.model
    if spec.num_attention_heads % req.tensor_parallel != 0:
        raise ValidationError(
            f"tensor_parallel {req.tensor_parallel} must divide "
            f"num_attention_heads {spec.num_attention_heads} of {spec.id}"
        )
    if req.context_len > spec.max_position_embeddings:
        raise ValidationError(
            f"context_len {req.context_len} exceeds max_position_embeddings "
            f"{spec.max_position_embeddings} of {spec.id}"
        )


def _notes(req: FitRequest) -> tuple[str, ...]:
    spec = req.model
    notes = [engine_model.overhead_note(req.engine)]
    if req.tensor_parallel > 1:
        notes.append("tensor parallel: even weight split assumed (replicated embeddings ignored)")
    if req.tensor_parallel > spec.num_kv_heads:
        notes.append(
            f"tensor parallel {req.tensor_parallel} > num_kv_heads {spec.num_kv_heads}: "
            "KV heads replicated across GPUs"
        )
    if req.dtype in QUANTIZED_INT:
        notes.append("int4/int8: scales/zeros not counted (typically +2-5%)")
        if not req.quantize_embeddings:
            notes.append("int4/int8: embeddings and lm_head kept in 16 bits")
    if spec.sliding_window is not None:
        notes.append(
            f"sliding window {spec.sliding_window}: KV cache sized for full context (conservative)"
        )
        if req.context_len > spec.sliding_window:
            notes.append(f"context_len {req.context_len} > sliding_window {spec.sliding_window}")
    if spec.head_dim != spec.hidden_size // spec.num_attention_heads:
        notes.append(
            f"explicit head_dim {spec.head_dim} differs from hidden_size // "
            f"num_attention_heads ({spec.hidden_size // spec.num_attention_heads}); trusted"
        )
    if spec.param_count_override is not None:
        notes.append(f"param_count_override {spec.param_count_override} used (not derived)")
    return tuple(notes)


def fit(req: FitRequest) -> FitResult:
    """Compute whether `req.model` fits on `req.gpu` and its KV-cache concurrency.

    Per GPU: usable = floor(vram * gpu_memory_utilization); KV budget = usable - weights -
    overhead. Token capacity = budget // KV bytes per token per GPU; concurrency = capacity
    // context_len; the model fits when at least one full-context sequence fits. Raises
    `ValidationError` when `tensor_parallel` does not divide the attention heads or
    `context_len` exceeds `max_position_embeddings`. A non-fit is a result, not an error.
    """
    _validate(req)
    spec, tp = req.model, req.tensor_parallel
    usable = engine_model.usable_bytes(req.engine, req.gpu)
    weights = per_gpu_weight_bytes(spec, req.dtype, tp, quantize_embeddings=req.quantize_embeddings)
    overhead = engine_model.overhead_bytes(req.engine, spec)
    budget = usable - weights - overhead
    kv_per_token = kv_bytes_per_token_per_gpu(spec, req.engine.kv_dtype, tp)
    capacity = max(0, budget // kv_per_token)
    concurrent = capacity // req.context_len
    fits = concurrent >= 1
    binding: Literal["weights", "overhead", "ok"]
    if weights > usable:
        binding = "weights"
    elif not fits:
        binding = "overhead"
    else:
        binding = "ok"
    return FitResult(
        fits=fits,
        per_gpu_usable_bytes=usable,
        per_gpu_weight_bytes=weights,
        per_gpu_overhead_bytes=overhead,
        per_gpu_kv_budget_bytes=budget,
        kv_bytes_per_token_per_gpu=kv_per_token,
        kv_token_capacity=capacity,
        max_concurrent_seqs_at_context=concurrent,
        binding=binding,
        notes=_notes(req),
        confidence="estimated" if spec.param_count_override is not None else "exact",
    )
