"""vLLM memory model: usable VRAM and per-GPU overhead (M1_DESIGN.md section 5).

Assumptions (the only non-exact inputs in M1, to be calibrated in M3):
- `fixed_overhead_bytes` (default 1 GiB) covers the CUDA context, NCCL buffers, and CUDA
  graphs on each GPU.
- Peak activation memory during a forward pass is
  `max_num_batched_tokens * hidden_size * 2 bytes * activation_multiplier` (default 16).
Every `FitResult` states these in its notes; they do not lower `confidence`.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.types import KVDType

GIB = 2**30
ACTIVATION_BYTES_PER_ELEMENT = 2


class EngineProfile(BaseModel):
    """Inference-engine settings that shape per-GPU memory. Defaults are vLLM's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: Literal["vllm"] = "vllm"
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    max_num_batched_tokens: int = Field(default=8192, gt=0)
    fixed_overhead_bytes: int = Field(default=1 * GIB, ge=0)
    activation_multiplier: float = Field(default=16.0, ge=0)
    kv_dtype: KVDType = "bf16"


def usable_bytes(engine: EngineProfile, gpu: GPUSpec) -> int:
    """Bytes of one GPU the engine may use: `floor(vram_bytes * gpu_memory_utilization)`."""
    return floor(gpu.vram_bytes * engine.gpu_memory_utilization)


def activation_bytes(engine: EngineProfile, spec: ModelSpec) -> int:
    """Peak activation bytes per GPU under the documented multiplier assumption."""
    elements = engine.max_num_batched_tokens * spec.hidden_size * ACTIVATION_BYTES_PER_ELEMENT
    return ceil(elements * engine.activation_multiplier)


def overhead_bytes(engine: EngineProfile, spec: ModelSpec) -> int:
    """Non-weight, non-KV bytes per GPU: fixed overhead plus peak activations."""
    return engine.fixed_overhead_bytes + activation_bytes(engine, spec)


def overhead_note(engine: EngineProfile) -> str:
    """Human-readable statement of the overhead assumption, for `FitResult.notes`."""
    fixed = engine.fixed_overhead_bytes
    fixed_text = f"{fixed // GIB} GiB" if fixed % GIB == 0 else f"{fixed} B"
    return (
        f"overhead model: {fixed_text} fixed + activations(max_num_batched_tokens x hidden x "
        f"{ACTIVATION_BYTES_PER_ELEMENT} B x {engine.activation_multiplier:g})"
    )
