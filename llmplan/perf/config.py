"""Replica configuration for the performance model (M3_DESIGN.md section 3).

A `ReplicaConfig` is the per-replica knob set the planner chooses (tensor parallelism,
dtypes, vLLM limits). `fit_for` maps it onto the M1 memory model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, FitResult, fit
from llmplan.types import DType, KVDType


class ReplicaConfig(BaseModel):
    """One replica's engine settings: vLLM flags plus tensor parallelism and dtypes.

    `max_model_len` must not exceed the model's `max_position_embeddings`; that check needs
    the model, so `llmplan.perf.estimate` enforces it (raising `ValidationError`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tensor_parallel: int = Field(default=1, ge=1)
    dtype: DType = "bf16"
    kv_dtype: KVDType = "bf16"
    max_num_seqs: int = Field(default=256, gt=0)
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    max_num_batched_tokens: int = Field(default=8192, gt=0)


def fit_for(model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig) -> FitResult:
    """Run M1 `fit()` for one replica, with `max_model_len` as the context length.

    Engine overhead constants other than those in `ReplicaConfig` keep their vLLM defaults.
    Raises `ValidationError` as `fit()` does (tensor parallel not dividing the heads).
    """
    return fit(
        FitRequest(
            model=model,
            gpu=gpu,
            engine=EngineProfile(
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_num_batched_tokens=config.max_num_batched_tokens,
                kv_dtype=config.kv_dtype,
            ),
            tensor_parallel=config.tensor_parallel,
            dtype=config.dtype,
            context_len=config.max_model_len,
        )
    )
