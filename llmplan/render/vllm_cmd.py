"""`vllm serve` command lines for planned replicas (M4_DESIGN.md section 8).

Pure functions, not a registry renderer: they format a replica configuration, which only
plans produce. The text renderer embeds them and `llmplan plan --format vllm` prints them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmplan.catalog.models import ModelSpec
from llmplan.memory.dtypes import QUANTIZED_INT
from llmplan.perf.config import ReplicaConfig

if TYPE_CHECKING:
    from llmplan.planner.request import PlanRequest
    from llmplan.planner.result import PlanResult

MODEL_PLACEHOLDER = "<MODEL_ID>"
VLLM_DTYPE = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16", "fp8": "bfloat16"}


def serve_command(model: ModelSpec, config: ReplicaConfig) -> str:
    """One `vllm serve` line for `config`, preceded by a comment line for int8/int4.

    `fp8` weights render `--dtype bfloat16 --quantization fp8`; an fp8 KV cache adds
    `--kv-cache-dtype fp8`. int8/int4 need a pre-quantized checkpoint, so the dtype is left
    to vLLM (`--dtype auto`) and a comment says so. Fixture model ids render as
    `<MODEL_ID>`, a placeholder for the real Hugging Face id.
    """
    model_id = MODEL_PLACEHOLDER if model.id.startswith("fixture:") else model.id
    quantized = config.dtype in QUANTIZED_INT
    parts = [
        "vllm serve",
        model_id,
        f"--tensor-parallel-size {config.tensor_parallel}",
        f"--max-num-seqs {config.max_num_seqs}",
        f"--max-model-len {config.max_model_len}",
        f"--gpu-memory-utilization {config.gpu_memory_utilization:g}",
        f"--dtype {'auto' if quantized else VLLM_DTYPE[config.dtype]}",
    ]
    if config.dtype == "fp8":
        parts.append("--quantization fp8")
    if config.kv_dtype == "fp8":
        parts.append("--kv-cache-dtype fp8")
    line = " ".join(parts)
    if quantized:
        return (
            f"# {config.dtype} weights need a pre-quantized {config.dtype} checkpoint "
            f"(e.g. AWQ or GPTQ); vLLM reads the quantization method from it\n{line}"
        )
    return line


def plan_commands(request: PlanRequest, result: PlanResult) -> str:
    """Every replica plan of `result` as a comment line and its `vllm serve` command."""
    lines = []
    for replica in result.replicas:
        row = replica.candidate.price_row
        lines.append(
            f"# {replica.count} replica(s) on {replica.instances} x {row.provider} "
            f"{row.instance} ({row.gpu_count} x {row.gpu_id} per instance)"
        )
        lines.append(serve_command(request.model, replica.candidate.config))
    return "\n".join(lines) + "\n"
