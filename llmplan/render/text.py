"""Plain-text renderer. GB means 1e9 bytes (M1_DESIGN.md section 12)."""

from __future__ import annotations

from collections.abc import Mapping

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.memory.engine import GIB, activation_bytes
from llmplan.memory.fit import FitRequest, FitResult
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import model_info, weight_bytes
from llmplan.render import register

LABEL_WIDTH = 10


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def _fixed(n_bytes: int) -> str:
    return f"{n_bytes // GIB} GiB" if n_bytes % GIB == 0 else _gb(n_bytes)


def _heads(spec: ModelSpec) -> str:
    return f"{spec.attention.upper()} {spec.num_attention_heads}/{spec.num_kv_heads} heads"


def _opt(value: float | None, unit: str) -> str:
    return "-" if value is None else f"{value:g} {unit}"


@register("text")
class TextRenderer:
    """Aligned, human-readable lines; content per M1_DESIGN.md section 8."""

    def fit(self, request: FitRequest, result: FitResult) -> str:
        spec, gpu, engine = request.model, request.gpu, request.engine
        info = model_info(spec)
        total_weights = weight_bytes(
            spec, request.dtype, quantize_embeddings=request.quantize_embeddings
        )
        util = f"{engine.gpu_memory_utilization * 100:g}%"
        verdict = "FITS" if result.fits else "DOES NOT FIT"
        lines = [
            _line(
                "Model",
                f"{spec.id}  ({spec.architecture}, {info.param_count / 1e9:.2f}B params, "
                f"{_heads(spec)})",
            ),
            _line(
                "GPU",
                f"{gpu.name} x TP={request.tensor_parallel}   usable "
                f"{_gb(result.per_gpu_usable_bytes)}/GPU ({util} of {_gb(gpu.vram_bytes)})",
            ),
            _line(
                "Weights",
                f"{request.dtype}   {_gb(total_weights)} total   "
                f"{_gb(result.per_gpu_weight_bytes)}/GPU",
            ),
            _line(
                "Overhead",
                f"{_gb(result.per_gpu_overhead_bytes)}/GPU  ({_fixed(engine.fixed_overhead_bytes)} "
                f"fixed + {_gb(activation_bytes(engine, spec))} activations)",
            ),
            _line(
                "KV cache",
                f"{result.kv_bytes_per_token_per_gpu:,} B/token/GPU ({engine.kv_dtype})   budget "
                f"{_gb(result.per_gpu_kv_budget_bytes)}/GPU   capacity "
                f"{result.kv_token_capacity:,} tokens",
            ),
            _line(
                "Context",
                f"{request.context_len:,} tokens  ->  max "
                f"{result.max_concurrent_seqs_at_context:,} concurrent sequences",
            ),
            _line("Result", f"{verdict}  (binding: {result.binding}, {result.confidence})"),
        ]
        for i, note in enumerate(result.notes):
            lines.append(_line("Notes" if i == 0 else "", f"- {note}"))
        return "\n".join(lines) + "\n"

    def model_info(self, spec: ModelSpec) -> str:
        info = model_info(spec)
        window = "none" if spec.sliding_window is None else f"{spec.sliding_window:,}"
        weights = " | ".join(f"{d} {_gb(b)}" for d, b in info.weight_bytes_by_dtype.items())
        width = LABEL_WIDTH + 3
        lines = [
            f"{'Model':<{width}}{spec.id}  (source: {spec.source})",
            f"{'Architecture':<{width}}{spec.architecture}",
            f"{'Params':<{width}}{info.param_count:,} ({info.param_count / 1e9:.2f}B)"
            + ("  [override]" if spec.param_count_override is not None else ""),
            f"{'Attention':<{width}}{_heads(spec)}, head_dim {spec.head_dim}",
            f"{'Layers':<{width}}{spec.num_layers}   hidden {spec.hidden_size:,}   "
            f"intermediate {spec.intermediate_size:,}   vocab {spec.vocab_size:,}",
            f"{'Context':<{width}}max_position_embeddings {spec.max_position_embeddings:,}   "
            f"sliding_window {window}",
            f"{'KV cache':<{width}}{kv_bytes_per_token_total(spec, 'bf16'):,} B/token (bf16)",
            f"{'Weights':<{width}}{weights}",
        ]
        return "\n".join(lines) + "\n"

    def gpus(self, gpus: Mapping[str, GPUSpec]) -> str:
        header = (
            f"{'id':<16}{'name':<24}{'vram':>10}{'bandwidth':>13}{'fp16 dense':>14}"
            f"{'fp8 dense':>14}{'nvlink':>8}  as_of"
        )
        rows = [header]
        for gpu in gpus.values():
            rows.append(
                f"{gpu.id:<16}{gpu.name:<24}{_gb(gpu.vram_bytes):>10}"
                f"{_opt(gpu.memory_bandwidth_gbps, 'GB/s'):>13}"
                f"{_opt(gpu.fp16_dense_tflops, 'TFLOPS'):>14}"
                f"{_opt(gpu.fp8_dense_tflops, 'TFLOPS'):>14}"
                f"{'yes' if gpu.nvlink else 'no':>8}  {gpu.as_of.isoformat()}"
            )
        return "\n".join(rows) + "\n"
