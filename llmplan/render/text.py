"""Plain-text renderer. GB means 1e9 bytes (M1_DESIGN.md section 12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.memory.engine import GIB, activation_bytes
from llmplan.memory.fit import FitRequest, FitResult
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import model_info, weight_bytes
from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import PerfEstimate, StatsLike
from llmplan.planner.request import PlanRequest
from llmplan.planner.result import PlanResult
from llmplan.render import register
from llmplan.render.plan_text import plan_text

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

    def perf_estimate(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        config: ReplicaConfig,
        stats: StatsLike,
        result: PerfEstimate,
    ) -> str:
        r = result
        lines = [
            _line(
                "Replica",
                f"{model.id} on {gpu.name} x TP={config.tensor_parallel}  ({config.dtype}, "
                f"KV {config.kv_dtype}, max_num_seqs {config.max_num_seqs}, max_model_len "
                f"{config.max_model_len:,})",
            ),
            _line(
                "Workload",
                f"input mean {stats.input_tokens_mean:g} / p50 {stats.input_tokens_p50:g} / p95 "
                f"{stats.input_tokens_p95:g}   output mean {stats.output_tokens_mean:g} / p50 "
                f"{stats.output_tokens_p50:g} / p95 {stats.output_tokens_p95:g} tokens",
            ),
            _line("Backend", f"{r.backend}  (confidence: {r.confidence})"),
            _line("Batch", f"{r.effective_batch:,} concurrent sequences"),
            _line(
                "Decode",
                f"{r.decode_tokens_per_s:,.0f} tokens/s   TPOT p50 {r.tpot_ms_p50:.2f} ms, "
                f"p95 {r.tpot_ms_p95:.2f} ms",
            ),
            _line(
                "Prefill",
                f"{r.prefill_tokens_per_s:,.0f} tokens/s   TTFT p50 {r.ttft_ms_p50:.2f} ms, "
                f"p95 {r.ttft_ms_p95:.2f} ms",
            ),
            _line("Capacity", f"{r.requests_per_s_capacity:,.2f} requests/s (service time only)"),
        ]
        for i, note in enumerate(r.assumptions):
            lines.append(_line("Notes" if i == 0 else "", f"- {note}"))
        for i, url in enumerate(r.source_urls):
            lines.append(_line("Sources" if i == 0 else "", url))
        return "\n".join(lines) + "\n"

    def benchmarks(self, rows: Sequence[BenchmarkRow]) -> str:
        header = (
            f"{'gpu':<15}{'model':<36}{'engine':<13}{'tp':>3}{'dtype':>6}{'in/out':>11}"
            f"{'conc':>6}{'out tok/s':>11}"
        )
        lines = [header]
        for row in rows:
            engine = f"{row.engine} {row.engine_version}"
            shape = f"{row.input_len}/{row.output_len}"
            lines.append(
                f"{row.gpu_id:<15}{row.model_id:<36}{engine:<13}{row.tensor_parallel:>3}"
                f"{row.dtype:>6}{shape:>11}{row.concurrency:>6}{row.output_tokens_per_s:>11,.2f}"
            )
        sources = sorted({(row.source_url, row.as_of.isoformat()) for row in rows})
        lines.append(f"{len(rows)} rows")
        lines.extend(f"source: {url} (as of {as_of})" for url, as_of in sources)
        return "\n".join(lines) + "\n"

    def plan(self, request: PlanRequest, result: PlanResult) -> str:
        return plan_text(request, result)
