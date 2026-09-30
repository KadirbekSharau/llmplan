"""Roofline backend: first-principles throughput and latency bounds (M3_DESIGN.md section 4).

Decode is bounded by streaming the weights plus the batch's KV cache from HBM once per step,
or by the step's matmul FLOPs, whichever is slower; prefill by FLOPs alone. The efficiency
constants below are assumptions, restated in every estimate's `assumptions`.
"""

from __future__ import annotations

from llmplan.catalog import architectures
from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.memory.dtypes import QUANTIZED_INT
from llmplan.perf.config import ReplicaConfig, fit_for
from llmplan.perf.estimate import PerfEstimate, StatsLike, register
from llmplan.types import DType

BANDWIDTH_EFFICIENCY = 0.70  # fraction of datasheet HBM bandwidth a decode step achieves
PREFILL_MFU = 0.50  # model FLOPs utilization during prefill
DECODE_MFU = 0.30  # model FLOPs utilization during decode
P95_FACTOR = 1.5  # p95 TPOT = p50 x factor (the table backend uses measured p95 instead)

CONSTANTS_NOTE = (
    f"roofline constants: bandwidth efficiency {BANDWIDTH_EFFICIENCY:g}, decode MFU "
    f"{DECODE_MFU:g}, prefill MFU {PREFILL_MFU:g}, p95 TPOT = p50 x {P95_FACTOR:g}"
)
SERVICE_NOTE = "latencies are service times; queueing is not modeled"


def param_count(model: ModelSpec) -> int:
    """Exact parameter count (or the model's `param_count_override`)."""
    return architectures.get(model.architecture).count_params(model)


def tflops_field(dtype: DType) -> str:
    """Name of the `GPUSpec` dense-TFLOPS field used for weights stored in `dtype`."""
    return "fp8_dense_tflops" if dtype == "fp8" else "fp16_dense_tflops"


def dense_tflops(gpu: GPUSpec, dtype: DType) -> float | None:
    """Dense Tensor Core TFLOPS for `dtype`: fp8 uses the fp8 figure, all others fp16."""
    return gpu.fp8_dense_tflops if dtype == "fp8" else gpu.fp16_dense_tflops


def avg_ctx_tokens(stats: StatsLike) -> float:
    """Mean tokens in a sequence's KV cache over its life: input plus half the output."""
    return stats.input_tokens_mean + stats.output_tokens_mean / 2


def effective_batch(kv_token_capacity: int, max_num_seqs: int, ctx_tokens: float) -> int:
    """Concurrent sequences a replica runs: `max_num_seqs` capped by KV capacity, at least 1."""
    kv_seq_capacity = int(kv_token_capacity // max(ctx_tokens, 1))
    return max(1, min(max_num_seqs, kv_seq_capacity))


def decode_memory_s(
    weight_bytes_per_gpu: int,
    kv_bytes_per_token_per_gpu: int,
    batch: int,
    ctx_tokens: float,
    bandwidth_gbps: float,
    efficiency: float = BANDWIDTH_EFFICIENCY,
) -> float:
    """Seconds per decode step to read the weights and `batch * ctx_tokens` KV tokens.

    Per GPU of the replica, at `bandwidth_gbps * 1e9 * efficiency` bytes/s.
    """
    step_bytes = weight_bytes_per_gpu + batch * ctx_tokens * kv_bytes_per_token_per_gpu
    return step_bytes / (bandwidth_gbps * 1e9 * efficiency)


def decode_compute_s(
    params: int, batch: int, tensor_parallel: int, tflops: float, mfu: float = DECODE_MFU
) -> float:
    """Seconds per decode step for `2 * params * batch / tensor_parallel` FLOPs per GPU."""
    return 2 * params * batch / tensor_parallel / (tflops * 1e12 * mfu)


def prefill_tokens_per_s(tflops: float, tensor_parallel: int, params: int) -> float:
    """Prefill rate of one replica: `tflops * 1e12 * PREFILL_MFU * tp / (2 * params)`."""
    return tflops * 1e12 * PREFILL_MFU * tensor_parallel / (2 * params)


def _notes(config: ReplicaConfig, batch: int, kv_seqs: int, ctx: float) -> list[str]:
    notes = [
        CONSTANTS_NOTE,
        f"effective batch {batch}: min(max_num_seqs {config.max_num_seqs}, KV capacity "
        f"{kv_seqs} sequences at {ctx:g} mean context tokens)",
    ]
    if config.tensor_parallel > 1:
        notes.append("tensor parallel: ideal compute scaling assumed (communication not modeled)")
    if config.dtype in QUANTIZED_INT:
        notes.append("int8/int4: fp16 Tensor Core TFLOPS used; dequantization not modeled")
    return notes


def _evaluate(
    model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
) -> PerfEstimate | str:
    fit = fit_for(model, gpu, config)
    if not fit.fits:
        return (
            f"{model.id} does not fit on {gpu.id} at tensor_parallel {config.tensor_parallel} "
            f"with max_model_len {config.max_model_len} (binding: {fit.binding})"
        )
    bandwidth = gpu.memory_bandwidth_gbps
    tflops = dense_tflops(gpu, config.dtype)
    if bandwidth is None:
        return f"gpu {gpu.id} has memory_bandwidth_gbps null"
    if tflops is None:
        return f"gpu {gpu.id} has {tflops_field(config.dtype)} null"
    params = param_count(model)
    ctx = avg_ctx_tokens(stats)
    batch = effective_batch(fit.kv_token_capacity, config.max_num_seqs, ctx)
    t_mem = decode_memory_s(
        fit.per_gpu_weight_bytes, fit.kv_bytes_per_token_per_gpu, batch, ctx, bandwidth
    )
    t_compute = decode_compute_s(params, batch, config.tensor_parallel, tflops)
    tpot_s = max(t_mem, t_compute)
    prefill = prefill_tokens_per_s(tflops, config.tensor_parallel, params)
    service_s = stats.input_tokens_mean / prefill + stats.output_tokens_mean * tpot_s
    notes = _notes(config, batch, int(fit.kv_token_capacity // max(ctx, 1)), ctx)
    bound = "compute" if t_compute >= t_mem else "memory"
    notes.append(
        f"decode is {bound}-bound: memory {t_mem * 1e3:.3f} ms, compute "
        f"{t_compute * 1e3:.3f} ms per step"
    )
    notes.append(SERVICE_NOTE)
    return PerfEstimate(
        backend="roofline",
        confidence="roofline",
        effective_batch=batch,
        decode_tokens_per_s=batch / tpot_s,
        prefill_tokens_per_s=prefill,
        requests_per_s_capacity=batch / service_s,
        ttft_ms_p50=stats.input_tokens_p50 / prefill * 1e3,
        ttft_ms_p95=stats.input_tokens_p95 / prefill * 1e3,
        tpot_ms_p50=tpot_s * 1e3,
        tpot_ms_p95=tpot_s * 1e3 * P95_FACTOR,
        assumptions=tuple(notes),
        source_urls=(),
    )


@register("roofline")
class RooflineBackend:
    """Always-available bounds from M1 memory arithmetic and the GPU catalog.

    Returns `None` when the model does not fit the replica or the GPU lacks
    `memory_bandwidth_gbps` or the dense TFLOPS figure for the weight dtype.
    """

    name = "roofline"

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        result = _evaluate(model, gpu, config, stats)
        return result if isinstance(result, PerfEstimate) else None

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        result = _evaluate(model, gpu, config, stats)
        return result if isinstance(result, str) else "an estimate is available"
