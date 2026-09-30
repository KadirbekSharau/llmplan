"""Table backend: interpolation over published benchmark rows (M3_DESIGN.md section 5.3).

Rows are matched on (model, GPU, tensor parallel, dtype), narrowed to one engine and to the
request shape nearest the workload's mean, then interpolated linearly in log(concurrency) at
the replica's effective batch. The backend never extrapolates request shape.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.perf.benchmarks import BenchmarkRow, BenchmarkTable, default_table
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import PerfEstimate, StatsLike, register
from llmplan.perf.roofline import (
    P95_FACTOR,
    SERVICE_NOTE,
    avg_ctx_tokens,
    dense_tflops,
    effective_batch,
    fit_or_reason,
    param_count,
    prefill_tokens_per_s,
    tflops_field,
)

MAX_SHAPE_RATIO = 2.0  # nearest benchmark shape may differ from the workload mean by this
PREFERRED_ENGINE = "vllm"  # the engine EngineProfile models; other engines used otherwise
LATENCY_FIELDS = ("ttft_ms_p50", "ttft_ms_p95", "tpot_ms_p50", "tpot_ms_p95")

Confidence = Literal["interpolated", "measured"]


def _matching_rows(
    table: BenchmarkTable, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig
) -> list[BenchmarkRow]:
    canonical = table.canonical(model.id)
    return [
        row
        for row in table.rows
        if table.canonical(row.model_id) == canonical
        and row.gpu_id == gpu.id
        and row.tensor_parallel == config.tensor_parallel
        and row.dtype == config.dtype
    ]


def _nearest_shape(
    rows: Sequence[BenchmarkRow], want_in: float, want_out: float
) -> tuple[int, int]:
    def distance(shape: tuple[int, int]) -> tuple[float, int, int]:
        d = math.hypot(math.log(shape[0] / want_in), math.log(shape[1] / want_out))
        return d, shape[0], shape[1]

    return min({(row.input_len, row.output_len) for row in rows}, key=distance)


def _by_concurrency(rows: Sequence[BenchmarkRow]) -> dict[int, BenchmarkRow]:
    """One row per concurrency: the lowest throughput when several agree on it."""
    chosen: dict[int, BenchmarkRow] = {}
    for row in sorted(rows, key=lambda r: (r.output_tokens_per_s, r.source_url, r.engine_version)):
        chosen.setdefault(row.concurrency, row)
    return chosen


def log_interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation of y in log(x) between (x0, y0) and (x1, y1)."""
    weight = math.log(x / x0) / math.log(x1 / x0)
    return y0 + weight * (y1 - y0)


def _latencies(batch: int, lo: BenchmarkRow, hi: BenchmarkRow) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in LATENCY_FIELDS:
        y0, y1 = getattr(lo, name), getattr(hi, name)
        if y0 is None or y1 is None:
            values[name] = None
        elif lo is hi:
            values[name] = y0
        else:
            values[name] = log_interp(batch, lo.concurrency, hi.concurrency, y0, y1)
    return values


def _at_batch(
    points: dict[int, BenchmarkRow], batch: int
) -> tuple[BenchmarkRow, BenchmarkRow, float, Confidence, str]:
    """Bracketing rows, output tokens/s, confidence, and a note, at concurrency `batch`.

    Callers cap `batch` at the largest concurrency. Below the smallest, the smallest row's
    per-sequence rate and latencies are used (conservative: smaller batches run no slower).
    """
    concurrencies = sorted(points)
    if batch in points:
        row = points[batch]
        return row, row, row.output_tokens_per_s, "measured", ""
    if batch < concurrencies[0]:
        row = points[concurrencies[0]]
        note = (
            f"effective batch {batch} below the smallest benchmarked {row.concurrency}: "
            "clamped to its per-sequence rate and latencies"
        )
        return row, row, row.output_tokens_per_s * batch / row.concurrency, "interpolated", note
    lo = points[max(c for c in concurrencies if c < batch)]
    hi = points[min(c for c in concurrencies if c > batch)]
    tokens_per_s = log_interp(
        batch, lo.concurrency, hi.concurrency, lo.output_tokens_per_s, hi.output_tokens_per_s
    )
    note = f"interpolated in log(concurrency) between {lo.concurrency} and {hi.concurrency}"
    return lo, hi, tokens_per_s, "interpolated", note


def _evaluate(
    table: BenchmarkTable,
    model: ModelSpec,
    gpu: GPUSpec,
    config: ReplicaConfig,
    stats: StatsLike,
) -> PerfEstimate | str:
    rows = _matching_rows(table, model, gpu, config)
    if not rows:
        return (
            f"no benchmark rows for {model.id} on {gpu.id} at tensor_parallel "
            f"{config.tensor_parallel}, dtype {config.dtype}"
        )
    fit = fit_or_reason(model, gpu, config)
    if isinstance(fit, str):
        return fit
    engines = sorted({row.engine for row in rows})
    engine = PREFERRED_ENGINE if PREFERRED_ENGINE in engines else engines[0]
    rows = [row for row in rows if row.engine == engine]
    want_in, want_out = stats.input_tokens_mean, max(stats.output_tokens_mean, 1.0)
    shape_in, shape_out = _nearest_shape(rows, want_in, want_out)
    ratios = (shape_in / want_in, shape_out / want_out)
    if any(r > MAX_SHAPE_RATIO or r < 1 / MAX_SHAPE_RATIO for r in ratios):
        return (
            f"nearest benchmark shape {shape_in}/{shape_out} input/output tokens is more than "
            f"{MAX_SHAPE_RATIO:g}x from the workload mean {want_in:g}/{want_out:g}"
        )
    points = _by_concurrency(
        [r for r in rows if (r.input_len, r.output_len) == (shape_in, shape_out)]
    )
    notes = [
        f"benchmark rows: engine {engine}, shape {shape_in}/{shape_out} input/output tokens "
        f"(workload mean {want_in:g}/{want_out:g}); KV-cache dtype not matched",
    ]
    if len(engines) > 1:
        notes.append(f"rows from engines {', '.join(engines)}; {engine} preferred")
    ctx = avg_ctx_tokens(stats)
    wanted = effective_batch(fit.kv_token_capacity, config.max_num_seqs, ctx)
    batch = min(wanted, max(points))
    if batch < wanted:
        notes.append(f"effective batch {wanted} capped at the largest benchmarked {batch}")

    lo, hi, tokens_per_s, confidence, note = _at_batch(points, batch)
    if note:
        notes.append(note)
    lat = _latencies(batch, lo, hi)

    tpot_p50 = lat["tpot_ms_p50"]
    if tpot_p50 is None:
        tpot_p50 = batch / tokens_per_s * 1e3
        notes.append("TPOT p50 derived as effective batch / output tokens per second")
    tpot_p95 = lat["tpot_ms_p95"]
    if tpot_p95 is None:
        tpot_p95 = tpot_p50 * P95_FACTOR
        notes.append(f"TPOT p95 = p50 x {P95_FACTOR:g} (rows carry no p95)")
    ttft_p50 = lat["ttft_ms_p50"]
    if ttft_p50 is not None:
        prefill = shape_in / (ttft_p50 / 1e3)
        notes.append("prefill rate = benchmark input_len / TTFT p50")
    else:
        tflops = dense_tflops(gpu, config.dtype)
        if tflops is None:
            return f"rows carry no TTFT and gpu {gpu.id} has {tflops_field(config.dtype)} null"
        prefill = prefill_tokens_per_s(tflops, config.tensor_parallel, param_count(model))
        notes.append("prefill rate from the roofline model (rows carry no TTFT)")
    notes.append(SERVICE_NOTE)
    service_s = stats.input_tokens_mean / prefill + stats.output_tokens_mean * tpot_p50 / 1e3
    return PerfEstimate(
        backend="table",
        confidence=confidence,
        effective_batch=batch,
        decode_tokens_per_s=tokens_per_s,
        prefill_tokens_per_s=prefill,
        requests_per_s_capacity=batch / service_s,
        ttft_ms_p50=stats.input_tokens_p50 / prefill * 1e3,
        ttft_ms_p95=stats.input_tokens_p95 / prefill * 1e3,
        tpot_ms_p50=tpot_p50,
        tpot_ms_p95=tpot_p95,
        assumptions=tuple(notes),
        source_urls=tuple(sorted({lo.source_url, hi.source_url})),
    )


@register("table")
class TableBackend:
    """Interpolates published benchmark rows; `None` when no row matches closely enough.

    `table` defaults to the shipped, validated `data/benchmarks` table (loaded on first use).
    Confidence is `"measured"` on an exact concurrency hit, `"interpolated"` otherwise.
    """

    name = "table"

    def __init__(self, table: BenchmarkTable | None = None) -> None:
        self._table = table

    @property
    def table(self) -> BenchmarkTable:
        return self._table if self._table is not None else default_table()

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        result = _evaluate(self.table, model, gpu, config, stats)
        return result if isinstance(result, PerfEstimate) else None

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        result = _evaluate(self.table, model, gpu, config, stats)
        return result if isinstance(result, str) else "an estimate is available"
