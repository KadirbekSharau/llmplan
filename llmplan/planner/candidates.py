"""Candidate enumeration, pre-solve evaluation, and dominance pruning (M4_DESIGN.md section 4).

A candidate is one price row with one replica configuration. Each is checked in order:
tensor parallelism against the instance and the model, memory fit (M1), a performance
estimate (M3), then the SLO. Survivors are `eligible`. Pruning then drops eligible
candidates that another candidate on the same row and tensor parallelism beats on both
capacities; the MILP only sees (row, tp, capacities), so pruning never changes its optimum.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec
from llmplan.errors import PerfError
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, FitResult, fit
from llmplan.perf import ReplicaConfig, estimate
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.planner.result import CandidateEval, Status
from llmplan.types import DType


def rows_in_scope(prices: Sequence[PriceRow], options: PlanOptions) -> tuple[PriceRow, ...]:
    """Price rows matching `options.gpu_ids`, `providers`, and `commitments`, in input order."""
    return tuple(
        row
        for row in prices
        if (options.gpu_ids is None or row.gpu_id in options.gpu_ids)
        and (options.providers is None or row.provider in options.providers)
        and row.commitment in options.commitments
    )


def _rejected(
    row: PriceRow,
    config: ReplicaConfig,
    status: Status,
    reason: str,
    fit_result: FitResult | None = None,
    replicas: int = 0,
) -> CandidateEval:
    return CandidateEval(
        price_row=row,
        config=config,
        replicas_per_instance=replicas,
        fit=fit_result,
        perf=None,
        status=status,
        reason=reason,
        usd_per_hour_per_rps=None,
    )


def _tp_problem(model: ModelSpec, row: PriceRow, tp: int) -> tuple[Status, str] | None:
    if tp > row.gpu_count:
        return "tp_gt_gpus", f"tensor_parallel {tp} > gpu_count {row.gpu_count} of {row.instance}"
    if row.gpu_count % tp:
        # The GPU-count packing constraint is exact only when tp divides the instance.
        return "tp_gt_gpus", (
            f"tensor_parallel {tp} does not divide gpu_count {row.gpu_count} of {row.instance}"
        )
    if model.num_attention_heads % tp:
        return "no_fit", (
            f"tensor_parallel {tp} does not divide num_attention_heads "
            f"{model.num_attention_heads} of {model.id}"
        )
    return None


def evaluate_one(
    request: PlanRequest, row: PriceRow, gpu: GPUSpec, config: ReplicaConfig
) -> CandidateEval:
    """Run the section 4 checks for one candidate and return its evaluation."""
    model, slo, tp = request.model, request.slo, config.tensor_parallel
    problem = _tp_problem(model, row, tp)
    if problem is not None:
        return _rejected(row, config, *problem)
    replicas = row.gpu_count // tp
    fit_result = fit(
        FitRequest(
            model=model,
            gpu=gpu,
            engine=request.engine,
            tensor_parallel=tp,
            dtype=config.dtype,
            context_len=config.max_model_len,
        )
    )
    if not fit_result.fits:
        reason = (
            f"{model.id} {config.dtype} does not fit on {gpu.id} x{tp} at max_model_len "
            f"{config.max_model_len} (binding: {fit_result.binding})"
        )
        return _rejected(row, config, "no_fit", reason, fit_result, replicas)
    try:
        perf = estimate(model, gpu, config, request.stats, backend=request.options.perf_backend)
    except PerfError as exc:
        return _rejected(row, config, "no_perf", str(exc), fit_result, replicas)
    if slo.ttft_ms_p95 is not None and perf.ttft_ms_p95 > slo.ttft_ms_p95:
        status: Status = "slo_ttft"
        reason = f"TTFT p95 {perf.ttft_ms_p95:g} ms > SLO {slo.ttft_ms_p95:g} ms"
    elif slo.tpot_ms_p95 is not None and perf.tpot_ms_p95 > slo.tpot_ms_p95:
        status = "slo_tpot"
        reason = f"TPOT p95 {perf.tpot_ms_p95:g} ms > SLO {slo.tpot_ms_p95:g} ms"
    else:
        status = "eligible"
        derated = perf.requests_per_s_capacity * slo.utilization_target
        reason = (
            f"eligible: {replicas} replica(s) per instance, {derated:,.2f} req/s and "
            f"{perf.decode_tokens_per_s * slo.utilization_target:,.2f} output tokens/s each "
            f"(derated), perf {perf.backend}/{perf.confidence}"
        )
    usd_per_rps = (
        row.price_usd_per_hour / (replicas * perf.requests_per_s_capacity * slo.utilization_target)
        if status == "eligible"
        else None
    )
    return CandidateEval(
        price_row=row,
        config=config,
        replicas_per_instance=replicas,
        fit=fit_result,
        perf=perf,
        status=status,
        reason=reason,
        usd_per_hour_per_rps=usd_per_rps,
    )


def replica_config(
    engine: EngineProfile, options: PlanOptions, tp: int, dtype: DType, seqs: int
) -> ReplicaConfig:
    """The `ReplicaConfig` for one choice tuple; engine settings come from `engine`."""
    return ReplicaConfig(
        tensor_parallel=tp,
        dtype=dtype,
        kv_dtype=engine.kv_dtype,
        max_num_seqs=seqs,
        max_model_len=options.max_model_len,
        gpu_memory_utilization=engine.gpu_memory_utilization,
        max_num_batched_tokens=engine.max_num_batched_tokens,
    )


def evaluate_candidates(
    request: PlanRequest, rows: Sequence[PriceRow], gpus: Mapping[str, GPUSpec]
) -> tuple[CandidateEval, ...]:
    """Evaluate every row x tp x dtype x max_num_seqs combination, in enumeration order."""
    opts = request.options
    return tuple(
        evaluate_one(request, row, gpus[row.gpu_id], replica_config(request.engine, opts, *choice))
        for row in rows
        for choice in itertools.product(
            opts.tensor_parallel_choices, opts.dtype_choices, opts.max_num_seqs_choices
        )
    )


@dataclass(frozen=True)
class Column:
    """An eligible candidate as the MILP and the baseline see it: derated request and
    output-token capacity per replica, and per-GPU VRAM use (for pruning ties)."""

    candidate: CandidateEval
    rps: float
    tps: float
    vram_bytes: int

    @property
    def tp(self) -> int:
        return self.candidate.config.tensor_parallel


def vram_bytes_needed(fit_result: FitResult, config: ReplicaConfig) -> int:
    """Per-GPU bytes a replica can occupy: weights, overhead, and the KV cache its
    `max_num_seqs` full-length sequences need (capped by the KV budget)."""
    kv_tokens = min(fit_result.kv_token_capacity, config.max_num_seqs * config.max_model_len)
    return (
        fit_result.per_gpu_weight_bytes
        + fit_result.per_gpu_overhead_bytes
        + kv_tokens * fit_result.kv_bytes_per_token_per_gpu
    )


def columns(candidates: Sequence[CandidateEval], slo: SLO) -> tuple[Column, ...]:
    """The eligible candidates, in order, with capacities derated by `utilization_target`."""
    return tuple(
        Column(
            candidate=c,
            rps=c.perf.requests_per_s_capacity * slo.utilization_target,
            tps=c.perf.decode_tokens_per_s * slo.utilization_target,
            vram_bytes=vram_bytes_needed(c.fit, c.config),
        )
        for c in candidates
        if c.status == "eligible" and c.perf is not None and c.fit is not None
    )


def _dominates(a: Column, b: Column, a_first: bool) -> bool:
    key_a, key_b = (a.rps, a.tps, -a.vram_bytes), (b.rps, b.tps, -b.vram_bytes)
    if any(x < y for x, y in zip(key_a, key_b, strict=True)):
        return False
    return key_a != key_b or a_first


def prune_dominated(eligible: Sequence[Column]) -> tuple[Column, ...]:
    """Drop columns dominated within their (price row, tensor parallel) group.

    `b` is dominated by `a` when `a` has at least `b`'s request and token capacity and at
    most its VRAM use, and is strictly better in one of them; of exact ties the first in
    enumeration order is kept. The survivors keep their order.
    """
    kept = []
    for i, b in enumerate(eligible):
        group = (b.candidate.price_row, b.tp)
        if not any(
            j != i and (a.candidate.price_row, a.tp) == group and _dominates(a, b, a_first=j < i)
            for j, a in enumerate(eligible)
        ):
            kept.append(b)
    return tuple(kept)
