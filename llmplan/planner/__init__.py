"""MILP fleet planner (M4): `plan(PlanRequest) -> PlanResult` (ARCHITECTURE.md section 5).

Pipeline: validate, enumerate and evaluate candidates, prune dominated ones, build and
solve the MILP, explain the binding constraint with an LP relaxation, and compare against
the best homogeneous fleet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmplan import perf
from llmplan.errors import CatalogError, InfeasiblePlan, SolverError, ValidationError
from llmplan.planner.baseline import HomogeneousFleet, best_homogeneous
from llmplan.planner.candidates import (
    Column,
    columns,
    evaluate_candidates,
    prune_dominated,
    rows_in_scope,
)
from llmplan.planner.model import HOURS_PER_DAY, build_model
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.planner.result import (
    CandidateEval,
    FleetItem,
    PlanResult,
    ReplicaPlan,
    SolverInfo,
    binding_label,
    infeasible_reason,
    label,
)
from llmplan.planner.solve import INTEGER_ONLY, TIGHT_TOLERANCE, Relaxation, check_backend
from llmplan.planner.solve import relaxation as solve_relaxation
from llmplan.planner.solve import solve as solve_model

__all__ = ["SLO", "PlanOptions", "PlanRequest", "PlanResult", "plan"]


def _validate(request: PlanRequest) -> None:
    model, opts = request.model, request.options
    if opts.max_model_len > model.max_position_embeddings:
        raise ValidationError(
            f"max_model_len {opts.max_model_len} exceeds max_position_embeddings "
            f"{model.max_position_embeddings} of {model.id}"
        )
    for gpu_id in opts.gpu_ids or ():
        if gpu_id not in request.gpus:
            raise CatalogError(f"unknown gpu id {gpu_id!r}; known: {', '.join(request.gpus)}")
    for index, row in enumerate(request.prices):
        if row.gpu_id not in request.gpus:
            raise CatalogError(f"price row {index} ({row.instance}): unknown gpu_id {row.gpu_id!r}")
    if opts.perf_backend != "auto":
        perf.get(opts.perf_backend)
    check_backend(opts.solver)


def _infeasible(reason: str) -> InfeasiblePlan:
    return InfeasiblePlan(f"no feasible fleet: {reason}", reason=reason)


def _sorted(candidates: Sequence[CandidateEval]) -> tuple[CandidateEval, ...]:
    """Eligible by ascending `usd_per_hour_per_rps`, then rejected, each in stable order."""
    return tuple(
        sorted(
            candidates,
            key=lambda c: (c.usd_per_hour_per_rps is None, c.usd_per_hour_per_rps or 0.0),
        )
    )


def _relax(used: Sequence[Column], request: PlanRequest) -> Relaxation:
    """LP relaxation over the columns the fleet uses (see M4_NOTES.md, binding)."""
    stats = request.stats
    return solve_relaxation(
        build_model(
            used,
            demand_rps=stats.peak_window_rps,
            demand_tps=stats.peak_output_tokens_per_s,
            homogeneous=False,
            max_instances_per_row=request.options.max_instances_per_row,
            relax=True,
        )
    )


def _replicas(counts: Sequence[tuple[Column, int]]) -> tuple[ReplicaPlan, ...]:
    return tuple(
        ReplicaPlan(
            candidate=col.candidate,
            count=count,
            instances=math.ceil(count * col.tp / col.candidate.price_row.gpu_count),
        )
        for col, count in counts
        if count > 0
    )


def _assumptions(request: PlanRequest, replicas: Sequence[ReplicaPlan]) -> list[str]:
    stats, slo = request.stats, request.slo
    confidence = "; ".join(
        f"{label(r.candidate)}: {r.candidate.perf.backend}/{r.candidate.perf.confidence}"
        for r in replicas
        if r.candidate.perf is not None
    )
    return [
        f"capacity derated to {slo.utilization_target:g} x the estimated throughput "
        "(utilization_target)",
        "GPUs left idle on a paid instance are paid for (leftover-GPU waste)",
        f"sized for the peak {stats.window_s:g} s window ({stats.peak_window_rps:g} req/s, "
        f"{stats.peak_output_tokens_per_s:g} output tokens/s); no autoscaling in M4",
        "TTFT and TPOT are p95 service-time estimates without queueing (queueing is M5)",
        f"perf confidence of chosen replicas: {confidence}",
    ]


def _baseline(request: PlanRequest, cols: Sequence[Column]) -> PlanResult | None:
    stats = request.stats
    found = best_homogeneous(
        cols,
        stats.peak_window_rps,
        stats.peak_output_tokens_per_s,
        request.options.max_instances_per_row,
    )
    if found is None:
        return None
    return _homogeneous_result(request, found)


def _homogeneous_result(request: PlanRequest, found: HomogeneousFleet) -> PlanResult:
    col, stats = found.column, request.stats
    relax = _relax([col], request)
    replicas = _replicas([(col, found.replicas)])
    return PlanResult(
        fleet=(
            FleetItem(
                price_row=col.candidate.price_row,
                instances=found.instances,
                usd_per_day=found.usd_per_day,
            ),
        ),
        replicas=replicas,
        cost_usd_per_day=found.usd_per_day,
        baseline=None,
        baseline_saving_pct=None,
        demand_rps=stats.peak_window_rps,
        demand_output_tokens_per_s=stats.peak_output_tokens_per_s,
        capacity_rps=col.rps * found.replicas,
        capacity_output_tokens_per_s=col.tps * found.replicas,
        binding=binding_label(relax.requests_tight, relax.tokens_tight),
        candidates=(),
        solver=SolverInfo(
            backend="enumeration",
            status="optimal",
            objective_usd_per_day=found.usd_per_day,
            best_bound_usd_per_day=found.usd_per_day,
            solve_time_s=0.0,
            n_variables=0,
            n_constraints=0,
        ),
        assumptions=(
            "baseline: best fleet of one price row and one replica configuration, by "
            "enumeration; candidates are listed on the main result",
            *_assumptions(request, replicas),
        ),
    )


def plan(request: PlanRequest) -> PlanResult:
    """Find the cheapest fleet and replica configurations meeting peak demand and the SLO.

    Demand is the workload's peak window (`peak_window_rps`, `peak_output_tokens_per_s`);
    each candidate's capacity is derated by `slo.utilization_target`. Raises
    `ValidationError` (`max_model_len` too long), `CatalogError` (unknown GPU id),
    `UnknownRegistryKey` (unknown perf backend), `InfeasiblePlan` with a reason built from
    the candidate statuses, and `SolverError` (backend unavailable, time limit without a
    fleet). Same request, same result: solves are single-threaded, seeded, time-limited.
    """
    _validate(request)
    opts, slo, stats = request.options, request.slo, request.stats
    rows = rows_in_scope(request.prices, opts)
    if not rows:
        raise _infeasible(
            f"no price rows match gpu_ids {opts.gpu_ids or 'any'}, providers "
            f"{opts.providers or 'any'}, commitments {list(opts.commitments)}"
        )
    candidates = evaluate_candidates(request, rows, request.gpus)
    cols = columns(candidates, slo)
    if not cols:
        raise _infeasible(infeasible_reason(candidates))
    kept = prune_dominated(cols)
    demand_rps, demand_tps = stats.peak_window_rps, stats.peak_output_tokens_per_s
    formulation = build_model(
        kept,
        demand_rps=demand_rps,
        demand_tps=demand_tps,
        homogeneous=opts.homogeneous,
        max_instances_per_row=opts.max_instances_per_row,
        integer_scaling=opts.solver in INTEGER_ONLY,
    )
    try:
        solution = solve_model(
            formulation, opts.solver, time_limit_s=opts.time_limit_s, seed=opts.seed
        )
    except InfeasiblePlan:
        scope = " on a single price row" if opts.homogeneous else ""
        raise _infeasible(
            f"{len(cols)} of {len(candidates)} candidates eligible, but no fleet of at most "
            f"{opts.max_instances_per_row} instances per price row{scope} meets "
            f"{demand_rps:g} req/s and {demand_tps:g} output tokens/s"
        ) from None
    counts = list(zip(kept, solution.replicas, strict=True))
    replicas = _replicas(counts)
    fleet = tuple(
        FleetItem(
            price_row=row, instances=n, usd_per_day=HOURS_PER_DAY * row.price_usd_per_hour * n
        )
        for row, n in zip(formulation.rows, solution.instances, strict=True)
        if n > 0
    )
    cost = sum(item.usd_per_day for item in fleet)
    capacity_rps = sum(col.rps * k for col, k in counts)
    capacity_tps = sum(col.tps * k for col, k in counts)
    if capacity_rps < demand_rps * (1 - TIGHT_TOLERANCE) or capacity_tps < demand_tps * (
        1 - TIGHT_TOLERANCE
    ):
        raise SolverError(f"{opts.solver} returned a fleet below demand ({capacity_rps:g} req/s)")
    relax = _relax([col for col, k in counts if k > 0], request)
    baseline = None if opts.homogeneous else _baseline(request, cols)
    saving = None
    if baseline is not None:
        saving = (baseline.cost_usd_per_day - cost) / baseline.cost_usd_per_day * 100
    assumptions = _assumptions(request, replicas)
    assumptions.append(
        f"dominance pruning removed {len(cols) - len(kept)} of {len(cols)} eligible candidates"
    )
    assumptions.append(
        "binding from the LP relaxation over the chosen candidates: request demand "
        f"{'tight' if relax.requests_tight else 'slack'} (shadow price "
        f"${relax.requests_shadow_price + 0.0:.4g}/day per req/s), token demand "
        f"{'tight' if relax.tokens_tight else 'slack'} (shadow price "
        f"${relax.tokens_shadow_price + 0.0:.4g}/day per output token/s)"
    )
    if solution.status == "feasible_time_limit":
        assumptions.append(
            f"time limit {opts.time_limit_s:g} s reached: the fleet is feasible but not "
            "proven optimal"
        )
    return PlanResult(
        fleet=fleet,
        replicas=replicas,
        cost_usd_per_day=cost,
        baseline=baseline,
        baseline_saving_pct=saving,
        demand_rps=demand_rps,
        demand_output_tokens_per_s=demand_tps,
        capacity_rps=capacity_rps,
        capacity_output_tokens_per_s=capacity_tps,
        binding=binding_label(relax.requests_tight, relax.tokens_tight),
        candidates=_sorted(candidates),
        solver=SolverInfo(
            backend=opts.solver,
            status=solution.status,
            objective_usd_per_day=cost,
            best_bound_usd_per_day=solution.best_bound_usd_per_day,
            solve_time_s=solution.solve_time_s,
            n_variables=solution.n_variables,
            n_constraints=solution.n_constraints,
        ),
        assumptions=tuple(assumptions),
    )
