"""MILP fleet planner (M4): `plan(PlanRequest) -> PlanResult` (ARCHITECTURE.md section 5).

Pipeline: validate, enumerate and evaluate candidates (per request-size class too, M7),
prune dominated ones, build and solve the MILP, explain the binding constraint with an LP
relaxation, and compare against the best homogeneous fleet.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmplan import perf
from llmplan.errors import CatalogError, InfeasiblePlan, SolverError, ValidationError
from llmplan.planner import classes, explain
from llmplan.planner.candidates import columns, evaluate_candidates, prune_dominated, rows_in_scope
from llmplan.planner.model import HOURS_PER_DAY, build_model
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.planner.result import (
    CandidateEval,
    FleetItem,
    PlanResult,
    SolverInfo,
    binding_label,
    infeasible_reason,
)
from llmplan.planner.solve import INTEGER_ONLY, TIGHT_TOLERANCE, check_backend
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


def plan(request: PlanRequest) -> PlanResult:
    """Find the cheapest fleet and replica configurations meeting peak demand and the SLO.

    Demand is the workload's peak window (`peak_window_rps`, `peak_output_tokens_per_s`),
    or, with `request.classes` (M7), each class's peak demand, which only candidates meeting
    that class's SLO may serve; each candidate's capacity is derated by
    `slo.utilization_target`. Raises `ValidationError` (`max_model_len` too long),
    `CatalogError` (unknown GPU id), `UnknownRegistryKey` (unknown perf backend),
    `InfeasiblePlan` with a reason built from the candidate statuses, and `SolverError`
    (backend unavailable, time limit without a fleet). Same request, same result: solves
    are single-threaded, seeded, time-limited.
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
    class_perf = None
    if request.classes:
        evals = classes.evaluate_classes(request, candidates, request.gpus)
        candidates = tuple(
            classes.reconcile(c, e, slo.utilization_target)
            for c, e in zip(candidates, evals, strict=True)
        )
        class_perf = [
            None if e is None else tuple(x.perf if x.status == "eligible" else None for x in e)
            for e in evals
        ]
    cols = columns(candidates, slo, class_perf)
    if not cols:
        raise _infeasible(infeasible_reason(candidates))
    if request.classes:
        missing = classes.unserved(request, evals, candidates)
        if missing is not None:
            raise _infeasible(missing)
    demands = classes.demands(request)
    kept = prune_dominated(cols)
    demand_rps, demand_tps = stats.peak_window_rps, stats.peak_output_tokens_per_s
    formulation = build_model(
        kept,
        demand_rps=demands[0][0],
        demand_tps=demands[0][1],
        homogeneous=opts.homogeneous,
        max_instances_per_row=opts.max_instances_per_row,
        integer_scaling=opts.solver in INTEGER_ONLY,
        class_demands=demands,
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
    replicas = explain.replica_plans(counts)
    fleet = tuple(
        FleetItem(
            price_row=row, instances=n, usd_per_day=HOURS_PER_DAY * row.price_usd_per_hour * n
        )
        for row, n in zip(formulation.rows, solution.instances, strict=True)
        if n > 0
    )
    cost = sum(item.usd_per_day for item in fleet)
    used = [(col, k) for col, k in counts if k > 0]
    fleet_capacity = explain.capacity(used, demands)
    _check_demand(fleet_capacity, demands, opts.solver)
    relax = explain.relax([col for col, _ in used], demands, opts.max_instances_per_row)
    baseline = None if opts.homogeneous else explain.baseline(request, cols, demands)
    saving = None
    if baseline is not None:
        saving = (baseline.cost_usd_per_day - cost) / baseline.cost_usd_per_day * 100
    assumptions = explain.assumptions(request, replicas)
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
    assumptions.append(
        "headroom of the whole-instance fleet over demand: requests "
        f"{explain.headroom(fleet_capacity.rps, demand_rps)}, tokens "
        f"{explain.headroom(fleet_capacity.tps, demand_tps)}"
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
        capacity_rps=fleet_capacity.rps,
        capacity_output_tokens_per_s=fleet_capacity.tps,
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


def _check_demand(
    fleet: explain.Capacity, demands: Sequence[tuple[float, float]], solver: str
) -> None:
    """Refuse a fleet below demand (beyond 1e-6 relative): one class against its two
    demands, several against the routing LP's uniform headroom."""
    if fleet.allocation is not None:
        short = fleet.allocation.theta < 1 - TIGHT_TOLERANCE
    else:
        (demand_rps, demand_tps), *_ = demands
        short = fleet.rps < demand_rps * (1 - TIGHT_TOLERANCE) or fleet.tps < demand_tps * (
            1 - TIGHT_TOLERANCE
        )
    if short:
        raise SolverError(f"{solver} returned a fleet below demand ({fleet.rps:g} req/s)")
