"""Explaining a fleet (M4_DESIGN.md section 7, M7_DESIGN.md section 4).

The binding demand constraints come from the LP relaxation over the columns the fleet
uses; capacity is summed over replicas (with request-size classes: what the routing LP
allocates to each class); the assumption lines restate every modeling choice; and the
best homogeneous fleet is turned into the baseline `PlanResult`. Moved out of
`llmplan.planner` (M4's `plan()` module) when M7 added classes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llmplan.planner.baseline import HomogeneousFleet, best_homogeneous
from llmplan.planner.candidates import Column
from llmplan.planner.classes import routing
from llmplan.planner.model import build_model
from llmplan.planner.request import PlanRequest
from llmplan.planner.result import (
    Binding,
    FleetItem,
    PlanResult,
    ReplicaPlan,
    SolverInfo,
    binding_label,
    label,
)
from llmplan.planner.solve import Allocation, Relaxation, balance, relaxation

Demands = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class Capacity:
    """Derated capacity of a fleet: request and output-token rates summed over replicas,
    and (with two or more classes) the routing LP's allocation they come from."""

    rps: float
    tps: float
    allocation: Allocation | None


def capacity(counts: Sequence[tuple[Column, int]], demands: Demands) -> Capacity:
    """With one class, `sum cap * count` (M4). With classes, the capacity the routing LP
    gives each class over this fleet, summed: `sum_k sum_r x_{r,k} * cap_{r,k}`."""
    if len(demands) < 2:
        return Capacity(
            rps=sum(col.rates()[0][0] * k for col, k in counts),
            tps=sum(col.rates()[0][1] * k for col, k in counts),
            allocation=None,
        )
    cols = [col for col, _ in counts]
    allocation = balance(cols, [k for _, k in counts], demands)
    rows = list(zip(cols, allocation.x, strict=True))
    return Capacity(
        rps=sum(r * x for col, xs in rows for (r, _), x in zip(col.rates(), xs, strict=True)),
        tps=sum(t * x for col, xs in rows for (_, t), x in zip(col.rates(), xs, strict=True)),
        allocation=allocation,
    )


def relax(used: Sequence[Column], demands: Demands, max_instances_per_row: int) -> Relaxation:
    """LP relaxation over the columns the fleet uses (see M4_NOTES.md, binding)."""
    return relaxation(
        build_model(
            used,
            demand_rps=demands[0][0],
            demand_tps=demands[0][1],
            homogeneous=False,
            max_instances_per_row=max_instances_per_row,
            relax=True,
            class_demands=demands,
        )
    )


def class_binding(request: PlanRequest, relaxed: Relaxation) -> tuple[Binding, ...]:
    """Each class's binding label (empty without classes)."""
    if not request.classes:
        return ()
    return tuple(binding_label(r, t) for r, t in relaxed.classes)


def binding_lines(request: PlanRequest, relaxed: Relaxation) -> list[str]:
    """How the LP relaxation over the chosen candidates explains the fleet: tight or slack
    demand constraints and their shadow prices (USD/day per unit), per class with two or
    more classes."""

    def pair(k: int) -> str:
        (r_tight, t_tight), (r_price, t_price) = relaxed.classes[k], relaxed.class_shadow_prices[k]
        return (
            f"request demand {'tight' if r_tight else 'slack'} (shadow price "
            f"${r_price + 0.0:.4g}/day per req/s), token demand "
            f"{'tight' if t_tight else 'slack'} (shadow price ${t_price + 0.0:.4g}/day per "
            "output token/s)"
        )

    if len(request.classes) < 2:
        return [f"binding from the LP relaxation over the chosen candidates: {pair(0)}"]
    return [
        f"binding from the LP relaxation over the chosen candidates, class {k}: {pair(k)}"
        for k in range(len(request.classes))
    ]


def class_lines(request: PlanRequest, fleet: Capacity) -> list[str]:
    """Assumption lines for two or more request-size classes (none otherwise)."""
    if len(request.classes) < 2 or fleet.allocation is None:
        return []
    lines = [
        f"demand split into {len(request.classes)} request-size classes, each sized for its "
        "requests in the peak windows; a candidate serves only the classes whose SLO it meets, "
        "with capacity estimated at each class's token shape",
        "routing weights come from an LP over the chosen fleet that maximizes the uniform "
        f"headroom: every class gets {fleet.allocation.theta:.4g} x its peak demand",
    ]
    for c in request.classes:
        lines.extend(f"class {c.index}: {note}" for note in c.notes)
    return lines


def headroom(capacity: float, demand: float) -> str:
    """`"+12.5%"`: how far `capacity` exceeds `demand`."""
    if demand <= 0:
        return "n/a (no demand)"
    return f"+{max(0.0, (capacity / demand - 1) * 100):.1f}%"


def replica_plans(counts: Sequence[tuple[Column, int]]) -> tuple[ReplicaPlan, ...]:
    """A `ReplicaPlan` per column with a positive count, in column order."""
    return tuple(
        ReplicaPlan(
            candidate=col.candidate,
            count=count,
            instances=math.ceil(count * col.tp / col.candidate.price_row.gpu_count),
        )
        for col, count in counts
        if count > 0
    )


def assumptions(request: PlanRequest, replicas: Sequence[ReplicaPlan]) -> list[str]:
    """The assumption lines every result carries (M4_DESIGN.md section 7)."""
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


def baseline(request: PlanRequest, cols: Sequence[Column], demands: Demands) -> PlanResult | None:
    """The best homogeneous fleet as a `PlanResult`, or None when no single column can
    meet demand within `max_instances_per_row`."""
    found = best_homogeneous(
        cols,
        demands[0][0],
        demands[0][1],
        request.options.max_instances_per_row,
        class_demands=demands,
    )
    if found is None:
        return None
    return _homogeneous_result(request, found, demands)


def _homogeneous_result(
    request: PlanRequest, found: HomogeneousFleet, demands: Demands
) -> PlanResult:
    col, stats = found.column, request.stats
    relaxed = relax([col], demands, request.options.max_instances_per_row)
    counts = [(col, found.replicas)]
    replicas = replica_plans(counts)
    fleet_capacity = capacity(counts, demands)
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
        capacity_rps=fleet_capacity.rps,
        capacity_output_tokens_per_s=fleet_capacity.tps,
        binding=binding_label(relaxed.requests_tight, relaxed.tokens_tight),
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
            *assumptions(request, replicas),
        ),
        classes=request.classes,
        routing=routing(counts, fleet_capacity.allocation, len(request.classes)),
        class_binding=class_binding(request, relaxed),
    )
