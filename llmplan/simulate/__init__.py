"""Trace replay on a planned fleet (M5): `replay()` returns a `Timeline`.

Public API: `replay` (ARCHITECTURE.md section 5), `replay_requests` (the per-request
records behind a timeline), and the models `SimOptions`, `Timeline`, `WindowRecord`,
`ReplicaWindowRecord`, `SimulationSummary`, `RequestLog`.
"""

from __future__ import annotations

from llmplan.planner.request import SLO
from llmplan.planner.result import PlanResult
from llmplan.simulate import routing
from llmplan.simulate.events import EngineRun, RequestLog, run_events
from llmplan.simulate.replica import ReplicaSpec, replica_specs
from llmplan.simulate.timeline import (
    ReplicaWindowRecord,
    SimOptions,
    SimulationSummary,
    Timeline,
    WindowRecord,
    build_timeline,
)
from llmplan.workload import Workload


def replay(
    plan: PlanResult,
    workload: Workload,
    *,
    slo: SLO | None = None,
    options: SimOptions | None = None,
) -> Timeline:
    """Replay `workload` on the fleet of `plan` and aggregate what happens per window.

    Every `ReplicaPlan` becomes `count` replicas with `perf.effective_batch` slots, the
    fit's KV-token capacity, and service times from the perf estimate (M5_DESIGN.md section
    4); arrivals are routed by `options.routing`. Unset budgets in `options` default to
    `slo.ttft_ms_p95` / `slo.tpot_ms_p95`; the timeline records the resolved options.
    Traces longer than `options.max_requests` are truncated with a note. Deterministic:
    the same inputs give byte-identical `model_dump_json()`. Raises `ValidationError` for
    a plan whose replicas cannot serve (no fit, no perf, no KV cache).
    """
    options = _resolve_budgets(options or SimOptions(), slo)
    specs, run, n_truncated = _run(plan, workload, options)
    return build_timeline(
        run,
        specs,
        capacity_rps=plan.capacity_rps,
        plan_cost_usd_per_day=plan.cost_usd_per_day,
        options=options,
        n_truncated=n_truncated,
        assumptions=_assumptions(plan, options, run, n_truncated),
    )


def replay_requests(
    plan: PlanResult, workload: Workload, *, options: SimOptions | None = None
) -> RequestLog:
    """The per-request records (`RequestLog`) of the same replay `replay` aggregates."""
    return _run(plan, workload, options or SimOptions())[1].requests


def _resolve_budgets(options: SimOptions, slo: SLO | None) -> SimOptions:
    if slo is None:
        return options
    ttft, tpot = options.ttft_budget_ms, options.tpot_budget_ms
    return options.model_copy(
        update={
            "ttft_budget_ms": slo.ttft_ms_p95 if ttft is None else ttft,
            "tpot_budget_ms": slo.tpot_ms_p95 if tpot is None else tpot,
        }
    )


def _run(
    plan: PlanResult, workload: Workload, options: SimOptions
) -> tuple[tuple[ReplicaSpec, ...], EngineRun, int]:
    specs = replica_specs(plan)
    frame = workload.frame.iloc[: options.max_requests]
    run = run_events(
        specs,
        frame["arrival_s"].to_numpy(),
        frame["input_tokens"].to_numpy(),
        frame["output_tokens"].to_numpy(),
        routing.get(options.routing),
    )
    return specs, run, len(workload.frame) - len(frame)


def _assumptions(
    plan: PlanResult, options: SimOptions, run: EngineRun, n_truncated: int
) -> tuple[str, ...]:
    confidences = sorted(
        {r.candidate.perf.confidence for r in plan.replicas if r.candidate.perf is not None}
    )
    notes = [
        f"service times from the plan's perf estimates (confidence: {', '.join(confidences)})",
        "time per output token is the estimate's full-batch tpot_ms_p50, constant: its "
        "dependence on batch size is not modeled (conservative)",
        "a request holds input + output KV tokens from admission to completion (full "
        "allocation, conservative); it starts only with a free slot and enough KV tokens",
        f"routing: {options.routing}; each replica serves its queue in FIFO order",
        f"windows of {options.window_s:g} s from the first arrival until the last completion; "
        "latency p95s and violations are over requests completed in each window",
        "capacity_rps is the plan's derated capacity, constant across windows",
    ]
    for kind, budget in (("TTFT", options.ttft_budget_ms), ("TPOT", options.tpot_budget_ms)):
        notes.append(
            f"no {kind} budget set: {kind} violations are not counted"
            if budget is None
            else f"{kind} budget {budget:g} ms (queueing included for TTFT)"
        )
    if n_truncated:
        notes.append(
            f"trace truncated to the first {options.max_requests:,} requests "
            f"({n_truncated:,} not simulated)"
        )
    if run.n_kv_capped:
        notes.append(
            f"{run.n_kv_capped:,} requests need more KV tokens than their replica holds; each "
            "was capped at the replica's KV capacity and ran alone"
        )
    return tuple(notes)


__all__ = [
    "ReplicaWindowRecord",
    "RequestLog",
    "SimOptions",
    "SimulationSummary",
    "Timeline",
    "WindowRecord",
    "replay",
    "replay_requests",
]
