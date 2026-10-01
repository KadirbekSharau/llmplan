"""Trace replay on a planned fleet (M5): `replay()` returns a `Timeline`.

Public API: `replay` (ARCHITECTURE.md section 5), `replay_requests` (the per-request
records behind a timeline), and the models `SimOptions`, `Timeline`, `WindowRecord`,
`ReplicaWindowRecord`, `SimulationSummary`, `RequestLog`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from llmplan.catalog.hardware import GPUSpec
from llmplan.errors import ValidationError
from llmplan.planner.request import SLO
from llmplan.planner.result import PlanResult
from llmplan.simulate import routing
from llmplan.simulate.events import EngineRun, RequestLog, Route, run_events
from llmplan.simulate.replica import ReplicaSpec, replica_specs
from llmplan.simulate.timeline import (
    ClassSummary,
    ReplicaWindowRecord,
    SimOptions,
    SimulationSummary,
    Timeline,
    WindowRecord,
    build_timeline,
)
from llmplan.workload import Workload
from llmplan.workload.classes import assign_classes

KV_NOTES = {
    "full": "a request holds input + output KV tokens from admission to completion (full "
    "allocation, conservative); it starts only with a free slot and enough KV tokens",
    "incremental": "a request reserves its mean KV occupancy (input + output / 2, as the "
    "planner assumes) and starts only with a free slot and that many unreserved KV tokens; "
    "its KV in use grows linearly from input at admission to input + output at completion "
    "(vLLM's incremental allocation, without preemption: in-use KV can briefly exceed the "
    "cache when many long requests finish together)",
}


def replay(
    plan: PlanResult,
    workload: Workload,
    *,
    slo: SLO | None = None,
    options: SimOptions | None = None,
    gpus: Mapping[str, GPUSpec] | None = None,
) -> Timeline:
    """Replay `workload` on the fleet of `plan` and aggregate what happens per window.

    Every `ReplicaPlan` becomes `count` replicas with `perf.effective_batch` slots, the
    fit's KV-token capacity, and service times from the perf estimate (M5_DESIGN.md section
    4); arrivals are routed by `options.routing` (M7: `class_weighted` follows the plan's
    routing weights) and KV is accounted per `options.kv_accounting`. A plan with classes
    gets one `ClassSummary` per class, whatever the routing. Unset budgets in `options` default to
    `slo.ttft_ms_p95` / `slo.tpot_ms_p95`; the timeline records the resolved options.
    Traces longer than `options.max_requests` are truncated with a note. `gpus` is the GPU
    catalog the plan was made with: it supplies each replica's total VRAM
    (`ReplicaWindowRecord.vram_bytes_total`; None, with a note, for a GPU it does not hold).
    Deterministic: the same inputs give byte-identical `model_dump_json()`. Raises
    `ValidationError` for a plan whose replicas cannot serve (no fit, no perf, no KV cache)
    or whose routing names a replica type it does not have.
    """
    options = _resolve_budgets(options or SimOptions(), slo)
    specs, run, n_truncated = _run(plan, workload, options, gpus)
    return build_timeline(
        run,
        specs,
        capacity_rps=plan.capacity_rps,
        plan_cost_usd_per_day=plan.cost_usd_per_day,
        options=options,
        n_truncated=n_truncated,
        assumptions=_assumptions(plan, options, run, n_truncated, gpus),
        n_classes=len(plan.classes),
    )


def replay_requests(
    plan: PlanResult, workload: Workload, *, options: SimOptions | None = None
) -> RequestLog:
    """The per-request records (`RequestLog`) of the same replay `replay` aggregates."""
    return _run(plan, workload, options or SimOptions(), None)[1].requests


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
    plan: PlanResult,
    workload: Workload,
    options: SimOptions,
    gpus: Mapping[str, GPUSpec] | None,
) -> tuple[tuple[ReplicaSpec, ...], EngineRun, int]:
    specs = replica_specs(plan, gpus)
    frame = workload.frame.iloc[: options.max_requests]
    inputs, outputs = frame["input_tokens"].to_numpy(), frame["output_tokens"].to_numpy()
    labels = assign_classes(plan.classes, inputs, outputs)
    route = (
        _class_weighted(plan, labels.tolist())
        if options.routing == "class_weighted"
        else routing.get(options.routing)
    )
    run = run_events(
        specs,
        frame["arrival_s"].to_numpy(),
        inputs,
        outputs,
        route,
        kv_accounting=options.kv_accounting,
        class_index=labels,
    )
    return specs, run, len(workload.frame) - len(frame)


def _class_weighted(plan: PlanResult, labels: Sequence[int]) -> Route:
    """The `class_weighted` policy for `plan`: replica types are its `ReplicaPlan`s; each
    routing rule is matched to the replica plan with the same candidate."""
    types = [i for i, replica in enumerate(plan.replicas) for _ in range(replica.count)]
    rules: list[list[tuple[int, float]]] = [[] for _ in range(max(1, len(plan.classes)))]
    for rule in plan.routing:
        found = [i for i, r in enumerate(plan.replicas) if r.candidate == rule.candidate]
        if not found:
            raise ValidationError(
                f"routing rule for class {rule.class_index} names a replica type "
                "the plan does not have"
            )
        rules[rule.class_index].append((found[0], rule.weight))
    return routing.class_weighted(types, [sorted(r) for r in rules], labels)


def _assumptions(
    plan: PlanResult,
    options: SimOptions,
    run: EngineRun,
    n_truncated: int,
    gpus: Mapping[str, GPUSpec] | None,
) -> tuple[str, ...]:
    confidences = sorted(
        {r.candidate.perf.confidence for r in plan.replicas if r.candidate.perf is not None}
    )
    notes = [
        f"service times from the plan's perf estimates (confidence: {', '.join(confidences)})",
        "time per output token is the estimate's full-batch tpot_ms_p50, constant: its "
        "dependence on batch size is not modeled (conservative)",
        KV_NOTES[options.kv_accounting],
        f"routing: {options.routing}; each replica serves its queue in FIFO order",
        f"windows of {options.window_s:g} s from the first arrival until the last completion; "
        "latency p95s and violations are over requests completed in each window",
        "capacity_rps is the plan's derated capacity, constant across windows",
    ]
    if options.routing == "class_weighted":
        notes.append(
            "class_weighted: each request is classified by the plan's class bounds and sent to "
            "a replica type by the plan's routing weights (largest cumulative deficit first, "
            "deterministic), then to that type's least-outstanding replica"
            if plan.routing
            else "class_weighted on a plan without classes routes least-outstanding"
        )
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
    unknown = sorted({r.candidate.price_row.gpu_id for r in plan.replicas} - set(gpus or {}))
    if unknown:
        notes.append(
            f"total VRAM unknown for {', '.join(unknown)} (no GPU spec given): "
            "vram_bytes_total is null and free VRAM is not shown"
        )
    return tuple(notes)


__all__ = [
    "ClassSummary",
    "ReplicaWindowRecord",
    "RequestLog",
    "SimOptions",
    "SimulationSummary",
    "Timeline",
    "WindowRecord",
    "replay",
    "replay_requests",
]
