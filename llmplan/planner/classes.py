"""Request-size classes in the planner (M7_DESIGN.md section 4).

Each candidate that passed the tensor-parallel and memory checks is estimated once per
class with the class's token statistics (the M3 `estimate()` takes any `StatsLike`; a
`DemandClass` is one), cached by (GPU, replica config, class) because several price rows
share a GPU. A candidate is eligible for the classes whose SLO it meets and has capacity 0
for the others; it stays a candidate as long as it is eligible for at least one class.
After the solve, the routing LP's allocation over the chosen fleet becomes per-class
routing weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llmplan.catalog.hardware import GPUSpec
from llmplan.errors import PerfError
from llmplan.perf import PerfEstimate, ReplicaConfig, estimate
from llmplan.planner.candidates import Column, slo_status
from llmplan.planner.request import PlanRequest
from llmplan.planner.result import CandidateEval, RoutingRule, Status, count_statuses
from llmplan.planner.solve import Allocation
from llmplan.workload.classes import DemandClass


@dataclass(frozen=True)
class ClassEval:
    """One candidate on one class: the class-shape estimate (None for `no_perf`), the
    status (`eligible`, `no_perf`, `slo_ttft`, `slo_tpot`), and a one-line reason."""

    perf: PerfEstimate | None
    status: Status
    reason: str


def demands(request: PlanRequest) -> tuple[tuple[float, float], ...]:
    """(req/s, output tokens/s) to meet per class; one pair (the peak windows of
    `request.stats`) without classes."""
    if request.classes:
        return tuple((c.peak_rps, c.peak_output_tokens_per_s) for c in request.classes)
    return ((request.stats.peak_window_rps, request.stats.peak_output_tokens_per_s),)


def _evaluate(
    request: PlanRequest, gpu: GPUSpec, config: ReplicaConfig, demand: DemandClass
) -> ClassEval:
    try:
        perf = estimate(request.model, gpu, config, demand, backend=request.options.perf_backend)
    except PerfError as exc:
        return ClassEval(None, "no_perf", str(exc))
    rejected = slo_status(perf, request.slo)
    if rejected is not None:
        return ClassEval(perf, *rejected)
    return ClassEval(perf, "eligible", "eligible")


def evaluate_classes(
    request: PlanRequest, candidates: Sequence[CandidateEval], gpus: Mapping[str, GPUSpec]
) -> tuple[tuple[ClassEval, ...] | None, ...]:
    """Per candidate, its evaluation on every class of `request` (None for candidates
    rejected before the performance step: tensor parallelism or memory)."""
    cache: dict[tuple[str, ReplicaConfig, int], ClassEval] = {}
    out: list[tuple[ClassEval, ...] | None] = []
    for c in candidates:
        if c.fit is None or not c.fit.fits:
            out.append(None)
            continue
        gpu = gpus[c.price_row.gpu_id]
        evals = []
        for demand in request.classes:
            key = (gpu.id, c.config, demand.index)
            if key not in cache:
                cache[key] = _evaluate(request, gpu, c.config, demand)
            evals.append(cache[key])
        out.append(tuple(evals))
    return tuple(out)


def _which(indices: Sequence[int]) -> str:
    return f"class{'es' if len(indices) > 1 else ''} {', '.join(map(str, indices))}"


def reconcile(
    candidate: CandidateEval, evals: Sequence[ClassEval] | None, utilization: float
) -> CandidateEval:
    """The candidate's status once classes are known. Eligible for some class: `eligible`
    (its reason names the classes it misses, or, when the whole workload failed, the
    classes it serves). Eligible for none: the first class's rejection. Unchanged when the
    classes agree with the whole-workload verdict or `evals` is None."""
    if not evals:
        return candidate
    served = [(k, e.perf) for k, e in enumerate(evals) if e.status == "eligible" and e.perf]
    missed = [(k, e) for k, e in enumerate(evals) if e.status != "eligible"]
    if candidate.status == "eligible":
        if not missed:
            return candidate
        if not served:
            reasons = "; ".join(f"class {k}: {e.reason}" for k, e in missed)
            return candidate.model_copy(
                update={
                    "status": missed[0][1].status,
                    "reason": f"not eligible for any request-size class ({reasons})",
                    "usd_per_hour_per_rps": None,
                }
            )
        reason = (
            f"{candidate.reason}; not eligible for {_which([k for k, _ in missed])} "
            f"({missed[0][1].reason})"
        )
        return candidate.model_copy(update={"reason": reason})
    if not served:
        return candidate
    perf = candidate.perf or served[0][1]
    capacity = candidate.replicas_per_instance * perf.requests_per_s_capacity * utilization
    return candidate.model_copy(
        update={
            "status": "eligible",
            "perf": perf,
            "reason": (
                f"eligible for {_which([k for k, _ in served])} only (whole workload: "
                f"{candidate.reason})"
            ),
            "usd_per_hour_per_rps": candidate.price_row.price_usd_per_hour / capacity,
        }
    )


def unserved(
    request: PlanRequest,
    class_evals: Sequence[tuple[ClassEval, ...] | None],
    candidates: Sequence[CandidateEval],
) -> str | None:
    """Why some class with demand has no eligible candidate (e.g. "class 1 (inputs
    1051..2000, outputs 0..65): 0 of 4 candidates eligible: 4 slo_ttft"), else None."""
    for c in request.classes:
        if c.peak_rps <= 0 and c.peak_output_tokens_per_s <= 0:
            continue
        statuses = [
            evals[c.index].status if evals is not None else cand.status
            for evals, cand in zip(class_evals, candidates, strict=True)
        ]
        if "eligible" not in statuses:
            return (
                f"class {c.index} (inputs {c.input_lo}..{c.input_hi}, outputs "
                f"{c.output_lo}..{c.output_hi}): 0 of {len(candidates)} candidates eligible: "
                f"{count_statuses(statuses)}"
            )
    return None


def routing(
    counts: Sequence[tuple[Column, int]], allocation: Allocation | None, n_classes: int
) -> tuple[RoutingRule, ...]:
    """Routing rules over a fleet (`counts`: chosen columns and replica counts).

    The weight of replica type r for class k is its share of the class's allocated request
    capacity, `x_{r,k} cap_{r,k} / sum_r x_{r,k} cap_{r,k}` (with one class, `x_r = m_r`).
    A class the allocation gives nothing (it has no peak demand) is spread over the
    replicas eligible for it in proportion to `m_r cap_{r,k}`; a class no chosen replica
    can serve gets no rules. Rules are ordered by class, then fleet order.
    """
    x = allocation.x if allocation is not None else tuple((float(k),) for _, k in counts)
    rules: list[RoutingRule] = []
    for k in range(n_classes):
        shares = [(col, xs[k], col.rates()[k]) for (col, _), xs in zip(counts, x, strict=True)]
        total = sum(xk * rps for _, xk, (rps, _) in shares)
        if total <= 0:
            shares = [(col, 0.0, col.rates()[k]) for col, _ in counts]
            weights = [m * col.rates()[k][0] for col, m in counts]
        else:
            weights = [xk * rps for _, xk, (rps, _) in shares]
        spread = sum(weights)
        rules.extend(
            RoutingRule(
                class_index=k,
                candidate=col.candidate,
                weight=w / spread,
                replicas=xk,
                capacity_rps=xk * rps,
                capacity_output_tokens_per_s=xk * tps,
            )
            for (col, xk, (rps, tps)), w in zip(shares, weights, strict=True)
            if w > 0
        )
    return tuple(rules)
