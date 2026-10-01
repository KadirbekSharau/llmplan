"""Planner outputs (M4_DESIGN.md sections 3 and 7) and the text that explains them.

`PlanResult` is the fleet, the replica configurations, the cost against the best
homogeneous fleet, which demand constraint was binding, and every candidate with the reason
it was used or rejected.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, get_args

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import PriceRow
from llmplan.errors import ValidationError
from llmplan.memory.fit import FitResult
from llmplan.perf.confidence import PlanConfidence, combined_confidence
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import PerfEstimate
from llmplan.planner.request import SLO
from llmplan.workload.classes import DemandClass

Status = Literal["eligible", "no_fit", "slo_ttft", "slo_tpot", "no_perf", "tp_gt_gpus"]
Binding = Literal["requests", "tokens", "both", "none"]
MAX_PLAN_JSON_BYTES = 50_000_000


class CandidateEval(BaseModel):
    """One (price row, replica config) pair and whether it may be used.

    `fit` is None when the candidate was rejected before the memory check (tensor parallel
    larger than, or not dividing, the instance's GPUs or the model's heads). `perf` is set
    once an estimate exists. `usd_per_hour_per_rps` is the instance price over its derated
    request capacity (`replicas_per_instance * perf.requests_per_s_capacity *
    utilization_target`); None unless eligible. `reason` is one human-readable line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_row: PriceRow
    config: ReplicaConfig
    replicas_per_instance: int = Field(ge=0)
    fit: FitResult | None
    perf: PerfEstimate | None
    status: Status
    reason: str
    usd_per_hour_per_rps: float | None


class ReplicaPlan(BaseModel):
    """`count` replicas of one candidate. `instances` is the number of its price row's
    instances those replicas occupy on their own (`ceil(count * tp / gpu_count)`); when
    several candidates share a row, `FleetItem.instances` is the row's authoritative total.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateEval
    count: int = Field(gt=0)
    instances: int = Field(gt=0)


class FleetItem(BaseModel):
    """`instances` of one price row, costing `24 * price_usd_per_hour * instances` per day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_row: PriceRow
    instances: int = Field(gt=0)
    usd_per_day: float = Field(gt=0)


class SolverInfo(BaseModel):
    """How the fleet was found. `objective_usd_per_day` is recomputed from the chosen integers
    with float prices, so every backend reports the same cost. `solve_time_s` is wall-clock
    and therefore excluded from serialization (JSON output stays byte-identical).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    status: Literal["optimal", "feasible_time_limit"]
    objective_usd_per_day: float
    best_bound_usd_per_day: float | None
    solve_time_s: float = Field(ge=0, exclude=True)
    n_variables: int = Field(ge=0)
    n_constraints: int = Field(ge=0)


class RoutingRule(BaseModel):
    """M7: the share `weight` of class `class_index`'s requests to send to the replicas of
    `candidate`, and what the routing LP gives that class there: `replicas` replica-
    equivalents (`x_{r,k}`), worth `capacity_rps` req/s and `capacity_output_tokens_per_s`
    output tokens/s (derated). A class's weights sum to 1; a weight is that replica type's
    share of the class's allocated request capacity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_index: int = Field(ge=0)
    candidate: CandidateEval
    weight: float = Field(gt=0, le=1)
    replicas: float = Field(ge=0)
    capacity_rps: float = Field(ge=0)
    capacity_output_tokens_per_s: float = Field(ge=0)


class PlanResult(BaseModel):
    """The cheapest fleet found for a `PlanRequest` and its explanation.

    Demand is the peak window (`stats.peak_window_rps`, `stats.peak_output_tokens_per_s`);
    capacity is derated by the SLO's `utilization_target` and summed over replicas.
    `binding` names the demand constraint(s) that are tight in the LP relaxation (see
    M4_NOTES.md). `baseline` is the best homogeneous fleet (None when the request was
    homogeneous or no single row can meet demand), and `baseline_saving_pct` is
    `(baseline - cost) / baseline * 100`. `candidates` holds every candidate, eligible first
    in order of `usd_per_hour_per_rps`, then the rejected ones. M7: `classes` are the
    request's demand classes, `routing` the per-class routing weights over the fleet, and
    `class_binding` each class's binding label (all empty without classes; then `binding`
    is M4's, else it says whether any class's request or token demand is tight).
    M8: the properties `perf_confidence` and `perf_sources` say how much the chosen
    replicas' performance estimates can be trusted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fleet: tuple[FleetItem, ...]
    replicas: tuple[ReplicaPlan, ...]
    cost_usd_per_day: float
    baseline: PlanResult | None
    baseline_saving_pct: float | None
    demand_rps: float
    demand_output_tokens_per_s: float
    capacity_rps: float
    capacity_output_tokens_per_s: float
    binding: Binding
    candidates: tuple[CandidateEval, ...]
    solver: SolverInfo
    assumptions: tuple[str, ...]
    classes: tuple[DemandClass, ...] = ()
    routing: tuple[RoutingRule, ...] = ()
    class_binding: tuple[Binding, ...] = ()

    @property
    def perf_confidence(self) -> PlanConfidence:
        """M8: the chosen replicas' `PerfEstimate.confidence`, or `"mixed"` when they differ.

        A property, not a field: it is derived from `replicas`, so serialized plans (and the
        pre-M8 JSON outputs) are unchanged."""
        return combined_confidence(
            r.candidate.perf.confidence for r in self.replicas if r.candidate.perf is not None
        )

    @property
    def perf_sources(self) -> tuple[str, ...]:
        """M8: the distinct benchmark `source_urls` behind the chosen replicas, sorted."""
        return tuple(
            sorted(
                {
                    url
                    for r in self.replicas
                    if r.candidate.perf is not None
                    for url in r.candidate.perf.source_urls
                }
            )
        )


def binding_label(requests_tight: bool, tokens_tight: bool) -> Binding:
    """Combine the two tightness flags into the `PlanResult.binding` label."""
    if requests_tight and tokens_tight:
        return "both"
    if requests_tight:
        return "requests"
    return "tokens" if tokens_tight else "none"


def status_counts(candidates: Sequence[CandidateEval]) -> str:
    """`"40 no_fit, 56 slo_ttft"`: nonzero rejection counts in `Status` order."""
    return count_statuses(c.status for c in candidates)


def count_statuses(statuses: Iterable[Status]) -> str:
    """`"40 no_fit, 56 slo_ttft"` for any statuses (M7: per-class verdicts)."""
    counts = Counter(statuses)
    parts = [f"{counts[s]} {s}" for s in get_args(Status) if s != "eligible" and counts[s]]
    return ", ".join(parts) or "none rejected"


def infeasible_reason(candidates: Sequence[CandidateEval]) -> str:
    """Why no fleet exists when no candidate is eligible, e.g.
    `"0 of 96 candidates eligible: 40 no_fit, 56 slo_ttft"`."""
    eligible = sum(c.status == "eligible" for c in candidates)
    return f"{eligible} of {len(candidates)} candidates eligible: {status_counts(candidates)}"


def label(candidate: CandidateEval) -> str:
    """Short name of a candidate: `"aws p5.48xlarge tp2 fp8 seqs128"`."""
    row, config = candidate.price_row, candidate.config
    return (
        f"{row.provider} {row.instance} tp{config.tensor_parallel} {config.dtype} "
        f"seqs{config.max_num_seqs}"
    )


def load_plan_json(
    path: Path, *, max_bytes: int = MAX_PLAN_JSON_BYTES
) -> tuple[PlanResult, SLO | None]:
    """Read the output of `llmplan plan --format json` (or a bare `PlanResult` JSON dump).

    Returns the `PlanResult` and the `SLO` recorded under `request.slo`, None when the file
    has no `request` object; an M8 `class_comparison` object is ignored.
    `solver.solve_time_s` is not serialized, so a loaded plan reports 0.0 (also for its
    baseline). Raises `ValidationError` naming the file for an
    unreadable, oversized (`max_bytes`, default 50 MB), non-JSON, or non-`PlanResult` file.
    """
    try:
        if path.stat().st_size > max_bytes:
            raise ValidationError(f"plan file {path} is larger than {max_bytes} bytes")
        doc: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"plan file {path} is not readable: {exc.strerror}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(f"plan file {path} is not valid JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise ValidationError(f"plan file {path} does not hold a JSON object")
    request = doc.pop("request", None)
    doc.pop("class_comparison", None)  # M8: written next to a class plan; not part of it
    recorded = request.get("slo") if isinstance(request, dict) else None
    try:
        result = PlanResult.model_validate(_with_solve_time(doc))
        slo = None if recorded is None else SLO.model_validate(recorded)
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "input"
        raise ValidationError(
            f"plan file {path} is not a plan result: {field}: {err['msg']}"
        ) from None
    return result, slo


def _with_solve_time(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    if isinstance(out.get("solver"), dict):
        out["solver"] = {"solve_time_s": 0.0, **out["solver"]}
    if isinstance(out.get("baseline"), dict):
        out["baseline"] = _with_solve_time(out["baseline"])
    return out
