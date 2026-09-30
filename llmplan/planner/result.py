"""Planner outputs (M4_DESIGN.md sections 3 and 7) and the text that explains them.

`PlanResult` is the fleet, the replica configurations, the cost against the best
homogeneous fleet, which demand constraint was binding, and every candidate with the reason
it was used or rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import PriceRow
from llmplan.memory.fit import FitResult
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import PerfEstimate

Status = Literal["eligible", "no_fit", "slo_ttft", "slo_tpot", "no_perf", "tp_gt_gpus"]
Binding = Literal["requests", "tokens", "both", "none"]


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


class PlanResult(BaseModel):
    """The cheapest fleet found for a `PlanRequest` and its explanation.

    Demand is the peak window (`stats.peak_window_rps`, `stats.peak_output_tokens_per_s`);
    capacity is derated by the SLO's `utilization_target` and summed over replicas.
    `binding` names the demand constraint(s) that are tight in the LP relaxation (see
    M4_NOTES.md). `baseline` is the best homogeneous fleet (None when the request was
    homogeneous or no single row can meet demand), and `baseline_saving_pct` is
    `(baseline - cost) / baseline * 100`. `candidates` holds every candidate, eligible first
    in order of `usd_per_hour_per_rps`, then the rejected ones.
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


def binding_label(requests_tight: bool, tokens_tight: bool) -> Binding:
    """Combine the two tightness flags into the `PlanResult.binding` label."""
    if requests_tight and tokens_tight:
        return "both"
    if requests_tight:
        return "requests"
    return "tokens" if tokens_tight else "none"


def status_counts(candidates: Sequence[CandidateEval]) -> str:
    """`"40 no_fit, 56 slo_ttft"`: nonzero rejection counts in `Status` order."""
    counts = Counter(c.status for c in candidates)
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
