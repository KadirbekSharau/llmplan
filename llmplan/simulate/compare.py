"""A class plan against the same request sized for the mean request (M8_DESIGN.md section 4).

With request-size classes (M7) every replica is sized per class. The comparison plans the
same request without classes and, given the trace, replays that single-class fleet on it
(M7_DESIGN.md section 6.3), so the results can say what sizing for the mean request would
cost and whether it would miss the latency target, instead of a negative "saving".
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.errors import InfeasiblePlan
from llmplan.perf.estimate import PerfBackend
from llmplan.planner import PlanRequest, PlanResult, plan
from llmplan.simulate import replay
from llmplan.simulate.timeline import SimOptions
from llmplan.workload import Workload


class ClassComparison(BaseModel):
    """The class plan's cost beside the single-class plan of the same request.

    `single_class_cost_usd_per_day` is None when no fleet sized for the mean request meets
    the target. `single_class_ttft_violation_pct` is the share of requests over the TTFT
    budget when the single-class fleet is replayed on the trace (least-outstanding routing,
    the request's SLO budgets); None when it was not replayed (no trace, or no such fleet).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_cost_usd_per_day: float = Field(gt=0)
    single_class_cost_usd_per_day: float | None = Field(gt=0)
    single_class_ttft_violation_pct: float | None = Field(ge=0, le=100)


def compare_single_class(
    request: PlanRequest,
    result: PlanResult,
    workload: Workload | None = None,
    *,
    options: SimOptions | None = None,
    gpus: Mapping[str, GPUSpec] | None = None,
    backends: Mapping[str, PerfBackend] | None = None,
) -> ClassComparison | None:
    """Plan `request` without its classes and, with `workload`, replay that fleet on it.

    `result` is the plan of `request` with classes; None is returned when the request has
    none (nothing to compare). `options` are the replay options (routing is forced to
    `least_outstanding`: the single-class plan has no routing weights); `gpus` is the
    catalog passed to `replay`; `backends` overrides perf backends as in `plan` (uploaded
    benchmark rows). Raises what `plan` and `replay` raise, except that an infeasible
    single-class plan is a result (cost None).
    """
    if not request.classes:
        return None
    try:
        single = plan(request.model_copy(update={"classes": ()}), backends=backends)
    except InfeasiblePlan:
        return ClassComparison(
            class_cost_usd_per_day=result.cost_usd_per_day,
            single_class_cost_usd_per_day=None,
            single_class_ttft_violation_pct=None,
        )
    violations = None
    if workload is not None:
        chosen = (options or SimOptions()).model_copy(update={"routing": "least_outstanding"})
        timeline = replay(single, workload, slo=request.slo, options=chosen, gpus=gpus)
        violations = timeline.summary.ttft_violation_pct
    return ClassComparison(
        class_cost_usd_per_day=result.cost_usd_per_day,
        single_class_cost_usd_per_day=single.cost_usd_per_day,
        single_class_ttft_violation_pct=violations,
    )
