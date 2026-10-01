"""Pure helpers behind the web UI (M6_DESIGN.md sections 4 and 6).

Inputs to a `PlanRequest` and `SimOptions`, price-table validation through `PriceRow`, the
cache key, the plan-and-replay run (with request-size classes, M7, also the single-class
cost it is compared with), the timeline window, and the per-session plan brake.
Every planning decision is a call into `llmplan.*`; nothing here imports Streamlit, so it
is unit-testable without a browser.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec
from llmplan.errors import InfeasiblePlan, ValidationError
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest, PlanResult, plan
from llmplan.simulate import SimOptions, Timeline, replay
from llmplan.types import DType
from llmplan.ui.presets import MAX_PLANS_PER_HOUR, MAX_SIM_REQUESTS, MAX_TIME_LIMIT_S
from llmplan.workload import DemandClass, Workload, WorkloadStats

PRICE_COLUMNS = tuple(PriceRow.model_fields)
SECONDS_PER_HOUR = 3600.0
MIN_WINDOWS, MAX_WINDOWS = 60, 200
_WINDOW_STEPS = (1.0, 2.0, 5.0)
# HiGHS keeps a process-wide thread pool (M4_NOTES.md); Streamlit serves sessions from
# several threads, so solves and replays run one at a time.
_RUN_LOCK = threading.Lock()


class PlanRun(BaseModel):
    """The outcome of one Plan click: the request, the plan, and its replay. With
    request-size classes (M7), `single_class_cost_usd_per_day` is the cost of the same
    request planned without them (None when that plan is infeasible or there are no
    classes), the base of the UI's "saving from request-size routing"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: PlanRequest
    result: PlanResult
    timeline: Timeline
    single_class_cost_usd_per_day: float | None = None

    @property
    def class_saving_pct(self) -> float | None:
        """`(single-class cost - cost) / single-class cost * 100`, or None."""
        base = self.single_class_cost_usd_per_day
        if base is None:
            return None
        return (base - self.result.cost_usd_per_day) / base * 100


def prices_frame(rows: Sequence[PriceRow]) -> pd.DataFrame:
    """The editable price table: one row per `PriceRow`, columns in model order."""
    return pd.DataFrame([row.model_dump() for row in rows], columns=list(PRICE_COLUMNS))


def _cell(value: Any) -> Any:
    """A table cell as `PriceRow` input: blanks and NaN become None, datetimes dates."""
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, datetime):  # pandas Timestamps included
        return value.date()
    return value


def price_rows(frame: pd.DataFrame, gpus: Mapping[str, GPUSpec]) -> tuple[PriceRow, ...]:
    """Validate an edited price table through `PriceRow` (and the GPU catalog's ids).

    Rows whose cells are all blank (added in the editor and left empty) are skipped.
    Raises `ValidationError` naming the row index and field of the first invalid row, an
    unknown `gpu_id`, or an empty table.
    """
    rows: list[PriceRow] = []
    for index, record in enumerate(frame.to_dict("records")):
        values = {column: _cell(record.get(column)) for column in PRICE_COLUMNS}
        if all(value is None for value in values.values()):
            continue
        try:
            row = PriceRow.model_validate(values)
        except pydantic.ValidationError as exc:
            err = exc.errors()[0]
            field = ".".join(str(p) for p in err["loc"]) or "row"
            raise ValidationError(f"price row {index}: field {field!r}: {err['msg']}") from None
        if row.gpu_id not in gpus:
            raise ValidationError(
                f"price row {index}: unknown gpu_id {row.gpu_id!r}; known: {', '.join(gpus)}"
            )
        rows.append(row)
    if not rows:
        raise ValidationError("the price table has no rows")
    return tuple(rows)


def build_request(
    *,
    model: ModelSpec,
    stats: WorkloadStats,
    slo: SLO,
    gpus: Mapping[str, GPUSpec],
    prices: tuple[PriceRow, ...],
    gpu_ids: Sequence[str],
    providers: Sequence[str],
    tensor_parallel: Sequence[int],
    dtypes: Sequence[DType],
    max_num_seqs: Sequence[int],
    max_model_len: int,
    perf_backend: str,
    solver: str,
    time_limit_s: float,
    classes: tuple[DemandClass, ...] = (),
) -> PlanRequest:
    """Collect the sidebar inputs into a `PlanRequest` (vLLM's default engine profile).

    The solver time limit is capped at 30 s whatever the caller passes (section 6).
    `classes` are the request-size classes of the traffic (M7; empty: none). Raises
    `ValidationError` naming an empty selection or the first invalid option.
    """
    for name, chosen in (
        ("GPU", gpu_ids),
        ("provider", providers),
        ("tensor parallel degree", tensor_parallel),
        ("dtype", dtypes),
        ("max_num_seqs value", max_num_seqs),
    ):
        if not chosen:
            raise ValidationError(f"select at least one {name}")
    try:
        options = PlanOptions.model_validate(
            {
                "gpu_ids": tuple(gpu_ids),
                "providers": tuple(providers),
                "tensor_parallel_choices": tuple(tensor_parallel),
                "dtype_choices": tuple(dtypes),
                "max_num_seqs_choices": tuple(max_num_seqs),
                "max_model_len": max_model_len,
                "perf_backend": perf_backend,
                "solver": solver,
                "time_limit_s": min(time_limit_s, MAX_TIME_LIMIT_S),
            }
        )
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "options"
        raise ValidationError(f"invalid {field}: {err['msg']}") from None
    return PlanRequest(
        model=model,
        stats=stats,
        slo=slo,
        engine=EngineProfile(),
        options=options,
        gpus=gpus,
        prices=prices,
        classes=classes,
    )


def timeline_window_s(span_s: float) -> float:
    """A round window (1, 2 or 5 x 10^k seconds) giving 60 to 200 windows over `span_s`.

    The smallest such value of at least `span_s / 150` is chosen; 1 s for an empty span.
    """
    if span_s <= 0:
        return 1.0
    lower = span_s / 150
    exponent = math.floor(math.log10(lower))
    candidates = (
        round(step * 10.0**e, 12) for e in (exponent, exponent + 1) for step in _WINDOW_STEPS
    )
    return next(w for w in candidates if w >= lower * (1 - 1e-12))


def sim_options(stats: WorkloadStats) -> SimOptions:
    """Replay settings: the auto window over the trace's duration and the 200,000-request
    cap of section 6. Budgets come from the plan's SLO in `run_plan`."""
    return SimOptions(window_s=timeline_window_s(stats.duration_s), max_requests=MAX_SIM_REQUESTS)


def workload_digest(workload: Workload) -> str:
    """SHA-256 over the columns a replay reads (arrivals and token counts)."""
    digest = hashlib.sha256()
    for column in ("arrival_s", "input_tokens", "output_tokens"):
        digest.update(np.ascontiguousarray(workload.frame[column].to_numpy()).tobytes())
    return digest.hexdigest()


def cache_key(request: PlanRequest, options: SimOptions, workload: Workload) -> str:
    """SHA-256 of the request and replay options JSON plus the workload rows' digest (the
    timeline depends on the rows, not only on the statistics in the request)."""
    digest = hashlib.sha256()
    for part in (request.model_dump_json(), options.model_dump_json(), workload_digest(workload)):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run_plan(
    request: PlanRequest,
    workload: Workload,
    options: SimOptions,
    gpus: Mapping[str, GPUSpec],
) -> PlanRun:
    """Plan, then replay the workload on the planned fleet with the request's SLO budgets.

    A plan with request-size classes is replayed with `class_weighted` routing and is
    compared with the same request planned without classes. When queues make the replay
    run far past the last arrival (more than 200 windows), it is replayed once more with a
    window chosen over the full span. Runs one at a time per process. Raises whatever
    `plan` and `replay` raise.
    """
    single: float | None = None
    with _RUN_LOCK:
        result = plan(request)
        if request.classes:
            options = options.model_copy(update={"routing": "class_weighted"})
            try:
                single = plan(request.model_copy(update={"classes": ()})).cost_usd_per_day
            except InfeasiblePlan:
                single = None
        timeline = replay(result, workload, slo=request.slo, options=options, gpus=gpus)
        if len(timeline.windows) > MAX_WINDOWS:
            span = len(timeline.windows) * options.window_s
            wider = options.model_copy(update={"window_s": timeline_window_s(span)})
            timeline = replay(result, workload, slo=request.slo, options=wider, gpus=gpus)
    return PlanRun(
        request=request, result=result, timeline=timeline, single_class_cost_usd_per_day=single
    )


def admit_plan(
    times: Sequence[float], now: float, *, limit: int = MAX_PLANS_PER_HOUR
) -> tuple[bool, tuple[float, ...]]:
    """The per-session brake: whether one more plan may run at `now`, given the times of
    this session's plans; returns the times of the last hour (plus `now` when admitted)."""
    recent = tuple(t for t in times if now - t < SECONDS_PER_HOUR)
    if len(recent) >= limit:
        return False, recent
    return True, (*recent, now)


def assumptions(run: PlanRun) -> dict[str, tuple[str, ...]]:
    """Every assumption string, verbatim, grouped: the plan's, the chosen replicas' perf
    estimates' (first occurrence order, duplicates dropped), and the replay's."""
    perf: dict[str, None] = {}
    for replica in run.result.replicas:
        if replica.candidate.perf is not None:
            perf.update(dict.fromkeys(replica.candidate.perf.assumptions))
    return {
        "Plan": run.result.assumptions,
        "Performance estimates of the chosen replicas": tuple(perf),
        "Simulation": run.timeline.assumptions,
    }


def as_of_range(rows: Sequence[PriceRow]) -> tuple[date, date]:
    """Oldest and newest `as_of` dates in a price table (shown next to the editor)."""
    dates = [row.as_of for row in rows]
    return min(dates), max(dates)
