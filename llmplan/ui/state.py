"""Pure helpers behind the web UI (M6 sections 4, 6): inputs to a `PlanRequest`, price-table
validation, the cache key, the plan-and-replay run, timeline windows, the plan brake; M9:
number formats and error hints. Every planning decision is a call into `llmplan.*`.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec
from llmplan.errors import (
    FetchError,
    InfeasiblePlan,
    LLMPlanError,
    SolverError,
    ValidationError,
    WorkloadFormatError,
)
from llmplan.memory.engine import EngineProfile
from llmplan.perf import PerfBackend
from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.planner import SLO, PlanOptions, PlanRequest, PlanResult, plan
from llmplan.simulate import SimOptions, Timeline, replay
from llmplan.simulate.compare import ClassComparison, compare_single_class
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
    """The outcome of one Plan click: the request, the plan, and its replay. With classes
    (M7) also the cost of the request planned without them (None: infeasible or no classes)
    and (M8) the share of requests over the TTFT budget when that fleet is replayed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: PlanRequest
    result: PlanResult
    timeline: Timeline
    single_class_cost_usd_per_day: float | None = None
    single_class_ttft_violation_pct: float | None = None

    @property
    def comparison(self) -> ClassComparison | None:
        """M8: the class plan against the single-class plan (None without classes)."""
        if not self.result.classes:
            return None
        return ClassComparison(
            class_cost_usd_per_day=self.result.cost_usd_per_day,
            single_class_cost_usd_per_day=self.single_class_cost_usd_per_day,
            single_class_ttft_violation_pct=self.single_class_ttft_violation_pct,
        )

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
    """Validate an edited price table through `PriceRow` and the GPU catalog's ids; rows
    left blank are skipped. Raises `ValidationError` naming the row index and field of the
    first invalid row, an unknown `gpu_id`, or an empty table."""
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
    """Collect the inputs into a `PlanRequest` (vLLM's default engine profile), the solver
    time limit capped at 30 s (section 6); `classes` (M7) may be empty. Raises
    `ValidationError` naming an empty selection or the first invalid option."""
    selections = (gpu_ids, providers, tensor_parallel, dtypes, max_num_seqs)
    names = ("GPU", "provider", "tensor parallel degree", "dtype", "max_num_seqs value")
    if empty := [name for name, chosen in zip(names, selections, strict=True) if not chosen]:
        raise ValidationError(f"select at least one {empty[0]}")
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
    """The smallest round window (1, 2 or 5 x 10^k s) of at least `span_s / 150`, giving 60
    to 200 windows over `span_s`; 1 s for an empty span."""
    if span_s <= 0:
        return 1.0
    lower = span_s / 150
    exponent = math.floor(math.log10(lower))
    candidates = (
        round(step * 10.0**e, 12) for e in (exponent, exponent + 1) for step in _WINDOW_STEPS
    )
    return next(w for w in candidates if w >= lower * (1 - 1e-12))


def window_choices(window_s: float) -> tuple[float, ...]:
    """The Timeline tab's windows (M9): round values (1, 2 or 5 x 10^k s) from a fifth of
    `window_s` to five times it; `window_s` itself is one when it is round."""
    top = math.floor(math.log10(window_s)) + 1
    values = (round(s * 10.0**e, 12) for e in range(top - 2, top + 1) for s in _WINDOW_STEPS)
    return tuple(w for w in values if window_s / 5 * (1 - 1e-9) <= w <= window_s * 5 * (1 + 1e-9))


def replay_window(run: PlanRun, workload: Workload, window_s: float) -> Timeline:
    """The run's replay repeated with windows of `window_s` (the Timeline tab's selector,
    M9): same plan, trace, routing, budgets and GPU catalog; the plan is not recomputed."""
    options = run.timeline.options.model_copy(update={"window_s": window_s})
    with _RUN_LOCK:
        return replay(
            run.result, workload, slo=run.request.slo, options=options, gpus=run.request.gpus
        )


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


def cache_key(
    request: PlanRequest,
    options: SimOptions,
    workload: Workload,
    benchmarks: Sequence[BenchmarkRow] = (),
) -> str:
    """SHA-256 of the request and replay options JSON, the workload rows' digest (the replay
    reads the rows) and (M8) the uploaded benchmark rows (they change the estimates)."""
    rows = "\n".join(row.model_dump_json() for row in benchmarks)
    digest = hashlib.sha256()
    parts = (request.model_dump_json(), options.model_dump_json(), workload_digest(workload), rows)
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run_plan(
    request: PlanRequest,
    workload: Workload,
    options: SimOptions,
    gpus: Mapping[str, GPUSpec],
    backends: Mapping[str, PerfBackend] | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> PlanRun:
    """Plan, then replay the workload on the fleet with the request's SLO budgets; a plan with
    classes is replayed with `class_weighted` routing and compared with the request planned
    without classes (`compare_single_class`). Over 200 windows, the replay is repeated with
    a window chosen over its full span. `backends` (M8) carries uploaded benchmark rows;
    `progress` (M9) gets the seconds of the "estimate" (plan wall time less the solver's),
    "solve" and "replay" stages. One run at a time per process; raises what `plan` and
    `replay` raise."""
    report = progress or (lambda stage, seconds: None)
    with _RUN_LOCK:
        started = time.perf_counter()
        result = plan(request, backends=backends)
        planned = time.perf_counter()
        report("estimate", planned - started - result.solver.solve_time_s)
        report("solve", result.solver.solve_time_s)
        comparison = compare_single_class(
            request, result, workload, options=options, gpus=gpus, backends=backends
        )
        if request.classes:
            options = options.model_copy(update={"routing": "class_weighted"})
        timeline = replay(result, workload, slo=request.slo, options=options, gpus=gpus)
        if len(timeline.windows) > MAX_WINDOWS:
            span = len(timeline.windows) * options.window_s
            wider = options.model_copy(update={"window_s": timeline_window_s(span)})
            timeline = replay(result, workload, slo=request.slo, options=wider, gpus=gpus)
        report("replay", time.perf_counter() - planned)
    single = {} if comparison is None else comparison.model_dump(exclude={"class_cost_usd_per_day"})
    return PlanRun(request=request, result=result, timeline=timeline, **single)


def admit_plan(
    times: Sequence[float], now: float, *, limit: int = MAX_PLANS_PER_HOUR
) -> tuple[bool, tuple[float, ...]]:
    """The per-session brake: whether one more plan may run at `now` given this session's
    plan times; returns the times of the last hour (plus `now` when admitted)."""
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


HINTS: tuple[tuple[type[LLMPlanError], str], ...] = (  # first match wins
    (InfeasiblePlan, "relax the latency target (TTFT, TPOT, utilization) or add GPUs."),
    (FetchError, "check the Hugging Face id (org/name), or set HF_TOKEN for a gated model."),
    (WorkloadFormatError, "use the columns arrival_s or timestamp, input_tokens, output_tokens."),
    (SolverError, "raise the solver time limit under Advanced, or select fewer options."),
)


def error_text(error: LLMPlanError) -> str:
    """The library's message, then a "What to change" hint chosen by the error's type."""
    hint = next((text for kind, text in HINTS if isinstance(error, kind)), None)
    return str(error) if hint is None else f"{error}\n\nWhat to change: {hint}"


def usd(value: float | None) -> str:
    """Dollars with thousands separators and cents (`$1,234.50`); `n/a` for None."""
    return "n/a" if value is None else f"${value:,.2f}"


def duration(ms: float) -> str:
    """A duration in ms below one second (3 significant digits), else in seconds."""
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1000:,.2f} s"
