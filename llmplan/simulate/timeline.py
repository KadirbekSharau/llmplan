"""Replay outputs (M5_DESIGN.md sections 3 and 7): `Timeline` and its window aggregation.

Windows are half-open, `[start + k * window_s, start + (k + 1) * window_s)` from the first
arrival as in M2, and extend until the last completion so no completion or busy time is
dropped. Rates and time-weighted means divide by the full `window_s`, so a partial last
window reads low, as in M2. Percentiles use `numpy.percentile` ("linear").
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from llmplan.simulate.events import EngineRun, StepLog
from llmplan.simulate.replica import ReplicaSpec

FloatArray = npt.NDArray[np.float64]


class SimOptions(BaseModel):
    """Replay settings. `max_requests` truncates the trace (with an assumption note); `seed`
    is reserved for tie-breaking (both policies break ties deterministically, so it does
    not change results today). Budgets default to the SLO's p95 targets when one is given
    to `replay`; with no budget, violations of that kind are not counted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    window_s: float = Field(default=60.0, gt=0)
    routing: Literal["least_outstanding", "round_robin"] = "least_outstanding"
    max_requests: int = Field(default=500_000, gt=0)
    seed: int = Field(default=0, ge=0)
    ttft_budget_ms: float | None = Field(default=None, gt=0)
    tpot_budget_ms: float | None = Field(default=None, gt=0)


class ReplicaWindowRecord(BaseModel):
    """One replica in one window. `utilization` is busy slot-seconds over
    `slots * window_s`; KV and queue means are time-weighted over the window, maxima are
    over the states the replica passed through. Bytes are summed over the replica's GPUs;
    `vram_bytes_total` is the GPU's `vram_bytes` times tensor parallel (None when `replay`
    was not given the GPU spec), so free VRAM is total minus weights minus KV in use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    replica_index: int = Field(ge=0)
    utilization: float = Field(ge=0, le=1)
    kv_tokens_in_use_mean: int = Field(ge=0)
    kv_tokens_in_use_max: int = Field(ge=0)
    kv_bytes_in_use_max: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    queue_depth_mean: float = Field(ge=0)
    queue_depth_max: int = Field(ge=0)
    requests_started: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    vram_bytes_total: int | None = Field(ge=0)


class WindowRecord(BaseModel):
    """One window: arrivals and completions, demand (`arrivals / window_s`) against the
    plan's constant derated capacity, latency p95s and SLO violations over the requests
    completed in the window (None when none completed), and every replica's record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    start_s: float
    arrivals: int = Field(ge=0)
    completions: int = Field(ge=0)
    demand_rps: float = Field(ge=0)
    capacity_rps: float = Field(ge=0)
    ttft_ms_p95: float | None
    e2e_ms_p95: float | None
    ttft_violations: int = Field(ge=0)
    tpot_violations: int = Field(ge=0)
    replicas: tuple[ReplicaWindowRecord, ...]


class SimulationSummary(BaseModel):
    """Whole-replay figures over every simulated request (`n_truncated` were cut by
    `max_requests`). Violation percentages are of simulated requests; `mean_utilization`
    is the mean over replicas and windows; `max_queue_depth` is over replicas and time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_requests: int = Field(ge=1)
    n_truncated: int = Field(ge=0)
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_p95: float
    e2e_ms_p95: float
    ttft_violation_pct: float = Field(ge=0, le=100)
    tpot_violation_pct: float = Field(ge=0, le=100)
    mean_utilization: float = Field(ge=0, le=1)
    max_queue_depth: int = Field(ge=0)


class Timeline(BaseModel):
    """What the planned fleet does over the trace: per-window records, a summary, the
    options used (budgets resolved), and the modeling assumptions. Contains no wall-clock
    values, so its JSON is byte-identical for identical inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_cost_usd_per_day: float
    options: SimOptions
    windows: tuple[WindowRecord, ...]
    summary: SimulationSummary
    assumptions: tuple[str, ...]


def build_timeline(
    run: EngineRun,
    replicas: Sequence[ReplicaSpec],
    *,
    capacity_rps: float,
    plan_cost_usd_per_day: float,
    options: SimOptions,
    n_truncated: int,
    assumptions: tuple[str, ...],
) -> Timeline:
    """Aggregate an engine run into windows and a summary (M5_DESIGN.md section 7)."""
    frame = run.requests.frame
    arrival = frame["arrival_s"].to_numpy()
    complete = frame["complete_s"].to_numpy()
    ttft = frame["ttft_ms"].to_numpy()
    tpot = frame["tpot_ms"].to_numpy()
    e2e = frame["e2e_ms"].to_numpy()
    placed = frame["replica_index"].to_numpy()
    w = options.window_s
    origin = float(arrival[0])
    n = math.floor((max(float(arrival[-1]), float(complete.max())) - origin) / w) + 1

    def window_of(times: FloatArray) -> npt.NDArray[np.int64]:
        return np.minimum(np.floor((times - origin) / w).astype(np.int64), n - 1)

    done_in = window_of(complete)
    ttft_bad = _violations(ttft, options.ttft_budget_ms)
    tpot_bad = _violations(tpot, options.tpot_budget_ms)
    arrivals = np.bincount(window_of(arrival), minlength=n)
    completions = np.bincount(done_in, minlength=n)
    ttft_violations = np.bincount(done_in, weights=ttft_bad, minlength=n)
    tpot_violations = np.bincount(done_in, weights=tpot_bad, minlength=n)
    ttft_p95 = _p95_by_window(ttft, done_in, completions)
    e2e_p95 = _p95_by_window(e2e, done_in, completions)
    started = _per_replica(placed, window_of(frame["start_s"].to_numpy()), len(replicas), n)
    finished = _per_replica(placed, done_in, len(replicas), n)
    bounds = origin + w * np.arange(n + 1, dtype=np.float64)
    columns = [
        _replica_windows(r, spec, step, bounds, started[r], finished[r])
        for r, (spec, step) in enumerate(zip(replicas, run.steps, strict=True))
    ]
    windows = tuple(
        WindowRecord(
            index=k,
            start_s=float(bounds[k]),
            arrivals=int(arrivals[k]),
            completions=int(completions[k]),
            demand_rps=float(arrivals[k]) / w,
            capacity_rps=capacity_rps,
            ttft_ms_p95=ttft_p95[k],
            e2e_ms_p95=e2e_p95[k],
            ttft_violations=round(float(ttft_violations[k])),
            tpot_violations=round(float(tpot_violations[k])),
            replicas=tuple(column[k] for column in columns),
        )
        for k in range(n)
    )
    ttft_p50, ttft_p95_all = (float(v) for v in np.percentile(ttft, [50, 95]))
    summary = SimulationSummary(
        n_requests=len(frame),
        n_truncated=n_truncated,
        ttft_ms_p50=ttft_p50,
        ttft_ms_p95=ttft_p95_all,
        tpot_ms_p95=float(np.percentile(tpot, 95)),
        e2e_ms_p95=float(np.percentile(e2e, 95)),
        ttft_violation_pct=float(ttft_bad.mean()) * 100,
        tpot_violation_pct=float(tpot_bad.mean()) * 100,
        mean_utilization=float(np.mean([r.utilization for c in columns for r in c])),
        max_queue_depth=max(r.queue_depth_max for c in columns for r in c),
    )
    return Timeline(
        plan_cost_usd_per_day=plan_cost_usd_per_day,
        options=options,
        windows=windows,
        summary=summary,
        assumptions=assumptions,
    )


def _violations(values_ms: FloatArray, budget_ms: float | None) -> FloatArray:
    if budget_ms is None:
        return np.zeros(len(values_ms))
    return (values_ms > budget_ms).astype(np.float64)


def _p95_by_window(
    values: FloatArray, window: npt.NDArray[np.int64], counts: npt.NDArray[np.int64]
) -> list[float | None]:
    ordered = values[np.argsort(window, kind="stable")]
    ends = np.cumsum(counts)
    return [
        float(np.percentile(ordered[end - count : end], 95)) if count else None
        for count, end in zip(counts.tolist(), ends.tolist(), strict=True)
    ]


def _per_replica(
    placed: npt.NDArray[np.int64], window: npt.NDArray[np.int64], n_replicas: int, n: int
) -> npt.NDArray[np.int64]:
    flat = np.bincount(placed * n + window, minlength=n_replicas * n)
    return flat.reshape(n_replicas, n)


def _replica_windows(
    index: int,
    spec: ReplicaSpec,
    step: StepLog,
    bounds: FloatArray,
    started: npt.NDArray[np.int64],
    finished: npt.NDArray[np.int64],
) -> list[ReplicaWindowRecord]:
    w = float(bounds[1] - bounds[0])
    busy, _ = _step_windows(step.time_s, step.busy_slots, bounds)
    kv_sum, kv_max = _step_windows(step.time_s, step.kv_tokens, bounds)
    queue_sum, queue_max = _step_windows(step.time_s, step.queue_depth, bounds)
    return [
        ReplicaWindowRecord(
            replica_index=index,
            utilization=min(1.0, float(busy[k]) / (spec.slots * w)),
            kv_tokens_in_use_mean=round(float(kv_sum[k]) / w),
            kv_tokens_in_use_max=int(kv_max[k]),
            kv_bytes_in_use_max=int(kv_max[k]) * spec.kv_bytes_per_token,
            weight_bytes=spec.weight_bytes,
            queue_depth_mean=float(queue_sum[k]) / w,
            queue_depth_max=int(queue_max[k]),
            requests_started=int(started[k]),
            requests_completed=int(finished[k]),
            vram_bytes_total=spec.vram_bytes_total,
        )
        for k in range(len(bounds) - 1)
    ]


def _step_windows(
    time_s: FloatArray, value: FloatArray, bounds: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Integral and maximum per window of a step function that is 0 before `time_s[0]` and
    takes `value[i]` from `time_s[i]` on. Window `k` is `[bounds[k], bounds[k + 1])`; states
    that last zero time (several events at one instant) still count toward the maximum."""
    n = len(bounds) - 1
    time_s = np.concatenate([bounds[:1], time_s])  # explicit idle state at the origin
    value = np.concatenate([[0.0], value])
    at_bounds = value[np.searchsorted(time_s, bounds, side="right") - 1]
    times = np.concatenate([time_s, bounds])
    values = np.concatenate([value, at_bounds])
    order = np.argsort(times, kind="stable")  # at equal times, bounds follow the events
    times, values = times[order], values[order]
    window = np.searchsorted(bounds, times, side="right") - 1
    inside = window < n
    integral = np.bincount(
        window[:-1][inside[:-1]],
        weights=(np.diff(times) * values[:-1])[inside[:-1]],
        minlength=n,
    ).astype(np.float64)
    maximum = np.zeros(n)
    np.maximum.at(maximum, window[inside], values[inside])
    return integral, maximum
