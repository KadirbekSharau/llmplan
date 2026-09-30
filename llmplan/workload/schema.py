"""Workload data models (M2_DESIGN.md section 3): `Workload`, `Distribution`, `WorkloadStats`.

A `Workload` wraps a validated pandas frame of requests. The frame is checked once, at
construction; downstream code trusts its columns, dtypes, and ordering.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field

# Column name -> dtype check, in the exact order every frame carries them.
FRAME_COLUMNS: tuple[str, ...] = ("arrival_s", "input_tokens", "output_tokens", "model", "tenant")


def _is_nullable_string(dtype: object) -> bool:
    return isinstance(dtype, pd.StringDtype) and dtype.na_value is pd.NA


class Workload(BaseModel):
    """A request trace: one row per request, sorted by arrival time.

    `frame` has exactly the columns `arrival_s` (float64 seconds, >= 0, non-decreasing,
    first value 0.0), `input_tokens` (int64, >= 1), `output_tokens` (int64, >= 0), `model`
    and `tenant` (pandas nullable "string", may be all <NA>), with a default RangeIndex and
    at least one row. `source` is the file path or `synthetic:<seed>`, `format` the
    registry key that produced it, `dropped_rows` the rows removed during parsing, and
    `notes` human-readable remarks. The model is frozen but the frame is a mutable pandas
    object: callers must treat it as read-only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source: str = Field(min_length=1)
    format: str = Field(min_length=1)
    frame: pd.DataFrame
    dropped_rows: int = Field(ge=0)
    notes: tuple[str, ...] = ()

    @pydantic.model_validator(mode="after")
    def _check_frame(self) -> Self:
        frame = self.frame
        if tuple(frame.columns) != FRAME_COLUMNS:
            raise ValueError(
                f"frame columns must be {list(FRAME_COLUMNS)}, got {list(frame.columns)}"
            )
        if not isinstance(frame.index, pd.RangeIndex) or frame.index.start != 0:
            raise ValueError("frame must have a default RangeIndex starting at 0")
        if len(frame) == 0:
            raise ValueError("frame must contain at least one request")
        expected = {"arrival_s": np.float64, "input_tokens": np.int64, "output_tokens": np.int64}
        for column, dtype in expected.items():
            if frame[column].dtype != dtype:
                raise ValueError(f"column {column!r} must be {np.dtype(dtype)}")
        for column in ("model", "tenant"):
            if not _is_nullable_string(frame[column].dtype):
                raise ValueError(f"column {column!r} must have pandas 'string' dtype")
        arrival = frame["arrival_s"].to_numpy()
        if not np.all(np.isfinite(arrival)) or arrival[0] != 0.0:
            raise ValueError("arrival_s must be finite and start at 0.0")
        if np.any(np.diff(arrival) < 0):
            raise ValueError("arrival_s must be non-decreasing")
        if frame["input_tokens"].min() < 1:
            raise ValueError("input_tokens must be >= 1")
        if frame["output_tokens"].min() < 0:
            raise ValueError("output_tokens must be >= 0")
        return self


class Distribution(BaseModel):
    """A token-length distribution for the synthetic generator.

    `fixed` uses `value`; `lognormal` uses `mean` and `sigma` of the underlying normal and
    clips draws to `[lo, hi]`; `uniform` draws integers from `[lo, hi]` inclusive. Fields
    that do not belong to `kind` must be left unset. Draws are rounded to integers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed", "lognormal", "uniform"]
    value: int | None = Field(default=None, ge=0)
    mean: float | None = Field(default=None, allow_inf_nan=False)
    sigma: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    lo: int = Field(default=1, ge=0)
    hi: int = Field(default=131_072, ge=0)

    @pydantic.model_validator(mode="after")
    def _check_kind_fields(self) -> Self:
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) must be <= hi ({self.hi})")
        wanted = {"fixed": {"value"}, "lognormal": {"mean", "sigma"}, "uniform": set()}[self.kind]
        for name in ("value", "mean", "sigma"):
            is_set = getattr(self, name) is not None
            if is_set and name not in wanted:
                raise ValueError(f"{self.kind} distribution does not take {name!r}")
            if not is_set and name in wanted:
                raise ValueError(f"{self.kind} distribution requires {name!r}")
        if self.value is not None and not self.lo <= self.value <= self.hi:
            raise ValueError(f"value ({self.value}) must lie in [lo, hi] = [{self.lo}, {self.hi}]")
        return self


class WorkloadStats(BaseModel):
    """Summary of a `Workload` that later milestones size against (see `compute_stats`).

    Rates are per second over half-open windows of `window_s` starting at the first
    arrival; percentiles use numpy's default linear interpolation. `hourly_rps` holds 24
    mean hour-of-day request rates when the trace covers at least one full day, else None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_requests: int = Field(ge=1)
    duration_s: float = Field(ge=0)
    window_s: float = Field(gt=0)
    n_windows: int = Field(ge=1)
    mean_rps: float = Field(ge=0)
    peak_window_rps: float = Field(gt=0)
    peak_window_index: int = Field(ge=0)
    input_tokens_p50: float
    input_tokens_p95: float
    input_tokens_p99: float
    input_tokens_mean: float
    input_tokens_max: int
    output_tokens_p50: float
    output_tokens_p95: float
    output_tokens_p99: float
    output_tokens_mean: float
    output_tokens_max: int
    peak_input_tokens_per_s: float = Field(ge=0)
    peak_output_tokens_per_s: float = Field(ge=0)
    hourly_rps: tuple[float, ...] | None = Field(default=None, min_length=24, max_length=24)
