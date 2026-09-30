"""Deterministic synthetic workload generator (M2_DESIGN.md section 6)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic

from llmplan.errors import ValidationError
from llmplan.workload.schema import Distribution, Workload

MAX_EXPECTED_REQUESTS = 50_000_000  # rate_rps * duration_s above this is refused
SECONDS_PER_HOUR = 3600.0
_SYNTAX = "expected fixed:N, lognormal:MEAN:SIGMA[:LO:HI], or uniform:LO:HI"


def generate(
    *,
    rate_rps: float,
    duration_s: float,
    input_tokens: Distribution,
    output_tokens: Distribution,
    seed: int,
    diurnal: tuple[float, ...] | None = None,
) -> Workload:
    """Generate a Poisson workload; the same arguments always give the same frame.

    Arrivals start at 0.0 and follow exponential inter-arrival times with mean
    `1 / rate_rps` from `numpy.random.default_rng(seed)`; every arrival at or before
    `duration_s` is kept. With `diurnal` (24 non-negative multipliers, hour of day =
    `floor(arrival_s / 3600) % 24`), arrivals are generated at `rate_rps` and each is kept
    with probability `diurnal[h] / max(diurnal)`, so `rate_rps` is the rate of the peak hour
    and all-equal multipliers change nothing. The arrival at 0.0 is always kept so hours
    stay aligned. Token lengths are then drawn from the same generator (inputs, then
    outputs), rounded and clipped to each distribution's `[lo, hi]`. Raises
    `ValidationError` for non-positive rates or durations, more than 50 million expected
    requests, an input distribution that allows 0 tokens, or a malformed `diurnal`.
    """
    _check_args(rate_rps, duration_s, input_tokens, seed, diurnal)
    rng = np.random.default_rng(seed)
    arrival = _poisson_arrivals(rng, rate_rps, duration_s)
    if diurnal is not None:
        multipliers = np.asarray(diurnal, dtype=np.float64)
        hour = np.floor(arrival / SECONDS_PER_HOUR).astype(np.int64) % 24
        keep = rng.random(len(arrival)) < multipliers[hour] / multipliers.max()
        keep[0] = True
        arrival = arrival[keep]
    n = len(arrival)
    frame = pd.DataFrame(
        {
            "arrival_s": arrival,
            "input_tokens": _draw(rng, input_tokens, n),
            "output_tokens": _draw(rng, output_tokens, n),
            "model": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
            "tenant": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
        }
    )
    note = f"synthetic Poisson arrivals: rate_rps={rate_rps}, duration_s={duration_s}"
    if diurnal is not None:
        note += ", diurnal thinning"
    return Workload(
        source=f"synthetic:{seed}", format="synthetic", frame=frame, dropped_rows=0, notes=(note,)
    )


def parse_distribution(spec: str) -> Distribution:
    """Parse the CLI syntax `fixed:N`, `lognormal:MEAN:SIGMA[:LO:HI]`, or `uniform:LO:HI`.

    Raises `ValidationError` naming `spec` when the syntax or the values are invalid.
    """
    kind, *args = spec.strip().split(":")
    try:
        if kind == "fixed" and len(args) == 1:
            return Distribution(kind="fixed", value=int(args[0]))
        if kind == "lognormal" and len(args) in (2, 4):
            bounds = {"lo": int(args[2]), "hi": int(args[3])} if len(args) == 4 else {}
            return Distribution(
                kind="lognormal", mean=float(args[0]), sigma=float(args[1]), **bounds
            )
        if kind == "uniform" and len(args) == 2:
            return Distribution(kind="uniform", lo=int(args[0]), hi=int(args[1]))
    except ValueError as exc:  # int()/float() failures and pydantic errors alike
        detail = exc.errors()[0]["msg"] if isinstance(exc, pydantic.ValidationError) else str(exc)
        raise ValidationError(f"invalid distribution {spec!r}: {detail}") from None
    raise ValidationError(f"invalid distribution {spec!r}: {_SYNTAX}")


def _check_args(
    rate_rps: float,
    duration_s: float,
    input_tokens: Distribution,
    seed: int,
    diurnal: tuple[float, ...] | None,
) -> None:
    if not (math.isfinite(rate_rps) and rate_rps > 0):
        raise ValidationError(f"rate_rps must be a positive number, got {rate_rps}")
    if not (math.isfinite(duration_s) and duration_s > 0):
        raise ValidationError(f"duration_s must be a positive number, got {duration_s}")
    if rate_rps * duration_s > MAX_EXPECTED_REQUESTS:
        raise ValidationError(
            f"rate_rps * duration_s = {rate_rps * duration_s:.0f} expected requests exceeds "
            f"{MAX_EXPECTED_REQUESTS}"
        )
    if input_tokens.lo < 1:
        raise ValidationError("input_tokens distribution must have lo >= 1")
    if seed < 0:
        raise ValidationError(f"seed must be >= 0, got {seed}")
    if diurnal is not None:
        values = np.asarray(diurnal, dtype=np.float64)
        if values.shape != (24,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValidationError("diurnal must be 24 finite multipliers >= 0")
        if values.max() <= 0:
            raise ValidationError("diurnal must have at least one positive multiplier")


def _poisson_arrivals(
    rng: np.random.Generator, rate_rps: float, duration_s: float
) -> npt.NDArray[np.float64]:
    block = int(rate_rps * duration_s * 1.05) + 64
    gaps = np.empty(0, dtype=np.float64)
    while True:
        gaps = np.concatenate([gaps, rng.exponential(1.0 / rate_rps, block)])
        cumulative = np.cumsum(gaps)
        if cumulative[-1] > duration_s:
            break
    arrival = np.concatenate([[0.0], cumulative])
    return arrival[arrival <= duration_s]


def _draw(rng: np.random.Generator, dist: Distribution, n: int) -> npt.NDArray[np.int64]:
    if dist.kind == "fixed":
        return np.full(n, dist.value, dtype=np.int64)
    if dist.kind == "uniform":
        return rng.integers(dist.lo, dist.hi, size=n, endpoint=True, dtype=np.int64)
    if dist.mean is None or dist.sigma is None:  # excluded by Distribution's validator
        raise ValidationError("lognormal distribution requires mean and sigma")
    raw = rng.lognormal(dist.mean, dist.sigma, n)
    return np.clip(np.rint(raw), dist.lo, dist.hi).astype(np.int64)
