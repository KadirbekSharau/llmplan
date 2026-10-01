"""Window integrals and maxima of the replica state logs (M5_DESIGN.md section 7).

A log holds a value after each event; between events it is constant (busy slots, queue
depth, KV under full reservation) or, for KV under incremental accounting (M7), changes
linearly at the logged slope. Window `k` is `[bounds[k], bounds[k + 1])`; the value is 0
before the first event, and states that last zero time (several events at one instant)
still count toward the maximum. Moved out of `timeline.py` when M7 added the linear case.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _merge(
    time_s: FloatArray, columns: tuple[FloatArray, ...], bounds: FloatArray
) -> tuple[FloatArray, tuple[FloatArray, ...], npt.NDArray[np.int64]]:
    """Insert the window bounds into the log (carrying the state in force there, with the
    value advanced along the slope when `columns` = (value, slope)), sorted by time with
    bounds after events at equal times; returns times, columns, and each point's window."""
    time_s = np.concatenate([bounds[:1], time_s])  # explicit idle state at the origin
    columns = tuple(np.concatenate([[0.0], c]) for c in columns)
    at = np.searchsorted(time_s, bounds, side="right") - 1
    at_bounds = [c[at] for c in columns]
    if len(columns) == 2:
        at_bounds[0] = at_bounds[0] + at_bounds[1] * (bounds - time_s[at])
    times = np.concatenate([time_s, bounds])
    merged = tuple(np.concatenate([c, b]) for c, b in zip(columns, at_bounds, strict=True))
    order = np.argsort(times, kind="stable")
    times = times[order]
    merged = tuple(c[order] for c in merged)
    return times, merged, np.searchsorted(bounds, times, side="right") - 1


def step_windows(
    time_s: FloatArray, value: FloatArray, bounds: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Integral and maximum per window of a step function that is 0 before `time_s[0]` and
    takes `value[i]` from `time_s[i]` on."""
    n = len(bounds) - 1
    times, (values,), window = _merge(time_s, (value,), bounds)
    inside = window < n
    integral = np.bincount(
        window[:-1][inside[:-1]],
        weights=(np.diff(times) * values[:-1])[inside[:-1]],
        minlength=n,
    ).astype(np.float64)
    maximum = np.zeros(n)
    np.maximum.at(maximum, window[inside], values[inside])
    return integral, maximum


def linear_windows(
    time_s: FloatArray, value: FloatArray, slope: FloatArray, bounds: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Integral and maximum per window of a piecewise-linear function that is 0 before
    `time_s[0]` and equals `value[i] + slope[i] * (t - time_s[i])` from `time_s[i]` until
    the next event. The maximum of a window is over its points and the ends of its
    segments (a linear piece peaks at an end)."""
    n = len(bounds) - 1
    times, (values, slopes), window = _merge(time_s, (value, slope), bounds)
    dt = np.diff(times)
    ends = values[:-1] + slopes[:-1] * dt
    inside = window < n
    segment = inside[:-1]
    integral = np.bincount(
        window[:-1][segment],
        weights=((values[:-1] + ends) / 2 * dt)[segment],
        minlength=n,
    ).astype(np.float64)
    maximum = np.zeros(n)
    np.maximum.at(maximum, window[inside], values[inside])
    np.maximum.at(maximum, window[:-1][segment], ends[segment])
    return integral, maximum
