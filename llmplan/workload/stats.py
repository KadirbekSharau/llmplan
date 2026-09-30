"""Workload statistics (M2_DESIGN.md section 5): rate windows, token percentiles, diurnal."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from llmplan.errors import ValidationError
from llmplan.workload.schema import Workload, WorkloadStats

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24


def compute_stats(workload: Workload, window_s: float = 60.0) -> WorkloadStats:
    """Summarize `workload` over half-open windows of `window_s` seconds.

    Windows are `[k*window_s, (k+1)*window_s)` from the first arrival. The last window is
    usually partial and its counts are still divided by the full `window_s`: that
    under-counts its rate, which is acceptable for peak detection (a partial window never
    inflates the peak). `peak_window_index` is the first window with the highest count.
    Percentiles use `numpy.percentile` with the default "linear" method.

    `hourly_rps` buckets arrivals by `floor(arrival_s / 3600) % 24` and divides each
    bucket's count by `full_days * 3600`. Hours are windows too: a trace covers
    `floor(duration_s / 3600) + 1` hour windows (the last may be partial, as above),
    `full_days` is that count // 24, and only arrivals inside those full days are bucketed.
    It is None when the trace covers fewer than 24 hour windows. Raises `ValidationError`
    when `window_s` is not a positive finite number.
    """
    if not (math.isfinite(window_s) and window_s > 0):
        raise ValidationError(f"window_s must be a positive number of seconds, got {window_s}")
    frame = workload.frame
    arrival: npt.NDArray[np.float64] = frame["arrival_s"].to_numpy(dtype=np.float64)
    inputs: npt.NDArray[np.int64] = frame["input_tokens"].to_numpy(dtype=np.int64)
    outputs: npt.NDArray[np.int64] = frame["output_tokens"].to_numpy(dtype=np.int64)

    start = float(arrival.min())
    duration_s = float(arrival.max()) - start
    window = np.floor((arrival - start) / window_s).astype(np.int64)
    n_windows = int(window.max()) + 1
    counts = np.bincount(window, minlength=n_windows)
    input_sums = np.bincount(window, weights=inputs, minlength=n_windows)
    output_sums = np.bincount(window, weights=outputs, minlength=n_windows)
    in_p50, in_p95, in_p99 = (float(v) for v in np.percentile(inputs, [50, 95, 99]))
    out_p50, out_p95, out_p99 = (float(v) for v in np.percentile(outputs, [50, 95, 99]))
    n = len(arrival)

    return WorkloadStats(
        n_requests=n,
        duration_s=duration_s,
        window_s=window_s,
        n_windows=n_windows,
        mean_rps=n / duration_s if duration_s > 0 else 0.0,
        peak_window_rps=float(counts.max()) / window_s,
        peak_window_index=int(counts.argmax()),
        input_tokens_p50=in_p50,
        input_tokens_p95=in_p95,
        input_tokens_p99=in_p99,
        input_tokens_mean=float(inputs.mean()),
        input_tokens_max=int(inputs.max()),
        output_tokens_p50=out_p50,
        output_tokens_p95=out_p95,
        output_tokens_p99=out_p99,
        output_tokens_mean=float(outputs.mean()),
        output_tokens_max=int(outputs.max()),
        peak_input_tokens_per_s=float(input_sums.max()) / window_s,
        peak_output_tokens_per_s=float(output_sums.max()) / window_s,
        hourly_rps=_hourly_rps(arrival - start, duration_s),
    )


def _hourly_rps(offset_s: npt.NDArray[np.float64], duration_s: float) -> tuple[float, ...] | None:
    hour = np.floor(offset_s / SECONDS_PER_HOUR).astype(np.int64)
    full_days = (math.floor(duration_s / SECONDS_PER_HOUR) + 1) // HOURS_PER_DAY
    if full_days < 1:
        return None
    in_full_days = hour[hour < full_days * HOURS_PER_DAY]
    counts = np.bincount(in_full_days % HOURS_PER_DAY, minlength=HOURS_PER_DAY)
    return tuple(float(c) / (full_days * SECONDS_PER_HOUR) for c in counts)
