from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from llmplan.errors import ValidationError
from llmplan.workload.schema import Workload
from llmplan.workload.stats import compute_stats


def workload(arrival_s: Sequence[float], tokens: Sequence[int] | None = None) -> Workload:
    n = len(arrival_s)
    values = np.array(tokens if tokens is not None else [1] * n, dtype=np.int64)
    df = pd.DataFrame(
        {
            "arrival_s": np.array(arrival_s, dtype=np.float64),
            "input_tokens": values,
            "output_tokens": values,
            "model": pd.array([None] * n, dtype="string"),
            "tenant": pd.array([None] * n, dtype="string"),
        }
    )
    return Workload(source="test", format="csv", frame=df, dropped_rows=0)


@pytest.mark.parametrize("window_s", [0.0, -1.0, float("nan"), float("inf")])
def test_window_must_be_positive_finite(window_s: float) -> None:
    with pytest.raises(ValidationError, match="window_s"):
        compute_stats(workload([0.0]), window_s=window_s)


def test_single_request() -> None:
    stats = compute_stats(workload([0.0], [7]), window_s=10)
    assert stats.duration_s == 0.0
    assert stats.mean_rps == 0.0
    assert stats.n_windows == 1
    assert stats.peak_window_rps == pytest.approx(0.1)
    assert stats.input_tokens_p99 == 7.0
    assert stats.output_tokens_max == 7


def test_window_edges_are_half_open_and_partial_window_uses_full_width() -> None:
    # 120.0 opens window 2, which holds one request but is still divided by 60 s.
    stats = compute_stats(workload([0.0, 59.999, 60.0, 120.0], [1, 1, 5, 9]), window_s=60)
    assert stats.n_windows == 3
    assert stats.peak_window_index == 0  # first of the tied maxima (2, 1, 1 -> window 0)
    assert stats.peak_window_rps == pytest.approx(2 / 60)
    assert stats.peak_input_tokens_per_s == pytest.approx(9 / 60)


def test_hourly_rps_absent_below_24_hour_windows() -> None:
    assert compute_stats(workload([0.0, 82_799.0])).hourly_rps is None


def test_hourly_rps_for_one_day_ending_just_before_86400() -> None:
    arrivals = [h * 3600.0 + 10 for h in range(24)] + [86_399.5]
    arrivals[0] = 0.0
    stats = compute_stats(workload(arrivals))
    assert stats.hourly_rps is not None
    assert stats.hourly_rps[0] == pytest.approx(1 / 3600)
    assert stats.hourly_rps[23] == pytest.approx(2 / 3600)


def test_hourly_rps_averages_full_days_and_ignores_partial_day() -> None:
    # Two full days (hour windows 0..47) plus arrivals in day three, hour 0 and hour 5.
    arrivals = [0.0, 3600.0, 86_400.0, 86_400.0 + 3600.0, 172_800.0, 172_800.0 + 5 * 3600]
    stats = compute_stats(workload(arrivals))
    assert stats.hourly_rps is not None
    assert stats.hourly_rps[0] == pytest.approx(2 / (2 * 3600))
    assert stats.hourly_rps[1] == pytest.approx(2 / (2 * 3600))
    assert stats.hourly_rps[5] == 0.0
    assert sum(stats.hourly_rps) == pytest.approx(4 / 7200)
