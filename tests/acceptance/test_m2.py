"""M2 acceptance tests: M2_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmplan.errors import ValidationError, WorkloadFormatError
from llmplan.workload import compute_stats, load_workload
from llmplan.workload.formats import detect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def approx(value: float) -> object:
    return pytest.approx(value, rel=1e-9)


# 9.1 Fixture round-trips
@pytest.mark.parametrize("key", ["csv", "azure2023", "azure2024", "burstgpt"])
def test_9_1_fixture_round_trips(key: str) -> None:
    path = FIXTURES / f"workload_{key}_50.csv"
    assert detect(path) == key
    workload = load_workload(path)
    frame = workload.frame
    assert workload.format == key
    assert len(frame) == 50
    assert list(frame.columns) == ["arrival_s", "input_tokens", "output_tokens", "model", "tenant"]
    assert frame["arrival_s"].dtype == np.float64
    assert frame["input_tokens"].dtype == np.int64
    assert frame["output_tokens"].dtype == np.int64
    for column in ("model", "tenant"):
        assert isinstance(frame[column].dtype, pd.StringDtype)
        assert frame[column].dtype.na_value is pd.NA
    assert frame["arrival_s"].iloc[0] == 0.0
    assert frame["arrival_s"].is_monotonic_increasing
    assert workload.dropped_rows == 0


# 9.2 Hand-computed stats on tests/fixtures/workload_10.csv
def test_9_2_hand_computed_stats() -> None:
    stats = compute_stats(load_workload(FIXTURES / "workload_10.csv"), window_s=60)
    assert stats.n_requests == 10
    assert stats.duration_s == approx(90.0)
    assert stats.n_windows == 2
    assert stats.mean_rps == approx(10 / 90)
    assert stats.peak_window_rps == approx(0.1)
    assert stats.peak_window_index == 0
    assert stats.input_tokens_p50 == approx(550.0)
    assert stats.input_tokens_p95 == approx(955.0)
    assert stats.input_tokens_p99 == approx(991.0)
    assert stats.input_tokens_mean == approx(550.0)
    assert stats.input_tokens_max == 1000
    assert stats.output_tokens_p50 == approx(55.0)
    assert stats.output_tokens_p95 == approx(95.5)
    assert stats.output_tokens_p99 == approx(99.1)
    assert stats.peak_input_tokens_per_s == approx(3400 / 60)
    assert stats.peak_output_tokens_per_s == approx(340 / 60)
    assert stats.hourly_rps is None


# 9.3 Validation
def _token_csv(tmp_path: Path, n_rows: int, n_zero_input: int) -> Path:
    rows = [f"{i},{0 if i < n_zero_input else 100},10" for i in range(n_rows)]
    path = tmp_path / "trace.csv"
    path.write_text("arrival_s,input_tokens,output_tokens\n" + "\n".join(rows) + "\n")
    return path


def test_9_3_both_time_columns_rejected(tmp_path: Path) -> None:
    path = tmp_path / "both.csv"
    path.write_text("arrival_s,timestamp,input_tokens,output_tokens\n0,2024-01-01T00:00:00,1,1\n")
    with pytest.raises(WorkloadFormatError):
        load_workload(path)


def test_9_3_sixty_percent_invalid_rows_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkloadFormatError):
        load_workload(_token_csv(tmp_path, n_rows=50, n_zero_input=30))


def test_9_3_ten_percent_invalid_rows_dropped_with_note(tmp_path: Path) -> None:
    workload = load_workload(_token_csv(tmp_path, n_rows=50, n_zero_input=5))
    assert workload.dropped_rows == 5
    assert len(workload.frame) == 45
    assert workload.notes


def test_9_3_zero_window_rejected() -> None:
    with pytest.raises(ValidationError):
        compute_stats(load_workload(FIXTURES / "workload_10.csv"), window_s=0)
