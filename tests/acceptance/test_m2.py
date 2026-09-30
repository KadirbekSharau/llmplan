"""M2 acceptance tests: M2_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from llmplan.cli import app
from llmplan.errors import FetchError, ValidationError, WorkloadFormatError
from llmplan.workload import Distribution, Workload, compute_stats, generate, load_workload
from llmplan.workload.fetch import fetch_trace, load_manifest
from llmplan.workload.formats import detect
from llmplan.workload.formats.generic_csv import write_csv

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


# 9.4 Synthetic
IN_TOKENS = Distribution(kind="lognormal", mean=6.2, sigma=0.8, lo=16, hi=4096)
OUT_TOKENS = Distribution(kind="lognormal", mean=5.5, sigma=0.9, lo=1, hi=1024)


def _synth(duration_s: float = 3600, diurnal: tuple[float, ...] | None = None) -> Workload:
    return generate(
        rate_rps=5,
        duration_s=duration_s,
        input_tokens=IN_TOKENS,
        output_tokens=OUT_TOKENS,
        seed=1,
        diurnal=diurnal,
    )


def test_9_4_synthetic_is_deterministic_and_bounded(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(_synth(), first)
    write_csv(_synth(), second)
    assert first.read_bytes() == second.read_bytes()
    frame = _synth().frame
    assert len(frame) == pytest.approx(18_000, rel=0.05)
    assert frame["arrival_s"].iloc[0] == 0.0
    assert frame["input_tokens"].between(IN_TOKENS.lo, IN_TOKENS.hi).all()
    assert frame["output_tokens"].between(OUT_TOKENS.lo, OUT_TOKENS.hi).all()


def test_9_4_equal_diurnal_multipliers_keep_the_count() -> None:
    assert len(_synth(diurnal=(1.0,) * 24).frame) == pytest.approx(18_000, rel=0.05)


def test_9_4_zero_diurnal_hours_give_zero_hourly_rps() -> None:
    # Hours 0 and 23 stay non-zero: the first arrival is at 0.0, and a trace's duration ends
    # at its last arrival, so a silent hour 23 would shorten the trace below one day.
    zero_hours = set(range(6, 18))
    diurnal = tuple(0.0 if h in zero_hours else 1.0 for h in range(24))
    hourly = compute_stats(_synth(duration_s=86_400, diurnal=diurnal)).hourly_rps
    assert hourly is not None
    assert {h for h, rps in enumerate(hourly) if rps == 0.0} == zero_hours


# 9.5 Fetch safety
def test_9_5_fetch_without_yes_exits_2_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []
    monkeypatch.setattr(httpx.Client, "send", lambda self, *a, **k: sent.append(a))
    result = CliRunner().invoke(app, ["traces", "fetch", "azure2023-code", "--dest", str(tmp_path)])
    assert result.exit_code == 2
    assert sent == []
    assert list(tmp_path.iterdir()) == []


def test_9_5_checksum_mismatch_raises_and_leaves_no_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"not the trace"))
    with pytest.raises(FetchError, match="SHA-256"):
        fetch_trace("azure2023-code", tmp_path, yes=True, transport=transport)
    assert list(tmp_path.iterdir()) == []
    assert load_manifest()["azure2023-code"].sha256 is not None


# 9.6 Property tests
distributions = st.one_of(
    st.builds(Distribution, kind=st.just("fixed"), value=st.integers(1, 5000)),
    st.builds(
        Distribution,
        kind=st.just("lognormal"),
        mean=st.floats(0, 9),
        sigma=st.floats(0, 2),
        hi=st.integers(1, 131_072),
    ),
    st.integers(1, 4000).flatmap(
        lambda lo: st.builds(
            Distribution, kind=st.just("uniform"), lo=st.just(lo), hi=st.integers(lo, 8000)
        )
    ),
)


@settings(max_examples=60, deadline=None)
@given(
    rate_rps=st.floats(0.01, 50),
    duration_s=st.floats(1, 900),
    input_tokens=distributions,
    output_tokens=distributions,
    seed=st.integers(0, 2**32 - 1),
    window_s=st.floats(0.5, 300),
)
def test_9_6_properties(
    rate_rps: float,
    duration_s: float,
    input_tokens: Distribution,
    output_tokens: Distribution,
    seed: int,
    window_s: float,
) -> None:
    workload = generate(
        rate_rps=rate_rps,
        duration_s=duration_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        seed=seed,
    )
    stats = compute_stats(workload, window_s=window_s)
    assert np.all(np.diff(workload.frame["arrival_s"].to_numpy()) >= 0)
    # Design text: peak_window_rps >= mean_rps. That is false whenever the last window is
    # partial (9.2 itself has peak 0.1 < mean 10/90); see M2_NOTES.md. The true bound is the
    # peak against the mean over the windows the trace spans:
    peak_count = round(stats.peak_window_rps * stats.window_s)
    assert peak_count * stats.n_windows >= stats.n_requests
    assert stats.input_tokens_p99 >= stats.input_tokens_p95 >= stats.input_tokens_p50
