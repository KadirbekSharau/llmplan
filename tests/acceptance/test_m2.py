"""M2 acceptance tests: M2_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmplan.workload import compute_stats, load_workload

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def approx(value: float) -> object:
    return pytest.approx(value, rel=1e-9)


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
