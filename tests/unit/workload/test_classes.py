from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pydantic
import pytest

from llmplan.errors import ValidationError
from llmplan.workload import Workload, compute_stats
from llmplan.workload.classes import DemandClass, assign_classes, classify, classify_spec


def trace(rows: Sequence[tuple[float, int, int]]) -> Workload:
    arrival, inputs, outputs = zip(*rows, strict=True)
    n = len(rows)
    return Workload(
        source="test",
        format="synthetic",
        frame=pd.DataFrame(
            {
                "arrival_s": np.asarray(arrival, dtype=np.float64),
                "input_tokens": np.asarray(inputs, dtype=np.int64),
                "output_tokens": np.asarray(outputs, dtype=np.int64),
                "model": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
                "tenant": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
            }
        ),
        dropped_rows=0,
    )


def grid(n: int = 200) -> Workload:
    """Inputs 1..n and outputs n..1 (reversed), one request per second."""
    return trace([(float(i), i + 1, n - i) for i in range(n)])


def test_quantile_2x2_splits_at_the_medians() -> None:
    classes = classify(grid(), input_bins=2, output_bins=2)
    # input median 100.5 -> cut 100; output median 100.5 -> cut 100. Inputs <= 100 have
    # outputs >= 101, so each input row holds one non-empty cell; the empty cell's output
    # range joins it, so the classes still tile the plane.
    assert [(c.input_lo, c.input_hi, c.output_lo, c.output_hi) for c in classes] == [
        (1, 100, 0, 200),
        (101, 200, 0, 200),
    ]
    assert [c.share for c in classes] == [0.5, 0.5]
    assert [c.index for c in classes] == [0, 1]
    assert classes[0].input_tokens_mean == 50.5
    assert classes[0].output_tokens_p50 == 150.5
    assert all(not c.notes for c in classes)  # empty cells merge silently


def test_one_by_one_matches_the_workload_stats() -> None:
    workload = grid(50)
    stats = compute_stats(workload)
    (whole,) = classify(workload, input_bins=1, output_bins=1)
    assert whole.peak_rps == stats.peak_window_rps
    assert whole.peak_output_tokens_per_s == stats.peak_output_tokens_per_s
    assert whole.input_tokens_p95 == stats.input_tokens_p95
    assert whole.output_tokens_mean == stats.output_tokens_mean
    assert (whole.input_lo, whole.input_hi, whole.output_lo, whole.output_hi) == (1, 50, 0, 50)


def test_demand_is_measured_in_the_fleet_wide_peak_windows() -> None:
    # Window 0: three short requests. Window 1: one short and four long (the request peak).
    # Output tokens: window 0 = 3 x 10 = 30, window 1 = 10 + 4 x 5 = 30: the first maximum
    # (window 0) is the token peak, as in compute_stats.
    rows = [(0.0, 10, 10), (1.0, 10, 10), (2.0, 10, 10), (60.0, 10, 10)]
    rows += [(61.0 + i, 1000, 5) for i in range(4)]
    short, long = classify(trace(rows), input_bins=2, output_bins=1)
    assert short.peak_rps == 1 / 60  # its own busiest window (0) holds 3, but not the fleet's
    assert long.peak_rps == 4 / 60
    assert short.peak_output_tokens_per_s == 30 / 60
    assert long.peak_output_tokens_per_s == 0.0
    assert short.peak_rps + long.peak_rps == compute_stats(trace(rows)).peak_window_rps


def test_tiny_cells_merge_into_the_nearest_neighbour() -> None:
    # 199 requests with 10 output tokens and one with 1,000: the high-output cell (0.5%) of
    # the single input row merges into the low one.
    rows = [(float(i), 100, 10) for i in range(199)] + [(199.0, 100, 1000)]
    (only,) = classify(trace(rows), method="fixed", edges=((), (100,)))
    assert (only.output_lo, only.output_hi) == (0, 1000)
    assert only.notes == ("outputs 101..1000 (0.50% of requests, under 1%) merged into this class",)


def test_tiny_input_rows_merge_by_nearest_mean() -> None:
    # Inputs: 100 x 10 tokens, 1 x 500, 100 x 900. The 500-token row (0.5%) is nearer the
    # high row's mean (|500 - 900| = 400) than the low row's (|500 - 10| = 490).
    rows = [(float(i), 10, 1) for i in range(100)] + [(100.0, 500, 1)]
    rows += [(101.0 + i, 900, 1) for i in range(100)]
    low, high = classify(trace(rows), method="fixed", edges=((100, 600), ()))
    assert (low.input_lo, low.input_hi) == (1, 100)
    assert (high.input_lo, high.input_hi) == (101, 900)
    assert high.share == 101 / 201
    assert high.notes[0].startswith("inputs 101..600 (0.50% of requests")


def test_fixed_edges_outside_the_data_are_dropped_with_a_note() -> None:
    classes = classify(grid(), method="fixed", edges=((150, 5000), ()))
    assert [(c.input_lo, c.input_hi) for c in classes] == [(1, 150), (151, 200)]
    assert classes[0].notes == (
        "input edges [5000] lie outside the observed 1..200 and were dropped",
    )
    assert classes[1].notes == ()


def test_empty_bins_from_fixed_edges_disappear() -> None:
    rows = [(float(i), 10, 1) for i in range(50)] + [(50.0 + i, 900, 1) for i in range(50)]
    classes = classify(trace(rows), method="fixed", edges=((20, 40, 100), ()))
    assert [(c.input_lo, c.input_hi, c.share) for c in classes] == [(1, 100, 0.5), (101, 900, 0.5)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_bins": 0}, "input_bins must be between 1 and 6"),
        ({"output_bins": 7}, "output_bins must be between 1 and 6"),
        ({"edges": ((10,), ())}, "edges are only used with method='fixed'"),
        ({"method": "fixed"}, "method='fixed' needs edges"),
        ({"method": "fixed", "edges": ((0,), ())}, "input edges must be fewer than 6"),
        ({"method": "fixed", "edges": ((), (5, 5))}, "output edges must be strictly increasing"),
        ({"window_s": 0.0}, "window_s must be a positive number"),
    ],
)
def test_classify_rejects_bad_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        classify(grid(), **kwargs)  # type: ignore[arg-type]  # deliberately bad values


def test_classify_spec_forms() -> None:
    workload = grid()
    assert classify_spec(workload, "1") == ()
    assert len(classify_spec(workload, " 2X2 ")) == 2
    fixed = classify_spec(workload, "fixed:50,150/")
    assert [(c.input_lo, c.input_hi) for c in fixed] == [(1, 50), (51, 150), (151, 200)]
    assert len(classify_spec(workload, "fixed:/100")) == 2


@pytest.mark.parametrize("spec", ["2", "2x", "fixed:1,2", "fixed:a/1", "auto"])
def test_classify_spec_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValidationError, match="classes"):
        classify_spec(grid(), spec)


def test_assign_classes_inside_and_outside_the_bounds() -> None:
    classes = classify(grid(), input_bins=2, output_bins=2)
    inputs = np.array([1, 100, 101, 5000, 50, 150], dtype=np.int64)
    outputs = np.array([150, 200, 0, 0, 999, 999], dtype=np.int64)
    # Outside both: (5000, 0) is 4,800 tokens from class 1 and 4,900 from class 0; (50, 999)
    # is 799 from class 0 and 51 + 799 from class 1; (150, 999) the other way round.
    assert assign_classes(classes, inputs, outputs).tolist() == [0, 0, 1, 1, 0, 1]
    assert assign_classes((), inputs, outputs).tolist() == [0] * 6


def test_demand_class_bounds_are_validated() -> None:
    (whole,) = classify(grid(10), input_bins=1, output_bins=1)
    with pytest.raises(pydantic.ValidationError, match="lo <= hi"):
        DemandClass.model_validate({**whole.model_dump(), "input_lo": 11})
