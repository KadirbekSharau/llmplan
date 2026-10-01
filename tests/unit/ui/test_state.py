from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.errors import ValidationError
from llmplan.planner import SLO, PlanRequest
from llmplan.simulate import SimOptions
from llmplan.ui import presets, state
from llmplan.workload import compute_stats
from tests.conftest import UseFakePerf
from tests.fake_planner import GPUS, MODEL, ROW_A, FakePerf, request, stats, workload

CATALOG = load_gpus()
SHIPPED = load_prices(gpus=CATALOG)


def test_price_table_round_trips_through_price_row() -> None:
    frame = state.prices_frame(SHIPPED)
    assert list(frame.columns) == list(state.PRICE_COLUMNS)
    assert state.price_rows(frame, CATALOG) == SHIPPED


def test_price_cells_from_the_editor_are_normalized() -> None:
    frame = state.prices_frame(SHIPPED[:1])
    frame["gpu_count"] = frame["gpu_count"].astype("float64")  # a number column
    frame["as_of"] = pd.to_datetime(frame["as_of"])  # the date column may come back as Timestamp
    frame.loc[0, "region"] = float("nan")
    blank = pd.DataFrame([{column: None for column in state.PRICE_COLUMNS}])
    blank.loc[0, "instance"] = "  "
    (row,) = state.price_rows(pd.concat([frame, blank], ignore_index=True), CATALOG)
    assert row.gpu_count == SHIPPED[0].gpu_count
    assert row.as_of == date(2026, 9, 30)
    assert row.region is None


def test_invalid_price_rows_name_the_row_and_field() -> None:
    frame = state.prices_frame(SHIPPED)
    frame.loc[2, "price_usd_per_hour"] = -1.0
    with pytest.raises(ValidationError, match="price row 2: field 'price_usd_per_hour'"):
        state.price_rows(frame, CATALOG)
    frame = state.prices_frame(SHIPPED)
    frame.loc[0, "gpu_id"] = "b200"
    with pytest.raises(ValidationError, match="price row 0: unknown gpu_id 'b200'"):
        state.price_rows(frame, CATALOG)
    with pytest.raises(ValidationError, match="no rows"):
        state.price_rows(state.prices_frame(SHIPPED).iloc[0:0], CATALOG)


def _request(**overrides: object) -> PlanRequest:
    args: dict[str, object] = {
        "model": MODEL,
        "stats": stats(1.0),
        "slo": SLO(),
        "gpus": CATALOG,
        "prices": SHIPPED,
        "gpu_ids": ["l4-24gb"],
        "providers": ["aws"],
        "tensor_parallel": [1],
        "dtypes": ["bf16"],
        "max_num_seqs": [64],
        "max_model_len": 8192,
        "perf_backend": "roofline",
        "solver": "highs",
        "time_limit_s": 10.0,
    }
    return state.build_request(**{**args, **overrides})


def test_build_request_caps_the_time_limit_and_names_empty_selections() -> None:
    assert _request().options.time_limit_s == 10.0
    assert _request(time_limit_s=600.0).options.time_limit_s == presets.MAX_TIME_LIMIT_S
    for name, field in (("GPU", "gpu_ids"), ("provider", "providers"), ("dtype", "dtypes")):
        with pytest.raises(ValidationError, match=f"select at least one {name}"):
            _request(**{field: []})
    with pytest.raises(ValidationError, match="invalid time_limit_s: Input should be greater"):
        _request(time_limit_s=0.0)


@pytest.mark.parametrize(
    ("span_s", "window_s"),
    [(0.0, 1.0), (90.0, 1.0), (3600.0, 50.0), (7200.0, 50.0), (604_800.0, 5000.0), (1.0, 0.01)],
)
def test_timeline_window(span_s: float, window_s: float) -> None:
    assert state.timeline_window_s(span_s) == window_s


@given(st.floats(min_value=1e-3, max_value=1e8, allow_nan=False))
def test_timeline_window_gives_60_to_151_windows(span_s: float) -> None:
    window = state.timeline_window_s(span_s)
    assert state.MIN_WINDOWS <= math.floor(span_s / window) + 1 <= 151


def test_sim_options_and_cache_key() -> None:
    trace = workload([0.0, 30.0, 7200.0], 100, 10)
    options = state.sim_options(compute_stats(trace))
    assert options == SimOptions(window_s=50.0, max_requests=presets.MAX_SIM_REQUESTS)
    key = state.cache_key(_request(), options, trace)
    assert key == state.cache_key(_request(), options, trace)
    assert key != state.cache_key(_request(dtypes=["fp8"]), options, trace)
    assert key != state.cache_key(_request(), options, workload([0.0, 30.0, 7200.0], 101, 10))
    cheaper = (SHIPPED[0].model_copy(update={"price_usd_per_hour": 1.0}), *SHIPPED[1:])
    assert key != state.cache_key(_request(prices=cheaper), options, trace)


def test_run_plan_widens_the_window_when_queues_outlast_the_trace(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-a", 1): FakePerf(rps=1.0, tpot_ms_p95=20.0, prefill_tokens_per_s=1000.0)})
    plan_request = request(1.0, (ROW_A,), max_num_seqs_choices=(1,))
    trace = workload([0.25 * i for i in range(400)], 100, 40)  # 0.5 s service, 2x overload
    narrow = SimOptions(window_s=0.5)
    run = state.run_plan(plan_request, trace, narrow, GPUS)
    assert run.timeline.options.window_s == 2.0  # 400 windows of 0.5 s -> 200 s span / 150
    assert len(run.timeline.windows) <= state.MAX_WINDOWS
    assert run.result.cost_usd_per_day > 0
    sections = state.assumptions(run)
    assert list(sections) == [
        "Plan",
        "Performance estimates of the chosen replicas",
        "Simulation",
    ]
    assert sections["Performance estimates of the chosen replicas"] == ("fake backend",)


def test_admit_plan_brakes_at_30_per_hour() -> None:
    times: tuple[float, ...] = ()
    for i in range(presets.MAX_PLANS_PER_HOUR):
        admitted, times = state.admit_plan(times, 1000.0 + i)
        assert admitted
    admitted, times = state.admit_plan(times, 1100.0)
    assert not admitted
    assert len(times) == presets.MAX_PLANS_PER_HOUR
    admitted, times = state.admit_plan(times, 1000.0 + 3600.5)  # the first one expired
    assert admitted
    assert len(times) == presets.MAX_PLANS_PER_HOUR


def test_as_of_range() -> None:
    assert state.as_of_range(SHIPPED) == (date(2026, 9, 30), date(2026, 9, 30))


def test_run_plan_with_classes_routes_by_weight_and_compares() -> None:
    from tests.acceptance.test_m7 import two_class_request, two_class_workload

    trace = two_class_workload()
    req = two_class_request(trace)
    run = state.run_plan(req, trace, SimOptions(), req.gpus)
    assert run.timeline.options.routing == "class_weighted"
    assert run.single_class_cost_usd_per_day == 144.0
    assert run.class_saving_pct == pytest.approx(100 / 3)  # $96 against $144 (M7_NOTES.md)
    # Only the short class has demand and only SA is priced: without classes SA misses the
    # whole workload's TTFT, so there is no single-class fleet to compare with.
    short, long = req.classes
    idle = long.model_copy(update={"peak_rps": 0.0, "peak_output_tokens_per_s": 0.0})
    only_sa = req.model_copy(update={"classes": (short, idle), "prices": (req.prices[0],)})
    alone = state.run_plan(only_sa, trace, SimOptions(), req.gpus)
    assert alone.single_class_cost_usd_per_day is None
    assert alone.class_saving_pct is None
    from llmplan.ui import views  # the page's wording of both cases

    assert views._saving(run) == "33.3% (sized for the mean request: $144.00/day)"
    assert views._saving(alone).startswith("n/a (no fleet without classes")
    legacy = state.run_plan(req.model_copy(update={"classes": ()}), trace, SimOptions(), req.gpus)
    assert legacy.timeline.options.routing == "least_outstanding"
    assert legacy.single_class_cost_usd_per_day is None
