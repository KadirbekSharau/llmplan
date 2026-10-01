"""M7 acceptance tests: M7_DESIGN.md sections 8 and 8b. These define done; do not relax them.

Expected values are derived by hand in docs/milestones/M7_NOTES.md ("Acceptance arithmetic").
The two-class instance uses the shape-aware fake backend (`fake_shape`, tests/fake_planner.py):
capacity = max_num_seqs / (input_mean / prefill + output_mean * tpot), TTFT p95 =
input_p95 / prefill, so per-class estimates follow from the class token statistics alone.
M7 names are imported inside the tests so this file was collectable before M7 existed.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import load_model
from llmplan.cli import app
from llmplan.errors import InfeasiblePlan
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.planner.result import PlanResult
from llmplan.workload import Workload, WorkloadStats, compute_stats, load_workload
from tests.conftest import UseFakeClassPerf, UseFakePerf
from tests.fake_planner import (
    MODEL,
    ROW_A,
    ROW_B,
    ROW_SA,
    ROW_SB,
    SHAPE_GPUS,
    FakePerf,
    request,
    workload,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TEN = str(FIXTURES / "workload_10.csv")
FIFTY = str(FIXTURES / "workload_csv_50.csv")
NEW_PLAN_FIELDS = ("classes", "routing", "class_binding")
runner = CliRunner()

A1, B1 = ("fake-a", 1), ("fake-b", 1)
A_B = {A1: FakePerf(rps=3), B1: FakePerf(rps=4)}


# --- helpers ---------------------------------------------------------------------------


def single_class(stats: WorkloadStats) -> Any:
    """One explicit demand class equal to the whole workload (what classify(1, 1) gives)."""
    from llmplan.workload.classes import DemandClass

    return DemandClass(
        index=0,
        input_lo=1,
        input_hi=stats.input_tokens_max,
        output_lo=0,
        output_hi=stats.output_tokens_max,
        share=1.0,
        peak_rps=stats.peak_window_rps,
        peak_output_tokens_per_s=stats.peak_output_tokens_per_s,
        input_tokens_mean=stats.input_tokens_mean,
        output_tokens_mean=stats.output_tokens_mean,
        input_tokens_p50=stats.input_tokens_p50,
        output_tokens_p50=stats.output_tokens_p50,
        input_tokens_p95=stats.input_tokens_p95,
        output_tokens_p95=stats.output_tokens_p95,
    )


def with_single_class(req: PlanRequest) -> PlanRequest:
    """`req` with one explicit class equal to its whole workload."""
    assert "classes" in PlanRequest.model_fields  # model_copy would accept an unknown key
    return req.model_copy(update={"classes": (single_class(req.stats),)})


def without_new_fields(result: PlanResult) -> str:
    """`model_dump_json` without the fields M7 added (on the result and its baseline)."""
    exclude: dict[str, Any] = dict.fromkeys(NEW_PLAN_FIELDS, True)
    exclude["baseline"] = set(NEW_PLAN_FIELDS)
    return result.model_dump_json(exclude=exclude)


def strip_new_keys(doc: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k not in keys}
    if isinstance(out.get("baseline"), dict):
        out["baseline"] = strip_new_keys(out["baseline"], keys)
    return out


def two_class_workload() -> Workload:
    """600 requests, one every 0.1 s: even ones short (100 in / 10 out), odd ones long
    (2000 in / 65 out). One 60 s window holds them all."""
    n = 600
    short = np.arange(n) % 2 == 0
    return Workload(
        source="test",
        format="synthetic",
        frame=pd.DataFrame(
            {
                "arrival_s": np.arange(n, dtype=np.float64) * 0.1,
                "input_tokens": np.where(short, 100, 2000).astype(np.int64),
                "output_tokens": np.where(short, 10, 65).astype(np.int64),
                "model": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
                "tenant": pd.Series(pd.NA, index=pd.RangeIndex(n), dtype="string"),
            }
        ),
        dropped_rows=0,
    )


def two_class_request(trace: Workload, *, classes: bool = True) -> PlanRequest:
    """Rows SA ($1/h, prefill 2,000 tok/s) and SB ($3/h, prefill 20,000 tok/s), one GPU each,
    tpot 10 ms, max_num_seqs 4, utilization 1.0, TTFT SLO 500 ms, 2 x 1 classes."""
    from llmplan.workload.classes import classify

    return PlanRequest(
        model=MODEL,
        stats=compute_stats(trace),
        slo=SLO(ttft_ms_p95=500.0, utilization_target=1.0),
        engine=EngineProfile(),
        options=PlanOptions(
            tensor_parallel_choices=(1,),
            dtype_choices=("bf16",),
            max_num_seqs_choices=(4,),
            max_model_len=8192,
            perf_backend="fake_shape",
        ),
        gpus=SHAPE_GPUS,
        prices=(ROW_SA, ROW_SB),
        classes=classify(trace, input_bins=2, output_bins=1) if classes else (),
    )


def fleet(result: PlanResult) -> dict[str, int]:
    return {item.price_row.instance: item.instances for item in result.fleet}


# --- 8.1 K=1 reproduces every M4 acceptance result byte for byte -------------------------

M4_CASES: dict[str, tuple[Mapping[tuple[str, int], FakePerf], Callable[[], PlanRequest], float]] = {
    "10.1": (A_B, lambda: request(34), 504.0),
    "10.2": (A_B, lambda: request(34, homogeneous=True), 576.0),
    "10.3": (
        A_B,
        lambda: request(30, (ROW_A, ROW_B.model_copy(update={"price_usd_per_hour": 20.0})), seed=7),
        480.0,
    ),
    "10.4": (
        {A1: FakePerf(rps=3, tokens_per_s=300), B1: FakePerf(rps=4)},
        lambda: request(6, demand_tps=900),
        144.0,
    ),
    "10.5": (
        {A1: FakePerf(rps=3, ttft_ms_p95=100), B1: FakePerf(rps=4, ttft_ms_p95=800)},
        lambda: request(34, slo=SLO(ttft_ms_p95=500, utilization_target=1.0)),
        576.0,
    ),
    "10.7": (
        {
            ("fake-b", 1): FakePerf(rps=4),
            ("fake-b", 2): FakePerf(rps=9),
            ("fake-b", 4): FakePerf(rps=20),
        },
        lambda: request(40, (ROW_B,), tensor_parallel_choices=(1, 2, 4)),
        456.0,
    ),
    "10.8": (A_B, lambda: request(34, solver="cp_sat"), 504.0),
    "10.11": (A_B, lambda: request(34, dtype_choices=("fp8",)), 504.0),
}


@pytest.mark.parametrize("case", list(M4_CASES))
def test_8_1_single_class_reproduces_m4(case: str, fake_perf: UseFakePerf) -> None:
    capacities, make_request, cost = M4_CASES[case]
    fake_perf(capacities)
    req = make_request()
    legacy = plan(req)
    one = plan(with_single_class(req))
    assert legacy.classes == legacy.routing == legacy.class_binding == ()
    assert without_new_fields(one) == without_new_fields(legacy)
    assert one.cost_usd_per_day == legacy.cost_usd_per_day == cost
    assert len(one.classes) == 1
    assert one.class_binding == (legacy.binding,)
    weights = sum(rule.weight for rule in one.routing)
    assert weights == pytest.approx(1.0, abs=1e-9)


def test_8_1_single_class_infeasible_matches(fake_perf: UseFakePerf) -> None:
    fake_perf(A_B)
    req = request(34, slo=SLO(ttft_ms_p95=10, utilization_target=1.0))
    with pytest.raises(InfeasiblePlan) as legacy:
        plan(req)
    with pytest.raises(InfeasiblePlan) as one:
        plan(with_single_class(req))
    assert str(one.value) == str(legacy.value)
    assert "0 of" in str(one.value)
    assert "slo_ttft" in str(one.value)


def test_8_1_single_class_real_catalogs() -> None:
    from llmplan.workload.classes import classify

    trace = load_workload(FIXTURES / "workload_10.csv")
    req = PlanRequest(
        model=load_model("fixture:llama3-8b"),
        stats=compute_stats(trace),
        slo=SLO(),
        engine=EngineProfile(),
        options=PlanOptions(max_model_len=8192, perf_backend="roofline"),
        gpus=load_gpus(),
        prices=load_prices(),
    )
    (whole,) = classify(trace, input_bins=1, output_bins=1)
    assert whole == single_class(req.stats)
    legacy = plan(req)
    one = plan(req.model_copy(update={"classes": (whole,)}))
    assert len(one.classes) == 1
    assert without_new_fields(one) == without_new_fields(legacy)


def test_8_1_cli_k1_output_matches_pre_m7(tmp_path: Path) -> None:
    """`llmplan plan` and `llmplan simulate --kv-accounting full` reproduce the pre-M7 JSON
    (captured on main at f69ec54) except for the new fields, which hold their defaults."""
    args = [
        "plan",
        "--model",
        "fixture:llama3-8b",
        "--trace",
        TEN,
        "--max-model-len",
        "8192",
        "--ttft-p95-ms",
        "500",
        "--gpus",
        "l4-24gb,l40s-48gb",
        "--tp",
        "1",
        "--dtypes",
        "fp8",
        "--max-num-seqs",
        "64,128",
        "--format",
        "json",
    ]
    golden_plan = (FIXTURES / "m7" / "plan_workload10_pre_m7.json").read_text(encoding="utf-8")
    for extra in ([], ["--classes", "1"]):
        result = runner.invoke(app, [*args, *extra])
        assert result.exit_code == 0, result.stderr
        doc = json.loads(result.stdout)
        assert doc["classes"] == doc["routing"] == doc["class_binding"] == []
        assert json.dumps(strip_new_keys(doc, NEW_PLAN_FIELDS), indent=2) + "\n" == golden_plan
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(golden_plan, encoding="utf-8")
    sim_args = ["simulate", "--plan", str(plan_file), "--trace", FIFTY, "--format", "json"]
    sim = runner.invoke(app, [*sim_args, "--kv-accounting", "full"])
    assert sim.exit_code == 0, sim.stderr
    doc = json.loads(sim.stdout)
    assert doc["classes"] == []
    assert doc["options"].pop("kv_accounting") == "full"
    golden_sim = (FIXTURES / "m7" / "simulate_csv50_pre_m7.json").read_text(encoding="utf-8")
    assert json.dumps(strip_new_keys(doc, ("classes",)), indent=2) + "\n" == golden_sim


# --- 8.2 A cheap GPU that meets the short-class SLO only serves the short class ----------


def test_8_2_cheap_gpu_serves_the_short_class_only() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    short, long = req.classes
    assert (short.input_lo, short.input_hi, long.input_lo, long.input_hi) == (1, 1050, 1051, 2000)
    assert (short.share, long.share) == (0.5, 0.5)
    assert (short.peak_rps, long.peak_rps) == (5.0, 5.0)
    assert (short.peak_output_tokens_per_s, long.peak_output_tokens_per_s) == (50.0, 325.0)

    result = plan(req)
    assert fleet(result) == {"sa-1x": 1, "sb-1x": 1}
    assert result.cost_usd_per_day == 96.0
    assert result.baseline is not None
    assert fleet(result.baseline) == {"sb-1x": 2}
    assert result.baseline.cost_usd_per_day == 144.0
    assert result.baseline_saving_pct == pytest.approx(100 / 3)
    assert result.class_binding == ("requests", "requests")
    assert result.binding == "requests"
    (cheap,) = [c for c in result.candidates if c.price_row.instance == "sa-1x"]
    assert cheap.status == "eligible"
    assert "class" in cheap.reason

    weights = {(r.class_index, r.candidate.price_row.instance): r.weight for r in result.routing}
    assert weights == {(0, "sa-1x"): pytest.approx(1.0), (1, "sb-1x"): pytest.approx(1.0)}
    for k in (0, 1):
        total = sum(r.weight for r in result.routing if r.class_index == k)
        assert total == pytest.approx(1.0, abs=1e-9)

    single = plan(two_class_request(trace, classes=False))
    assert fleet(single) == {"sb-1x": 2}
    assert single.cost_usd_per_day == 144.0


# --- 8.3 Exactness against brute force (M7_DESIGN.md section 6.1) ------------------------


def test_8_3_exactness_against_brute_force(fake_class_perf: UseFakeClassPerf) -> None:
    from tests.brute_force import brute_force_optimum, random_instance

    start = time.perf_counter()
    infeasible = 0
    for seed in range(50):
        instance = random_instance(seed)
        fake_class_perf(instance.capacities)
        expected = brute_force_optimum(instance)
        if expected is None:
            infeasible += 1
            with pytest.raises(InfeasiblePlan):
                plan(instance.request())
            continue
        result = plan(instance.request())
        assert result.cost_usd_per_day == pytest.approx(expected, rel=0, abs=1e-6), seed
    elapsed = time.perf_counter() - start
    print(f"8.3: 50 seeds ({infeasible} infeasible) checked in {elapsed:.2f} s")
    assert elapsed < 10


# --- 8.4 Class-weighted routing proves the plan; class-blind routing does not ------------


def test_8_4_class_weighted_simulation() -> None:
    from llmplan.simulate import SimOptions, replay

    trace = two_class_workload()
    req = two_class_request(trace)
    result = plan(req)
    weighted = replay(result, trace, slo=req.slo, options=SimOptions(routing="class_weighted"))
    short, long = weighted.classes
    assert (short.n_requests, long.n_requests) == (300, 300)
    assert short.ttft_violation_pct == long.ttft_violation_pct == 0.0
    assert (short.ttft_ms_p95, long.ttft_ms_p95) == (50.0, 100.0)
    assert weighted.summary.max_queue_depth == 0
    blind = replay(result, trace, slo=req.slo, options=SimOptions(routing="least_outstanding"))
    assert blind.classes[1].ttft_violation_pct > 0


# --- 8.5 Determinism of JSON output ------------------------------------------------------


@pytest.mark.slow  # M8 9b: multi-second; runs in the `ui` CI job's slow step
def test_8_5_determinism() -> None:
    from llmplan import render
    from llmplan.simulate import SimOptions, replay

    trace = two_class_workload()
    req = two_class_request(trace)
    first, second = plan(req), plan(req)
    assert render.get("json").plan(req, first) == render.get("json").plan(req, second)
    options = SimOptions(routing="class_weighted")
    assert (
        replay(first, trace, slo=req.slo, options=options).model_dump_json()
        == replay(second, trace, slo=req.slo, options=options).model_dump_json()
    )
    args = ["plan", "--model", "fixture:llama3-8b", "--trace", FIFTY, "--max-model-len", "8192"]
    args += ["--classes", "2x2", "--perf-backend", "roofline", "--format", "json"]
    outputs = [runner.invoke(app, args) for _ in range(2)]
    assert [o.exit_code for o in outputs] == [0, 0], outputs[0].stderr
    assert outputs[0].stdout == outputs[1].stdout
    assert len(json.loads(outputs[0].stdout)["classes"]) == 4


# --- 8b.1 Incremental KV accounting: concurrency rises to the planner's effective_batch --


def test_8b_1_incremental_kv_reaches_effective_batch() -> None:
    from llmplan.simulate import SimOptions, replay

    trace = workload([0.05 * i for i in range(6000)], 2000, 1000)  # 20 req/s for 300 s
    req = PlanRequest(
        model=load_model("fixture:llama3-8b"),
        stats=compute_stats(trace),
        slo=SLO(),
        engine=EngineProfile(),
        options=PlanOptions(
            gpu_ids=("h100-sxm-80gb",),
            providers=("lambda",),
            tensor_parallel_choices=(1,),
            dtype_choices=("bf16",),
            max_num_seqs_choices=(256,),
            max_model_len=8192,
            perf_backend="roofline",
        ),
        gpus=load_gpus(),
        prices=load_prices(),
    )
    result = plan(req)
    first = result.replicas[0]
    one = result.model_copy(update={"replicas": (first.model_copy(update={"count": 1}),)})
    assert first.candidate.fit is not None
    assert first.candidate.perf is not None
    assert first.candidate.fit.kv_token_capacity == 448_308
    batch = first.candidate.perf.effective_batch
    assert batch == 179  # 448,308 // (2000 + 1000 / 2)

    def steady_concurrency(kv_accounting: str) -> float:
        timeline = replay(one, trace, options=SimOptions(kv_accounting=kv_accounting))
        steady = timeline.windows[1:5]  # 60 to 300 s: the queue never empties
        return float(np.mean([w.replicas[0].utilization for w in steady])) * batch

    incremental = steady_concurrency("incremental")
    assert incremental == pytest.approx(batch, rel=0.15)
    assert incremental == pytest.approx(179.0)  # 179 x 2,500 = 447,500 <= 448,308
    full = steady_concurrency("full")
    assert full == pytest.approx(149.0)  # 448,308 // 3,000; 149 / 179 = 0.83, outside 15%


# --- 8b.2 llmplan simulate --requests-csv --------------------------------------------------


def test_8b_2_requests_csv(tmp_path: Path) -> None:
    from llmplan.simulate.events import REQUEST_COLUMNS

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        (FIXTURES / "m7" / "plan_workload10_pre_m7.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = tmp_path / "requests.csv"
    result = runner.invoke(
        app, ["simulate", "--plan", str(plan_file), "--trace", FIFTY, "--requests-csv", str(out)]
    )
    assert result.exit_code == 0, result.stderr
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == REQUEST_COLUMNS
    assert len(rows) == 1 + 50
    trace = load_workload(FIFTY)
    assert [float(r[0]) for r in rows[1:]] == trace.frame["arrival_s"].tolist()
