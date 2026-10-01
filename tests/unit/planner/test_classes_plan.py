"""The M7 class model: formulation, CP-SAT scaling, routing LP, per-class verdicts.

Most tests use the two-class instance of acceptance test 8.2 (M7_NOTES.md): row SA serves
only the short class, SB both.
"""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

from llmplan.errors import InfeasiblePlan, PerfError
from llmplan.planner import SLO, plan
from llmplan.planner import classes as planner_classes
from llmplan.planner.candidates import Column, columns, evaluate_candidates, rows_in_scope
from llmplan.planner.classes import ClassEval, evaluate_classes, reconcile
from llmplan.planner.explain import relax
from llmplan.planner.model import build_model
from llmplan.planner.request import PlanRequest
from llmplan.planner.result import CandidateEval
from llmplan.planner.solve import balance, solve
from llmplan.workload import Workload
from tests.acceptance.test_m7 import two_class_request, two_class_workload
from tests.fake_planner import ROW_SA, fake_row

DEMANDS = ((5.0, 50.0), (5.0, 325.0))


def class_columns(req: PlanRequest) -> tuple[Column, ...]:
    candidates = evaluate_candidates(req, rows_in_scope(req.prices, req.options), req.gpus)
    evals = evaluate_classes(req, candidates, req.gpus)
    reconciled = [reconcile(c, e, 1.0) for c, e in zip(candidates, evals, strict=True)]
    per_class = [
        None if e is None else tuple(x.perf if x.status == "eligible" else None for x in e)
        for e in evals
    ]
    return columns(reconciled, req.slo, per_class)


@pytest.fixture
def instance() -> tuple[Workload, PlanRequest, tuple[Column, ...]]:
    trace = two_class_workload()
    req = two_class_request(trace)
    return trace, req, class_columns(req)


def _build(cols: tuple[Column, ...], **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "demand_rps": 0.0,
        "demand_tps": 0.0,
        "homogeneous": False,
        "max_instances_per_row": 6,
        "class_demands": DEMANDS,
        **kwargs,
    }
    return build_model(cols, **options)


def test_columns_carry_per_class_capacity(instance: Any) -> None:
    _, _, cols = instance
    sa, sb = cols
    assert sa.class_rps == pytest.approx((4 / 0.15, 0.0))
    assert sa.class_tps == pytest.approx((400.0, 0.0))
    assert sb.class_rps == pytest.approx((4 / 0.105, 4 / 0.75))
    assert sa.rates() == tuple(zip(sa.class_rps, sa.class_tps, strict=True))


def test_class_model_shape(instance: Any) -> None:
    _, _, cols = instance
    model = _build(cols)
    names = sorted(v.name for v in model.model.variables())
    assert names == ["m_0", "m_1", "n_0", "n_1", "x_0_0", "x_1_0", "x_1_1"]
    assert model.model.get_num_linear_constraints() == 2 + 2 + 4  # gpus, share, demands
    assert not any(v.integer for v in model.model.variables() if v.name.startswith("x_"))
    assert len(model.demands) == 2
    assert model.request_demand is model.demands[0][0]
    relaxed = _build(cols, relax=True)
    assert not any(v.integer for v in relaxed.model.variables())


def test_integer_scaled_class_model(instance: Any) -> None:
    _, _, cols = instance
    scaled = _build(cols, integer_scaling=True)
    x = {v.name: v for v in scaled.model.variables() if v.name.startswith("x_")}
    assert all(v.integer for v in x.values())
    assert x["x_0_0"].upper_bound == 6000  # 6 replicas, in milli-replicas
    short_requests, long_tokens = scaled.demands[0][0], scaled.demands[1][1]
    assert {t.variable.name: t.coefficient for t in short_requests.terms()} == {
        "x_0_0": 26666,  # floor(26.6667 x 1000) milli-req/s per milli-replica
        "x_1_0": 38095,
    }
    assert short_requests.lower_bound == 5_000_000  # 5 req/s in micro-units, rounded up
    assert long_tokens.lower_bound == 325_000_000
    highs = solve(_build(cols), "highs", time_limit_s=10, seed=0)
    cp_sat = solve(scaled, "cp_sat", time_limit_s=10, seed=0)
    assert highs.instances == cp_sat.instances == (1, 1)


def test_cp_sat_agrees_with_highs_on_classes() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    cp_sat = plan(
        req.model_copy(update={"options": req.options.model_copy(update={"solver": "cp_sat"})})
    )
    assert cp_sat.cost_usd_per_day == plan(req).cost_usd_per_day == 96.0


def test_routing_lp_balances_headroom(instance: Any) -> None:
    _, _, cols = instance
    allocation = balance(cols, (1, 1), DEMANDS)
    assert allocation.theta == pytest.approx(16 / 15)  # long: 5.333 x 1 >= 5 theta
    assert allocation.x[0] == pytest.approx((1.0, 0.0))  # SA: all its time to short
    assert allocation.x[1] == pytest.approx((0.0, 1.0), abs=1e-6)  # SB: all to long


def test_relaxation_reports_each_class(instance: Any) -> None:
    _, _, cols = instance
    relaxed = relax(cols, DEMANDS, 6)
    assert relaxed.classes == ((True, False), (True, False))
    assert relaxed.requests_tight and not relaxed.tokens_tight
    assert relaxed.class_shadow_prices[0][0] == pytest.approx(24 / (4 / 0.15))  # SA per req/s


def test_plan_with_classes_explains_the_fleet(instance: Any) -> None:
    _, req, _ = instance
    result = plan(req)
    assert result.capacity_rps == pytest.approx(4 / 0.15 + 4 / 0.75)
    assert result.capacity_output_tokens_per_s == pytest.approx(800.0)
    assert result.baseline is not None
    # 2 SB replicas need 1.06875 each per unit of demand: theta = 2 / 1.06875 on 10 req/s.
    assert result.baseline.capacity_rps == pytest.approx(10 * 2 / 1.06875)


def test_unserved_class_is_infeasible() -> None:
    trace = two_class_workload()
    req = two_class_request(trace).model_copy(
        update={"slo": SLO(ttft_ms_p95=50.0, utilization_target=1.0)}
    )
    with pytest.raises(
        InfeasiblePlan,
        match=r"class 1 \(inputs 1051\.\.2000, outputs 0\.\.65\): 0 of 2 candidates "
        "eligible: 2 slo_ttft",
    ):
        plan(req)


def test_class_without_peak_demand_needs_no_candidate() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    short, long = req.classes
    idle = long.model_copy(update={"peak_rps": 0.0, "peak_output_tokens_per_s": 0.0})
    tight = req.model_copy(
        update={"classes": (short, idle), "slo": SLO(ttft_ms_p95=50.0, utilization_target=1.0)}
    )
    result = plan(tight)
    assert {item.price_row.instance: item.instances for item in result.fleet} == {"sa-1x": 1}


def test_class_estimates_are_cached_per_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = two_class_workload()
    twin = fake_row("sa-twin", "shape-a", 1, 1.5)
    req = two_class_request(trace).model_copy(update={"prices": (ROW_SA, twin)})
    calls: list[int] = []
    real = planner_classes.estimate

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner_classes, "estimate", counting)
    candidates = evaluate_candidates(req, req.prices, req.gpus)
    evals = evaluate_classes(req, candidates, req.gpus)
    assert len(calls) == 2  # two classes on one GPU and config, shared by both rows
    assert evals[0] == evals[1]


def test_class_without_estimate_is_no_perf(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = two_class_workload()
    req = two_class_request(trace)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise PerfError("no estimate for this class")

    candidates = evaluate_candidates(req, req.prices, req.gpus)
    monkeypatch.setattr(planner_classes, "estimate", broken)
    evals = evaluate_classes(req, candidates, req.gpus)
    assert evals[0] == (ClassEval(None, "no_perf", "no estimate for this class"),) * 2


def _candidate(req: PlanRequest, index: int) -> CandidateEval:
    return evaluate_candidates(req, req.prices, req.gpus)[index]


def test_reconcile_verdicts() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    sa, sb = evaluate_candidates(req, req.prices, req.gpus)
    sa_evals, sb_evals = evaluate_classes(req, (sa, sb), req.gpus)
    assert sa is not None and sa_evals is not None and sb_evals is not None
    # SA fails the whole workload's TTFT but serves the short class.
    assert sa.status == "slo_ttft"
    served = reconcile(sa, sa_evals, 1.0)
    assert served.status == "eligible"
    assert served.reason.startswith("eligible for class 0 only (whole workload: TTFT p95 1000 ms")
    assert served.usd_per_hour_per_rps == pytest.approx(1.0 / (4 / 0.9))  # whole-workload perf
    # SB passes everything: unchanged.
    assert reconcile(sb, sb_evals, 1.0) == sb
    # Whole-workload eligible but missing a class: the reason says which.
    partial = reconcile(sb, (sb_evals[0], sa_evals[1]), 1.0)
    assert partial.status == "eligible"
    assert partial.reason.endswith("; not eligible for class 1 (TTFT p95 1000 ms > SLO 500 ms)")
    # Whole-workload eligible, no class: rejected with every class's reason.
    none = reconcile(sb, (sa_evals[1], sa_evals[1]), 1.0)
    assert none.status == "slo_ttft"
    assert none.reason.startswith("not eligible for any request-size class (class 0: TTFT")
    assert none.usd_per_hour_per_rps is None
    # Rejected everywhere, or never estimated: unchanged.
    assert reconcile(sa, (sa_evals[1], sa_evals[1]), 1.0) == sa
    assert reconcile(sa, None, 1.0) == sa
    # Rejected without an estimate on the whole workload: the class's estimate is used.
    no_perf = sa.model_copy(update={"status": "no_perf", "perf": None})
    assert reconcile(no_perf, sa_evals, 1.0).perf == sa_evals[0].perf


def test_plan_request_checks_classes() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    short, long = req.classes
    fields = {name: getattr(req, name) for name in PlanRequest.model_fields}
    with pytest.raises(pydantic.ValidationError, match=r"indexed 0\.\.K-1"):
        PlanRequest.model_validate({**fields, "classes": (long, short)})
    with pytest.raises(pydantic.ValidationError, match="shares must sum to 1"):
        PlanRequest.model_validate({**fields, "classes": (short,)})


def test_routing_rules_carry_the_allocation(instance: Any) -> None:
    _, req, _ = instance
    result = plan(req)
    short, long = result.routing
    assert (short.class_index, short.candidate.price_row.instance) == (0, "sa-1x")
    assert short.replicas == pytest.approx(1.0)
    assert short.capacity_rps == pytest.approx(4 / 0.15)
    assert long.capacity_output_tokens_per_s == pytest.approx(400.0)
    assert result.baseline is not None
    assert {r.class_index for r in result.baseline.routing} == {0, 1}
    assert result.baseline.class_binding == ("requests", "requests")
    assert any("routing weights come from an LP" in a for a in result.assumptions)
    assert sum("binding from the LP relaxation" in a for a in result.assumptions) == 2


def test_routing_for_classes_without_demand() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    short, long = req.classes
    idle_short = short.model_copy(update={"peak_rps": 0.0, "peak_output_tokens_per_s": 0.0})
    # Only the long class has demand: one SB, and the idle short class is spread over the
    # replicas that can serve it (SB) by their capacity.
    result = plan(req.model_copy(update={"classes": (idle_short, long)}))
    assert [(r.class_index, r.candidate.price_row.instance) for r in result.routing] == [
        (0, "sb-1x"),
        (1, "sb-1x"),
    ]
    assert result.routing[0].weight == 1.0
    # With a 50 ms TTFT only SA serves short and nothing serves long; an idle long class
    # then gets no rules at all.
    idle_long = long.model_copy(update={"peak_rps": 0.0, "peak_output_tokens_per_s": 0.0})
    slo = SLO(ttft_ms_p95=50.0, utilization_target=1.0)
    tight = req.model_copy(update={"classes": (short, idle_long), "slo": slo})
    assert [r.class_index for r in plan(tight).routing] == [0]
