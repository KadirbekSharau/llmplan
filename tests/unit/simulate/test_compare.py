from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmplan import render
from llmplan.errors import InfeasiblePlan
from llmplan.planner import plan
from llmplan.planner.result import load_plan_json
from llmplan.render.plan_text import baseline_saving, class_comparison_sentence
from llmplan.simulate import compare
from llmplan.simulate.compare import ClassComparison, compare_single_class
from tests.acceptance.test_m7 import two_class_request, two_class_workload
from tests.fake_planner import SHAPE_GPUS


def sentence(cost: float, single: float | None, violations: float | None = None) -> str:
    return class_comparison_sentence(
        ClassComparison(
            class_cost_usd_per_day=cost,
            single_class_cost_usd_per_day=single,
            single_class_ttft_violation_pct=violations,
        )
    )


def test_sentences() -> None:
    assert sentence(83.76, 44.66, 20.2) == (
        "Sized for the mean request this fleet would cost $44.66/day, but the replay shows it "
        "would miss the latency target for 20.2% of requests. The class-sized plan costs "
        "$83.76/day."
    )
    assert "for under 0.1% of requests" in sentence(83.76, 44.66, 0.01)
    under = "but it would under-provision the long-request class."
    assert under in sentence(83.76, 44.66, 0.0)
    assert under in sentence(83.76, 44.66, None)
    assert sentence(96.0, 144.0) == (
        "Request-size routing saves $48.00/day (33.3%) versus sizing every replica for the "
        "mean request."
    )
    assert sentence(44.66, 44.664) == (
        "Request-size routing does not change the fleet for this traffic."
    )
    assert sentence(83.76, None) == (
        "No fleet sized for the mean request meets the target; the class-sized plan costs "
        "$83.76/day."
    )


def test_baseline_saving_is_never_negative() -> None:
    assert baseline_saving(12.5) == "saving 12.5%"
    assert baseline_saving(0.0) == "saving 0.0%"
    assert baseline_saving(-3.0) == "baseline 3.0% cheaper"


def test_compare_without_classes_and_without_a_trace() -> None:
    trace = two_class_workload()
    single_req = two_class_request(trace, classes=False)
    assert compare_single_class(single_req, plan(single_req), trace) is None
    req = two_class_request(trace)
    result = plan(req)
    unreplayed = compare_single_class(req, result)
    assert unreplayed == ClassComparison(
        class_cost_usd_per_day=96.0,
        single_class_cost_usd_per_day=144.0,
        single_class_ttft_violation_pct=None,
    )
    replayed = compare_single_class(req, result, trace, gpus=SHAPE_GPUS)
    assert replayed is not None
    assert replayed.single_class_ttft_violation_pct == 0.0  # 2 x SB serve it within 500 ms


def test_infeasible_single_class_plan_is_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    result = plan(req)

    def infeasible(*_: object, **__: object) -> None:
        raise InfeasiblePlan("no feasible fleet: test", reason="test")

    monkeypatch.setattr(compare, "plan", infeasible)
    assert compare_single_class(req, result, trace) == ClassComparison(
        class_cost_usd_per_day=96.0,
        single_class_cost_usd_per_day=None,
        single_class_ttft_violation_pct=None,
    )


def test_plan_json_carries_the_comparison_and_still_loads(tmp_path: Path) -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    result = plan(req)
    comparison = compare_single_class(req, result, trace, gpus=SHAPE_GPUS)
    text = render.get("json").plan(req, result, comparison)
    assert json.loads(text)["class_comparison"]["single_class_cost_usd_per_day"] == 144.0
    path = tmp_path / "plan.json"
    path.write_text(text, encoding="utf-8")
    loaded, slo = load_plan_json(path)
    assert loaded.cost_usd_per_day == 96.0
    assert slo == req.slo
    assert "class_comparison" not in render.get("json").plan(req, result)
