from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import llmplan.planner as planner
from llmplan.errors import (
    CatalogError,
    InfeasiblePlan,
    SolverError,
    UnknownRegistryKey,
    ValidationError,
)
from llmplan.planner import solve as solve_module
from llmplan.planner.result import binding_label
from llmplan.planner.solve import Solution
from tests.conftest import UseFakePerf
from tests.fake_planner import ROW_A, ROW_B, FakePerf, fake_row, request

CAPACITY = {("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)}


def test_validation_errors() -> None:
    with pytest.raises(ValidationError, match="max_model_len 16384 exceeds"):
        planner.plan(request(1, max_model_len=16384))
    with pytest.raises(CatalogError, match="unknown gpu id 'h100'"):
        planner.plan(request(1, gpu_ids=("h100",)))
    with pytest.raises(CatalogError, match=r"price row 1 \(x-1x\): unknown gpu_id 'nope'"):
        planner.plan(request(1, (ROW_A, fake_row("x-1x", "nope", 1, 1.0))))
    with pytest.raises(UnknownRegistryKey, match="unknown perf backend 'nope'"):
        planner.plan(request(1, perf_backend="nope"))


def test_unavailable_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solve_module, "available", lambda backend: False)
    with pytest.raises(SolverError, match="backend not available"):
        planner.plan(request(1, solver="gurobi"))


def test_no_rows_in_scope() -> None:
    with pytest.raises(InfeasiblePlan, match="no price rows match") as info:
        planner.plan(request(1, providers=("aws",)))
    assert info.value.reason.startswith("no price rows match")


def test_eligible_but_demand_too_large(fake_perf: UseFakePerf) -> None:
    fake_perf(CAPACITY)
    with pytest.raises(InfeasiblePlan, match="2 of 2 candidates eligible, but no fleet"):
        planner.plan(request(100, max_instances_per_row=1))
    with pytest.raises(InfeasiblePlan, match="on a single price row"):
        planner.plan(request(34, max_instances_per_row=1, homogeneous=True))


def test_no_baseline_when_no_single_row_suffices(fake_perf: UseFakePerf) -> None:
    fake_perf(CAPACITY)
    result = planner.plan(request(34, max_instances_per_row=1))
    assert result.cost_usd_per_day == 504.0
    assert result.baseline is None
    assert result.baseline_saving_pct is None


def test_candidates_sorted_eligible_first(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-b", 1): FakePerf(rps=4), ("fake-b", 2): FakePerf(rps=9)})
    result = planner.plan(request(10, tensor_parallel_choices=(1, 2)))
    statuses = [c.status for c in result.candidates]
    assert statuses == ["eligible", "eligible", "no_perf", "tp_gt_gpus"]
    costs = [c.usd_per_hour_per_rps for c in result.candidates[:2]]
    assert costs == sorted(costs)  # type: ignore[type-var]
    assert "dominance pruning removed 0 of 2 eligible candidates" in result.assumptions


def test_baseline_result_shape(fake_perf: UseFakePerf) -> None:
    fake_perf(CAPACITY)
    baseline = planner.plan(request(34)).baseline
    assert baseline is not None
    assert baseline.solver.backend == "enumeration"
    assert baseline.candidates == ()
    assert baseline.binding == "requests"
    assert baseline.capacity_rps == 36.0
    assert baseline.replicas[0].count == 12


def test_binding_labels() -> None:
    assert [binding_label(r, t) for r, t in ((1, 1), (1, 0), (0, 1), (0, 0))] == [
        "both",
        "requests",
        "tokens",
        "none",
    ]


def test_time_limit_is_reported(monkeypatch: pytest.MonkeyPatch, fake_perf: UseFakePerf) -> None:
    fake_perf(CAPACITY)
    real = planner.solve_model

    def at_time_limit(*args: Any, **kwargs: Any) -> Solution:
        return dataclasses.replace(real(*args, **kwargs), status="feasible_time_limit")

    monkeypatch.setattr(planner, "solve_model", at_time_limit)
    result = planner.plan(request(34, time_limit_s=0.5))
    assert result.solver.status == "feasible_time_limit"
    assert any("time limit 0.5 s reached" in a for a in result.assumptions)


def test_fleet_below_demand_is_a_solver_error(
    monkeypatch: pytest.MonkeyPatch, fake_perf: UseFakePerf
) -> None:
    fake_perf(CAPACITY)
    real = planner.solve_model

    def short(*args: Any, **kwargs: Any) -> Solution:
        solution = real(*args, **kwargs)
        return dataclasses.replace(solution, replicas=(0, 1))

    monkeypatch.setattr(planner, "solve_model", short)
    with pytest.raises(SolverError, match="below demand"):
        planner.plan(request(34, (ROW_A, ROW_B)))
