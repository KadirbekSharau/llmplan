from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from ortools.math_opt.python import mathopt

from llmplan.errors import InfeasiblePlan, SolverError
from llmplan.planner import solve as solve_module
from llmplan.planner.baseline import best_homogeneous, replica_equivalents
from llmplan.planner.candidates import Column, columns, evaluate_candidates, rows_in_scope
from llmplan.planner.model import build_model
from llmplan.planner.solve import available, check_backend, relaxation, solve
from tests.conftest import UseFakePerf
from tests.fake_planner import FakePerf, request


@pytest.fixture
def cols(fake_perf: UseFakePerf) -> tuple[Column, ...]:
    fake_perf({("fake-a", 1): FakePerf(rps=3.3333), ("fake-b", 1): FakePerf(rps=4.35)})
    req = request(34)
    return columns(
        evaluate_candidates(req, rows_in_scope(req.prices, req.options), req.gpus), req.slo
    )


@pytest.fixture
def fresh_availability() -> Iterator[None]:
    available.cache_clear()
    yield
    available.cache_clear()


def _build(cols: tuple[Column, ...], **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "demand_rps": 34.0,
        "demand_tps": 0.0,
        "homogeneous": False,
        "max_instances_per_row": 10,
        **kwargs,
    }
    return build_model(cols, **options)


def test_build_model_shape(cols: tuple[Column, ...]) -> None:
    plain = _build(cols)
    assert [row.instance for row in plain.rows] == ["a-1x", "b-8x"]
    assert plain.model.get_num_variables() == 4
    assert plain.model.get_num_linear_constraints() == 4
    assert all(v.integer for v in plain.model.variables())
    assert plain.replicas[1].upper_bound == 80  # 10 instances x 8 replicas
    homogeneous = _build(cols, homogeneous=True)
    assert homogeneous.model.get_num_variables() == 6
    assert homogeneous.model.get_num_linear_constraints() == 7
    relaxed = _build(cols, relax=True)
    assert not any(v.integer for v in relaxed.model.variables())


def test_integer_scaling_is_conservative(cols: tuple[Column, ...]) -> None:
    scaled = _build(cols, integer_scaling=True, demand_rps=34.0001)
    coefficients = {t.variable.name: t.coefficient for t in scaled.request_demand.terms()}
    assert coefficients == {"m_0": 3333, "m_1": 4350}  # floor(3.3333e3), 4.35e3 guarded
    assert scaled.request_demand.lower_bound == 34001  # ceil(34000.1)
    assert scaled.objective_scale == 100
    objective = {t.variable.name: t.coefficient for t in scaled.model.objective.linear_terms()}
    assert objective == {"n_0": 4800, "n_1": 45600}


def test_solve_backends_agree(cols: tuple[Column, ...]) -> None:
    highs = solve(_build(cols), "highs", time_limit_s=10, seed=0)
    cp_sat = solve(_build(cols, integer_scaling=True), "cp_sat", time_limit_s=10, seed=0)
    assert highs.instances == cp_sat.instances
    assert highs.status == "optimal"
    assert highs.best_bound_usd_per_day == pytest.approx(cp_sat.best_bound_usd_per_day, abs=0.01)
    if available("scip"):
        scip = solve(_build(cols), "scip", time_limit_s=10, seed=0)
        assert scip.instances == highs.instances


def test_relaxation_reports_tightness(cols: tuple[Column, ...]) -> None:
    relax = relaxation(_build(cols, relax=True))
    assert relax.requests_tight
    assert not relax.tokens_tight
    assert relax.requests_shadow_price > 0
    tps = 34 / 3.3333 * 1000  # the tokens A needs for exactly as many replicas as requests
    both = relaxation(_build(cols[:1], relax=True, demand_tps=tps, max_instances_per_row=99))
    assert (both.requests_tight, both.tokens_tight) == (True, True)


def test_unknown_and_unavailable_backends(
    monkeypatch: pytest.MonkeyPatch, fresh_availability: None
) -> None:
    with pytest.raises(SolverError, match="unknown solver backend 'glpk'"):
        check_backend("glpk")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("solver type is not registered")

    monkeypatch.setattr(solve_module.mathopt, "solve", broken)
    assert available("highs")  # built in, never probed
    with pytest.raises(SolverError, match="backend not available"):
        check_backend("gurobi")


def _fake_result(reason: Any, limit: Any = None, detail: str = "") -> Any:
    return SimpleNamespace(termination=SimpleNamespace(reason=reason, limit=limit, detail=detail))


@pytest.mark.parametrize(
    ("result", "error", "match"),
    [
        (_fake_result(mathopt.TerminationReason.INFEASIBLE), InfeasiblePlan, "infeasible"),
        (
            _fake_result(mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED),
            InfeasiblePlan,
            "infeasible",
        ),
        (
            _fake_result(mathopt.TerminationReason.NO_SOLUTION_FOUND, detail="time"),
            SolverError,
            "stopped without a feasible fleet: time",
        ),
        (_fake_result(mathopt.TerminationReason.NUMERICAL_ERROR), SolverError, "NUMERICAL_ERROR"),
        (
            _fake_result(mathopt.TerminationReason.FEASIBLE, mathopt.Limit.ITERATION),
            SolverError,
            "FEASIBLE",
        ),
    ],
)
def test_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    cols: tuple[Column, ...],
    result: Any,
    error: type[Exception],
    match: str,
) -> None:
    monkeypatch.setattr(solve_module.mathopt, "solve", lambda *a, **k: result)
    with pytest.raises(error, match=match):
        solve(_build(cols), "highs", time_limit_s=1, seed=0)


def test_time_limit_with_incumbent(
    monkeypatch: pytest.MonkeyPatch, cols: tuple[Column, ...]
) -> None:
    formulation = _build(cols)
    real = mathopt.solve(formulation.model, mathopt.SolverType.HIGHS)
    result = SimpleNamespace(
        termination=SimpleNamespace(
            reason=mathopt.TerminationReason.FEASIBLE,
            limit=mathopt.Limit.TIME,
            detail="",
            objective_bounds=SimpleNamespace(dual_bound=float("-inf")),
        ),
        variable_values=real.variable_values,
        solve_time=real.solve_time,
    )
    monkeypatch.setattr(solve_module.mathopt, "solve", lambda *a, **k: result)
    solution = solve(formulation, "highs", time_limit_s=1, seed=0)
    assert solution.status == "feasible_time_limit"
    assert solution.best_bound_usd_per_day is None


def test_solver_runtime_error_and_bad_relaxation(
    monkeypatch: pytest.MonkeyPatch, cols: tuple[Column, ...]
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("INTERNAL: boom")

    monkeypatch.setattr(solve_module.mathopt, "solve", broken)
    with pytest.raises(SolverError, match=r"highs failed: INTERNAL: boom .*fresh process"):
        solve(_build(cols), "highs", time_limit_s=1, seed=0)
    with pytest.raises(SolverError, match=r"cp_sat failed: INTERNAL: boom$"):
        solve(_build(cols, integer_scaling=True), "cp_sat", time_limit_s=1, seed=0)
    monkeypatch.setattr(
        solve_module.mathopt,
        "solve",
        lambda *a, **k: _fake_result(mathopt.TerminationReason.NUMERICAL_ERROR),
    )
    with pytest.raises(SolverError, match="LP relaxation terminated with NUMERICAL_ERROR"):
        relaxation(_build(cols, relax=True))


def test_baseline_enumeration(cols: tuple[Column, ...]) -> None:
    assert replica_equivalents([(3.0, 300.0)], [(0.0, 0.0)]) == 0.0
    assert replica_equivalents([(3.0, 300.0)], [(30.0, 900.0)]) == 10.0
    assert replica_equivalents([(3.0, 300.0), (2.0, 100.0)], [(3.0, 0.0), (1.0, 150.0)]) == 2.5
    assert replica_equivalents([(3.0, 300.0), (0.0, 0.0)], [(3.0, 0.0), (1.0, 0.0)]) is None
    exact = [dataclasses.replace(c, rps=float(r)) for c, r in zip(cols, (3, 4), strict=True)]
    best = best_homogeneous(exact, 34.0, 0.0, 1000)
    assert best is not None
    assert (best.column.candidate.price_row.instance, best.replicas, best.instances) == (
        "a-1x",
        12,
        12,
    )
    assert best.usd_per_day == 576.0
    tokens = best_homogeneous(exact, 1.0, 2500.0, 1000)
    assert tokens is not None
    assert tokens.replicas == 3  # 1000 output tokens/s per replica
    assert best_homogeneous(exact, 34.0, 0.0, 1) is None  # A needs 12, B needs 2
