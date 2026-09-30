"""Solving a `Formulation`: backend selection, deterministic parameters, status handling.

Every solve is single-threaded with an explicit seed, time limit, and zero relative gap, so
the same inputs give the same answer (M4_DESIGN.md section 6). `highs` and `cp_sat` ship
with OR-Tools; `scip` and `gurobi` are used only when MathOpt can run them. The LP
relaxation that explains the binding constraint is always solved with GLOP.
"""

from __future__ import annotations

import datetime
import functools
import math
from dataclasses import dataclass
from typing import Any, Literal

from ortools.math_opt.python import mathopt
from ortools.math_opt.solvers import highs_pb2

from llmplan.errors import InfeasiblePlan, SolverError
from llmplan.planner.model import Formulation

SOLVERS: dict[str, Any] = {
    "highs": mathopt.SolverType.HIGHS,
    "cp_sat": mathopt.SolverType.CP_SAT,
    "scip": mathopt.SolverType.GSCIP,
    "gurobi": mathopt.SolverType.GUROBI,
}
OPTIONAL = frozenset({"scip", "gurobi"})  # used only when MathOpt can load them
INTEGER_ONLY = frozenset({"cp_sat"})  # backends that need the integer-scaled model
TIGHT_TOLERANCE = 1e-6  # relative slack below which a demand constraint counts as tight
LP_TIME_LIMIT_S = 10.0


@dataclass(frozen=True)
class Solution:
    """Integer values read back from a solve, in the formulation's row and column order.

    `best_bound_usd_per_day` is the solver's dual bound unscaled to USD per day (None when
    the solver reports none).
    """

    instances: tuple[int, ...]
    replicas: tuple[int, ...]
    status: Literal["optimal", "feasible_time_limit"]
    best_bound_usd_per_day: float | None
    solve_time_s: float
    n_variables: int
    n_constraints: int


@dataclass(frozen=True)
class Relaxation:
    """Tightness and shadow prices (USD/day per unit) of the two demand constraints in the
    LP relaxation."""

    requests_tight: bool
    tokens_tight: bool
    requests_shadow_price: float
    tokens_shadow_price: float


def parameters(backend: str, time_limit_s: float, seed: int) -> Any:
    """MathOpt parameters: one thread, `seed`, `time_limit_s`, zero relative gap, quiet."""
    common: dict[str, Any] = {
        "time_limit": datetime.timedelta(seconds=time_limit_s),
        "random_seed": seed,
        "relative_gap_tolerance": 0.0,
        "enable_output": False,
    }
    if backend == "highs":
        # MathOpt rejects the generic `threads` for HiGHS; it is a HiGHS option instead.
        common["highs"] = highs_pb2.HighsOptionsProto(int_options={"threads": 1})
    else:
        common["threads"] = 1
    return mathopt.SolveParameters(**common)


@functools.cache
def available(backend: str) -> bool:
    """Whether MathOpt can run `backend` here. `highs` and `cp_sat` are built into OR-Tools;
    an optional backend is probed once per process by solving an empty model with the
    same single-thread parameters (HiGHS fixes its thread count process-wide on first use,
    so it is never probed with other settings)."""
    if backend not in OPTIONAL:
        return True
    try:
        mathopt.solve(
            mathopt.Model(name="probe"), SOLVERS[backend], params=parameters(backend, 1.0, 0)
        )
    except RuntimeError:
        return False
    return True


def check_backend(backend: str) -> None:
    """Raise `SolverError` unless `backend` is known and available."""
    if backend not in SOLVERS:
        raise SolverError(f"unknown solver backend {backend!r}; known: {', '.join(SOLVERS)}")
    if not available(backend):
        raise SolverError(f"solver backend {backend!r} is not available (backend not available)")


def _status(result: Any, backend: str) -> Literal["optimal", "feasible_time_limit"]:
    reason = result.termination.reason
    kinds = mathopt.TerminationReason
    if reason == kinds.OPTIMAL:
        return "optimal"
    if reason == kinds.FEASIBLE and result.termination.limit == mathopt.Limit.TIME:
        return "feasible_time_limit"
    if reason in (kinds.INFEASIBLE, kinds.INFEASIBLE_OR_UNBOUNDED):
        # Every variable is bounded, so "infeasible or unbounded" can only be infeasible.
        raise InfeasiblePlan("infeasible", reason="")
    detail = f": {result.termination.detail}" if result.termination.detail else ""
    if reason == kinds.NO_SOLUTION_FOUND:
        raise SolverError(f"{backend} stopped without a feasible fleet{detail}")
    raise SolverError(f"{backend} terminated with {reason.name}{detail}")


def solve(formulation: Formulation, backend: str, *, time_limit_s: float, seed: int) -> Solution:
    """Solve `formulation` with `backend` and return its integer solution.

    Raises `InfeasiblePlan` (with an empty reason, filled in by the caller) when the model
    is infeasible, and `SolverError` for an unavailable backend, a time limit without an
    incumbent, or any other termination.
    """
    check_backend(backend)
    try:
        result = mathopt.solve(
            formulation.model, SOLVERS[backend], params=parameters(backend, time_limit_s, seed)
        )
    except RuntimeError as exc:
        raise SolverError(f"{backend} failed: {exc}") from None
    status = _status(result, backend)
    bound = result.termination.objective_bounds.dual_bound
    model = formulation.model
    return Solution(
        instances=tuple(round(result.variable_values(v)) for v in formulation.instances),
        replicas=tuple(round(result.variable_values(v)) for v in formulation.replicas),
        status=status,
        best_bound_usd_per_day=bound / formulation.objective_scale
        if math.isfinite(bound)
        else None,
        solve_time_s=result.solve_time().total_seconds(),
        n_variables=model.get_num_variables(),
        n_constraints=model.get_num_linear_constraints(),
    )


def _tight(constraint: Any, activity: float) -> bool:
    bound = float(constraint.lower_bound)
    return activity - bound <= TIGHT_TOLERANCE * max(1.0, abs(bound))


def relaxation(formulation: Formulation) -> Relaxation:
    """Solve the LP relaxation (built with `relax=True`) with GLOP and report which demand
    constraints are tight and their shadow prices. Raises `SolverError` if GLOP fails."""
    result = mathopt.solve(
        formulation.model,
        mathopt.SolverType.GLOP,
        params=parameters("glop", LP_TIME_LIMIT_S, 0),
    )
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise SolverError(f"LP relaxation terminated with {result.termination.reason.name}")
    values = result.variable_values()
    requests, tokens = formulation.request_demand, formulation.token_demand

    def activity(constraint: Any) -> float:
        return math.fsum(float(t.coefficient * values[t.variable]) for t in constraint.terms())

    duals = result.dual_values([requests, tokens])
    return Relaxation(
        requests_tight=_tight(requests, activity(requests)),
        tokens_tight=_tight(tokens, activity(tokens)),
        requests_shadow_price=duals[0] / formulation.objective_scale,
        tokens_shadow_price=duals[1] / formulation.objective_scale,
    )
