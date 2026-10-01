"""MathOpt formulation of the fleet MILP (M4_DESIGN.md section 5, M7_DESIGN.md section 4).

    minimize    sum_p 24 * price_p * n_p
    subject to  sum_{r in R_p} tp_r * m_r <= g_p * n_p      for each price row p
                sum_r cap_r * m_r >= D_req                   (request demand)
                sum_r tok_r * m_r >= D_tok                   (token demand)
                [homogeneous] n_p <= max_instances * y_p;  sum_p y_p <= 1

`n_p` are instances of row p, `m_r` replicas of column r, `y_p` binary row selectors.
With request-size classes k (M7), continuous `x_{r,k} >= 0` replica-equivalents of r
devoted to k replace the two demand rows:

                sum_k x_{r,k} <= m_r                         for each r   (time share)
                sum_r cap_{r,k} * x_{r,k} >= D_k             for each k   (class requests)
                sum_r tok_{r,k} * x_{r,k} >= T_k             for each k   (class tokens)

and `x_{r,k}` exists only where r meets class k's SLO. With one class this is the M4 model
(`x_r = m_r` is optimal), so one class is built exactly as M4.

With `integer_scaling` every coefficient is an integer for CP-SAT: cents per day for the
objective, milli-units for capacities, and allocations in milli-replicas (`x = X / 1000`),
rounded so a scaled solution is always feasible in the unscaled model (capacities down,
demand up). With `relax` the variables are continuous: the LP relaxation used to explain
which demand constraint binds. `build_balance` is the post-solve LP that spreads a fixed
fleet over the classes with the most uniform headroom (routing weights).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ortools.math_opt.python import mathopt

from llmplan.catalog.hardware import PriceRow
from llmplan.planner.candidates import Column

HOURS_PER_DAY = 24
CENTS_PER_USD = 100
MILLI = 1000
ROUNDING_GUARD = 1e-9  # absorbs float noise such as 4.35 * 1000 = 4349.999...
THETA_MAX = 1e4  # headroom cap: keeps GLOP well scaled; only tiny demands reach it


@dataclass(frozen=True)
class Formulation:
    """A built MathOpt model plus the handles needed to read a solution back.

    `rows[i]` owns `instances[i]`; `columns[j]` owns `replicas[j]`. The objective is in
    USD per day times `objective_scale` (1, or 100 when integer-scaled). `demands[k]` are
    the (request, token) demand constraints of class k (one pair without classes);
    `request_demand` and `token_demand` are the first pair.
    """

    model: Any  # mathopt.Model; ortools ships no type information
    rows: tuple[PriceRow, ...]
    columns: tuple[Column, ...]
    instances: tuple[Any, ...]
    replicas: tuple[Any, ...]
    request_demand: Any
    token_demand: Any
    objective_scale: int
    demands: tuple[tuple[Any, Any], ...] = ()


def _scaled_capacity(value: float, integer_scaling: bool) -> float:
    return math.floor(value * MILLI + ROUNDING_GUARD) if integer_scaling else value


def _scaled_demand(value: float, integer_scaling: bool, scale: int = MILLI) -> float:
    return math.ceil(value * scale - ROUNDING_GUARD) if integer_scaling else value


def _class_demands(
    model: Any,
    cols: Sequence[Column],
    replicas: Sequence[Any],
    demands: Sequence[tuple[float, float]],
    integer_scaling: bool,
) -> tuple[tuple[Any, Any], ...]:
    """The M7 allocation variables `x_<j>_<k>` (continuous, or integer milli-replicas when
    integer-scaled), the time-share rows, and the per-class demand rows."""
    share = MILLI if integer_scaling else 1
    add = model.add_integer_variable if integer_scaling else model.add_variable
    terms: list[list[tuple[float, float, Any]]] = [[] for _ in demands]
    for j, (col, m) in enumerate(zip(cols, replicas, strict=True)):
        mine = []
        for k, (rps, tps) in enumerate(col.rates()):
            if rps > 0:
                x = add(lb=0, ub=m.upper_bound * share, name=f"x_{j}_{k}")
                terms[k].append((rps, tps, x))
                mine.append(x)
        if mine:
            model.add_linear_constraint(mathopt.fast_sum(mine) <= share * m, name=f"share_{j}")

    def row(k: int, kind: str, demand: float, which: int) -> Any:
        activity = mathopt.fast_sum(
            _scaled_capacity(term[which], integer_scaling) * term[2] for term in terms[k]
        )
        return model.add_linear_constraint(
            activity >= _scaled_demand(demand, integer_scaling, MILLI * share),
            name=f"demand_{kind}_{k}",
        )

    return tuple(
        (row(k, "requests", rps, 0), row(k, "tokens", tps, 1))
        for k, (rps, tps) in enumerate(demands)
    )


def build_model(
    cols: Sequence[Column],
    *,
    demand_rps: float,
    demand_tps: float,
    homogeneous: bool,
    max_instances_per_row: int,
    integer_scaling: bool = False,
    relax: bool = False,
    class_demands: Sequence[tuple[float, float]] = (),
) -> Formulation:
    """Build the section 5 model over the eligible, pruned `cols` (at least one).

    `demand_rps` / `demand_tps` is the one-class demand. With two or more
    `class_demands` (M7: (req/s, output tokens/s) per class, matching `Column.rates()`),
    the class model is built instead and the one-class demand is unused. Rows are the
    distinct price rows of `cols` in first-seen order. Variable and constraint names are
    deterministic (`n_<i>`, `m_<j>`, `x_<j>_<k>`, `y_<i>`, `gpus_<i>`, `share_<j>`,
    `demand_*`).
    """
    rows = tuple(dict.fromkeys(c.candidate.price_row for c in cols))
    index = {row: i for i, row in enumerate(rows)}
    model = mathopt.Model(name="llmplan_fleet")

    def var(lb: float, ub: float, name: str) -> Any:
        if relax:
            return model.add_variable(lb=lb, ub=ub, name=name)
        return model.add_integer_variable(lb=lb, ub=ub, name=name)

    instances = tuple(var(0, max_instances_per_row, f"n_{i}") for i in range(len(rows)))
    replicas = tuple(
        var(0, max_instances_per_row * (c.candidate.price_row.gpu_count // c.tp), f"m_{j}")
        for j, c in enumerate(cols)
    )
    for i, row in enumerate(rows):
        used = mathopt.fast_sum(
            c.tp * m
            for c, m in zip(cols, replicas, strict=True)
            if index[c.candidate.price_row] == i
        )
        model.add_linear_constraint(used <= row.gpu_count * instances[i], name=f"gpus_{i}")
    if len(class_demands) >= 2:
        demands = _class_demands(
            model, cols, replicas, class_demands, integer_scaling and not relax
        )
    else:
        rates = [c.rates()[0] for c in cols]
        demands = (
            (
                model.add_linear_constraint(
                    mathopt.fast_sum(
                        _scaled_capacity(rps, integer_scaling) * m
                        for (rps, _), m in zip(rates, replicas, strict=True)
                    )
                    >= _scaled_demand(demand_rps, integer_scaling),
                    name="demand_requests",
                ),
                model.add_linear_constraint(
                    mathopt.fast_sum(
                        _scaled_capacity(tps, integer_scaling) * m
                        for (_, tps), m in zip(rates, replicas, strict=True)
                    )
                    >= _scaled_demand(demand_tps, integer_scaling),
                    name="demand_tokens",
                ),
            ),
        )
    if homogeneous:
        selectors = tuple(var(0, 1, f"y_{i}") for i in range(len(rows)))
        for i, (n, y) in enumerate(zip(instances, selectors, strict=True)):
            model.add_linear_constraint(n <= max_instances_per_row * y, name=f"select_{i}")
        model.add_linear_constraint(mathopt.fast_sum(selectors) <= 1, name="one_row")
    scale = CENTS_PER_USD if integer_scaling else 1
    day_cost = [HOURS_PER_DAY * row.price_usd_per_hour * scale for row in rows]
    if integer_scaling:
        day_cost = [round(cost) for cost in day_cost]
    model.minimize(mathopt.fast_sum(c * n for c, n in zip(day_cost, instances, strict=True)))
    return Formulation(
        model=model,
        rows=rows,
        columns=tuple(cols),
        instances=instances,
        replicas=replicas,
        request_demand=demands[0][0],
        token_demand=demands[0][1],
        objective_scale=scale,
        demands=demands,
    )


@dataclass(frozen=True)
class BalanceModel:
    """The routing LP over a fixed fleet: `theta` and `allocations[j][k]` (None where
    column j cannot serve class k)."""

    model: Any
    theta: Any
    allocations: tuple[tuple[Any, ...], ...]


def build_balance(
    cols: Sequence[Column],
    counts: Sequence[int],
    demands: Sequence[tuple[float, float]],
    *,
    theta_floor: float | None = None,
) -> BalanceModel:
    """LP spreading a fixed fleet (`counts[j]` replicas of `cols[j]`) over the classes.

    Allocations `x_{j,k} >= 0` with `sum_k x_{j,k} <= counts[j]`; class k's request and
    token capacity must reach `theta` times its demand (rows without demand are left out).
    It maximizes `theta` (at most 10,000), the uniform headroom; with `theta_floor` it
    keeps `theta >= theta_floor` and maximizes `sum x` instead, so replica time no class
    needs is still assigned to classes it can serve.
    """
    model = mathopt.Model(name="llmplan_balance")
    theta = model.add_variable(lb=0, ub=THETA_MAX, name="theta")
    allocations = tuple(
        tuple(
            model.add_variable(lb=0, ub=count, name=f"x_{j}_{k}") if rps > 0 else None
            for k, (rps, _) in enumerate(col.rates())
        )
        for j, (col, count) in enumerate(zip(cols, counts, strict=True))
    )
    for j, (row, count) in enumerate(zip(allocations, counts, strict=True)):
        used = [x for x in row if x is not None]
        if used:
            model.add_linear_constraint(mathopt.fast_sum(used) <= count, name=f"share_{j}")
    for k, demand in enumerate(demands):
        for which in (0, 1):
            if demand[which] <= 0:
                continue
            headroom = mathopt.fast_sum(  # capacity / demand, so coefficients are O(1)
                col.rates()[k][which] / demand[which] * x
                for col, row in zip(cols, allocations, strict=True)
                if (x := row[k]) is not None
            )
            model.add_linear_constraint(headroom >= theta, name=f"cap_{k}_{which}")
    if theta_floor is None:
        model.maximize(theta)
    else:
        model.add_linear_constraint(theta >= theta_floor, name="theta_floor")
        model.maximize(mathopt.fast_sum(x for row in allocations for x in row if x is not None))
    return BalanceModel(model=model, theta=theta, allocations=allocations)
