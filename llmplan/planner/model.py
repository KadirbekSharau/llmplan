"""MathOpt formulation of the fleet MILP (M4_DESIGN.md section 5). Building only; no solving.

    minimize    sum_p 24 * price_p * n_p
    subject to  sum_{r in R_p} tp_r * m_r <= g_p * n_p      for each price row p
                sum_r cap_r * m_r >= D_req                   (request demand)
                sum_r tok_r * m_r >= D_tok                   (token demand)
                [homogeneous] n_p <= max_instances * y_p;  sum_p y_p <= 1

`n_p` are instances of row p, `m_r` replicas of column r, `y_p` binary row selectors.
With `integer_scaling` every coefficient is an integer for CP-SAT: cents per day for the
objective and milli-units for capacities, rounded so a scaled solution is always feasible
in the unscaled model (capacities down, demand up). With `relax` the variables are
continuous: the LP relaxation used to explain which demand constraint binds.
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


@dataclass(frozen=True)
class Formulation:
    """A built MathOpt model plus the handles needed to read a solution back.

    `rows[i]` owns `instances[i]`; `columns[j]` owns `replicas[j]`. The objective is in
    USD per day times `objective_scale` (1, or 100 when integer-scaled).
    """

    model: Any  # mathopt.Model; ortools ships no type information
    rows: tuple[PriceRow, ...]
    columns: tuple[Column, ...]
    instances: tuple[Any, ...]
    replicas: tuple[Any, ...]
    request_demand: Any
    token_demand: Any
    objective_scale: int


def _scaled_capacity(value: float, integer_scaling: bool) -> float:
    return math.floor(value * MILLI + ROUNDING_GUARD) if integer_scaling else value


def _scaled_demand(value: float, integer_scaling: bool) -> float:
    return math.ceil(value * MILLI - ROUNDING_GUARD) if integer_scaling else value


def build_model(
    cols: Sequence[Column],
    *,
    demand_rps: float,
    demand_tps: float,
    homogeneous: bool,
    max_instances_per_row: int,
    integer_scaling: bool = False,
    relax: bool = False,
) -> Formulation:
    """Build the section 5 model over the eligible, pruned `cols` (at least one).

    Rows are the distinct price rows of `cols` in first-seen order. Variable and constraint
    names are deterministic (`n_<i>`, `m_<j>`, `y_<i>`, `gpus_<i>`, `demand_*`).
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
    request_demand = model.add_linear_constraint(
        mathopt.fast_sum(
            _scaled_capacity(c.rps, integer_scaling) * m
            for c, m in zip(cols, replicas, strict=True)
        )
        >= _scaled_demand(demand_rps, integer_scaling),
        name="demand_requests",
    )
    token_demand = model.add_linear_constraint(
        mathopt.fast_sum(
            _scaled_capacity(c.tps, integer_scaling) * m
            for c, m in zip(cols, replicas, strict=True)
        )
        >= _scaled_demand(demand_tps, integer_scaling),
        name="demand_tokens",
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
        request_demand=request_demand,
        token_demand=token_demand,
        objective_scale=scale,
    )
