"""The naive answer: the cheapest homogeneous fleet, found by enumeration (M4_DESIGN.md 7).

For each eligible column alone, the minimum replicas meeting both demands and the instances
of its price row they need; the cheapest over all columns (first in candidate order on
ties) is the baseline. No solver is involved. With request-size classes (M7) one column
must serve every class with demand, and its replicas split their time between classes, so
it needs `ceil(sum_k max(D_k / cap_k, T_k / tok_k))` replicas.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llmplan.planner.candidates import Column
from llmplan.planner.model import HOURS_PER_DAY

RATIO_GUARD = 1e-9  # 30 / 3 must need 10 replicas, not 11, despite float noise


@dataclass(frozen=True)
class HomogeneousFleet:
    """`replicas` of one column on `instances` of its price row, costing `usd_per_day`."""

    column: Column
    replicas: int
    instances: int
    usd_per_day: float


def replica_equivalents(
    rates: Sequence[tuple[float, float]], demands: Sequence[tuple[float, float]]
) -> float | None:
    """Replicas (fractional) one column needs for every class: `sum_k max(D_k / cap_k,
    T_k / tok_k)`, or None when it cannot serve a class that has demand."""
    total = 0.0
    for capacities, needs in zip(rates, demands, strict=True):
        need = 0.0
        for demand, capacity in zip(needs, capacities, strict=True):
            if demand > 0:
                if capacity <= 0:
                    return None
                need = max(need, demand / capacity)
        total += need
    return total


def best_homogeneous(
    cols: Sequence[Column],
    demand_rps: float,
    demand_tps: float,
    max_instances_per_row: int,
    *,
    class_demands: Sequence[tuple[float, float]] = (),
) -> HomogeneousFleet | None:
    """Cheapest single-column fleet meeting both demands within `max_instances_per_row`,
    or None when no column can. With two or more `class_demands` (M7) those replace the
    one-class demand and are matched against `Column.rates()`."""
    demands = tuple(class_demands) if len(class_demands) >= 2 else ((demand_rps, demand_tps),)
    best: HomogeneousFleet | None = None
    for col in cols:
        row = col.candidate.price_row
        needed = replica_equivalents(col.rates(), demands)
        if needed is None:
            continue
        replicas = max(math.ceil(needed * (1 - RATIO_GUARD)), 1)
        instances = math.ceil(replicas / col.candidate.replicas_per_instance)
        if instances > max_instances_per_row:
            continue
        cost = HOURS_PER_DAY * row.price_usd_per_hour * instances
        if best is None or cost < best.usd_per_day:
            best = HomogeneousFleet(col, replicas, instances, cost)
    return best
