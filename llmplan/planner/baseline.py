"""The naive answer: the cheapest homogeneous fleet, found by enumeration (M4_DESIGN.md 7).

For each eligible column alone, the minimum replicas meeting both demands and the instances
of its price row they need; the cheapest over all columns (first in candidate order on
ties) is the baseline. No solver is involved.
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


def replicas_needed(demand: float, capacity: float) -> int:
    """Smallest integer `k` with `k * capacity >= demand` (0 when there is no demand)."""
    if demand <= 0:
        return 0
    return math.ceil(demand / capacity * (1 - RATIO_GUARD))


def best_homogeneous(
    cols: Sequence[Column], demand_rps: float, demand_tps: float, max_instances_per_row: int
) -> HomogeneousFleet | None:
    """Cheapest single-column fleet meeting both demands within `max_instances_per_row`,
    or None when no column can."""
    best: HomogeneousFleet | None = None
    for col in cols:
        row = col.candidate.price_row
        replicas = max(
            replicas_needed(demand_rps, col.rps), replicas_needed(demand_tps, col.tps), 1
        )
        instances = math.ceil(replicas / col.candidate.replicas_per_instance)
        if instances > max_instances_per_row:
            continue
        cost = HOURS_PER_DAY * row.price_usd_per_hour * instances
        if best is None or cost < best.usd_per_day:
            best = HomogeneousFleet(col, replicas, instances, cost)
    return best
