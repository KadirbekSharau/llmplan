"""Brute force for the class MILP (M7_DESIGN.md section 6.1, acceptance test 8.3).

`random_instance(seed)` draws a small instance: 1 to 3 price rows of 1 or 2 GPUs, at most 4
candidates (tensor parallel 1, and 2 on two-GPU rows), 1 or 2 classes, capacities per
candidate and class from the `fake_class` backend (a class marked None fails the SLO), at
most 6 instances per row. `brute_force_optimum` enumerates every instance vector in order
of cost and, for each, every maximal replica vector the GPUs allow (unused GPUs never help:
replicas cost nothing), and decides whether the allocation LP is feasible (GLOP); the first
feasible fleet's cost is the optimum. It shares nothing with the planner but the inputs.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from llmplan.catalog.hardware import PriceRow
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest
from llmplan.workload.classes import DemandClass
from tests.fake_planner import MODEL, ClassCapacity, fake_gpu, fake_row, stats

MAX_INSTANCES = 6
MAX_CANDIDATES = 4
TP_CHOICES = (1, 2)
HOURS_PER_DAY = 24
EPS = 1e-9


@dataclass(frozen=True)
class Instance:
    """Price rows, `fake_class` capacities per (gpu_id, tp), and (req/s, tokens/s) demand
    per class."""

    rows: tuple[PriceRow, ...]
    capacities: dict[tuple[str, int], tuple[ClassCapacity, ...]]
    demands: tuple[tuple[float, float], ...]

    def request(self) -> PlanRequest:
        """The planner request for this instance (SLO 500 ms, utilization 1.0)."""
        classes = tuple(
            DemandClass(
                index=k,
                input_lo=1000 * k + 1,
                input_hi=1000 * (k + 1),
                output_lo=0,
                output_hi=100,
                share=1 / len(self.demands),
                peak_rps=rps,
                peak_output_tokens_per_s=tps,
                input_tokens_mean=500.0 + 1000 * k,
                output_tokens_mean=50.0,
                input_tokens_p50=500.0 + 1000 * k,
                output_tokens_p50=50.0,
                input_tokens_p95=900.0 + 1000 * k,
                output_tokens_p95=90.0,
            )
            for k, (rps, tps) in enumerate(self.demands)
        )
        return PlanRequest(
            model=MODEL,
            stats=stats(sum(d for d, _ in self.demands), sum(t for _, t in self.demands)),
            slo=SLO(ttft_ms_p95=500.0, utilization_target=1.0),
            engine=EngineProfile(),
            options=PlanOptions(
                tensor_parallel_choices=TP_CHOICES,
                dtype_choices=("bf16",),
                max_num_seqs_choices=(64,),
                max_model_len=8192,
                perf_backend="fake_class",
                max_instances_per_row=MAX_INSTANCES,
            ),
            gpus={row.gpu_id: fake_gpu(row.gpu_id) for row in self.rows},
            prices=self.rows,
            classes=classes,
        )


def random_instance(seed: int) -> Instance:
    """A seeded random instance within the section 6.1 limits."""
    rng = random.Random(seed)  # noqa: S311  # seeded test instances, not cryptography
    n_rows = rng.randint(1, 3)
    rows: list[PriceRow] = []
    budget = MAX_CANDIDATES
    for i in range(n_rows):
        gpus = rng.choice((1, 2))
        if gpus + (n_rows - i - 1) > budget:
            gpus = 1
        budget -= gpus  # one candidate per tensor-parallel degree that fits the row
        rows.append(fake_row(f"r{i}", f"bf-{seed}-{i}", gpus, round(rng.uniform(0.5, 6.0), 2)))
    n_classes = rng.randint(1, 2)
    capacities: dict[tuple[str, int], tuple[ClassCapacity, ...]] = {}
    for row in rows:
        for tp in TP_CHOICES[: row.gpu_count]:
            capacities[(row.gpu_id, tp)] = tuple(
                None
                if rng.random() < 0.25
                else (round(rng.uniform(0.5, 4.0) * tp, 3), round(rng.uniform(50, 500) * tp, 1))
                for _ in range(n_classes)
            )
    first = next(iter(capacities))
    for k in range(n_classes):
        if all(caps[k] is None for caps in capacities.values()):
            caps = list(capacities[first])
            caps[k] = (round(rng.uniform(0.5, 4.0), 3), round(rng.uniform(50, 500), 1))
            capacities[first] = tuple(caps)
    demands = tuple(
        (round(rng.uniform(0.5, 10.0), 2), round(rng.uniform(0.0, 1500.0), 1))
        for _ in range(n_classes)
    )
    return Instance(rows=tuple(rows), capacities=capacities, demands=demands)


def _row_replicas(gpus: int, tps: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Every replica vector for one row's candidates (tensor parallel `tps`) that fits in
    `gpus` GPUs and is maximal: no candidate could add a replica."""
    ranges = [range(gpus // tp + 1) for tp in tps]
    for counts in itertools.product(*ranges):
        left = gpus - sum(c * tp for c, tp in zip(counts, tps, strict=True))
        if left >= 0 and all(left < tp for tp in tps):
            yield counts


def _feasible(
    replicas: Sequence[int],
    rates: Sequence[Sequence[tuple[float, float]]],
    demands: Sequence[tuple[float, float]],
) -> bool:
    """Is there an allocation x >= 0, sum_k x[r][k] <= replicas[r], meeting every class's
    request and token demand? Necessary per-class checks first, then GLOP."""
    for k, (rps, tps) in enumerate(demands):
        if sum(m * r[k][0] for m, r in zip(replicas, rates, strict=True)) < rps - EPS:
            return False
        if sum(m * r[k][1] for m, r in zip(replicas, rates, strict=True)) < tps - EPS:
            return False
    if len(demands) == 1:
        return True  # one class: x = replicas
    model = mathopt.Model(name="brute_force")
    x = [
        [model.add_variable(lb=0, ub=m) if rate[k][0] > 0 else None for k in range(len(demands))]
        for m, rate in zip(replicas, rates, strict=True)
    ]
    for m, row in zip(replicas, x, strict=True):
        used = [v for v in row if v is not None]
        if used:
            model.add_linear_constraint(sum(used) <= m)
    for k, (rps, tps) in enumerate(demands):
        pairs = [
            (rate[k], row[k]) for rate, row in zip(rates, x, strict=True) if row[k] is not None
        ]
        model.add_linear_constraint(sum(r * v for (r, _), v in pairs) >= rps)
        model.add_linear_constraint(sum(t * v for (_, t), v in pairs) >= tps)
    result = mathopt.solve(model, mathopt.SolverType.GLOP)
    return bool(result.termination.reason == mathopt.TerminationReason.OPTIMAL)


def brute_force_optimum(instance: Instance) -> float | None:
    """The cheapest fleet's USD per day, or None when no fleet within the limits works."""
    rows, demands = instance.rows, instance.demands
    per_row = [
        [(tp, instance.capacities[(row.gpu_id, tp)]) for tp in TP_CHOICES[: row.gpu_count]]
        for row in rows
    ]
    rates = [
        [(0.0, 0.0) if c is None else c for c in caps] for cands in per_row for _, caps in cands
    ]

    def cost(instances: tuple[int, ...]) -> float:
        return sum(
            HOURS_PER_DAY * row.price_usd_per_hour * n
            for row, n in zip(rows, instances, strict=True)
            if n
        )

    vectors = itertools.product(range(MAX_INSTANCES + 1), repeat=len(rows))
    for instances in sorted(vectors, key=lambda v: (cost(v), v)):
        options = [
            list(_row_replicas(row.gpu_count * n, [tp for tp, _ in cands]))
            for row, n, cands in zip(rows, instances, per_row, strict=True)
        ]
        for combo in itertools.product(*options):
            replicas = [m for row_counts in combo for m in row_counts]
            if _feasible(replicas, rates, demands):
                return cost(instances)
    return None
