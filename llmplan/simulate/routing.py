"""Routing policies for trace replay (M5_DESIGN.md section 6).

A policy is a pure function `(outstanding, index) -> replica`: `outstanding[r]` is the
number of requests admitted or queued on replica `r` when request `index` (its position
in the trace) arrives. Adding a policy is one decorated function here. `class_weighted`
(M7) needs the plan's routing weights and every request's class, so it is not in the
registry: `class_weighted(...)` builds its policy for one replay.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from llmplan.errors import UnknownRegistryKey
from llmplan.simulate.events import Route

_REGISTRY: dict[str, Route] = {}


def register(key: str) -> Callable[[Route], Route]:
    """Function decorator: register a routing policy under `key`."""

    def decorator(policy: Route) -> Route:
        _REGISTRY[key] = policy
        return policy

    return decorator


def get(key: str) -> Route:
    """Return the routing policy registered under `key`; `UnknownRegistryKey` otherwise."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRegistryKey(f"unknown routing policy {key!r}; known: {known}") from None


@register("least_outstanding")
def least_outstanding(outstanding: Sequence[int], index: int) -> int:
    """The replica with the fewest admitted plus queued requests; ties go to the lowest index."""
    return min(range(len(outstanding)), key=outstanding.__getitem__)


@register("round_robin")
def round_robin(outstanding: Sequence[int], index: int) -> int:
    """Request `index` goes to replica `index % n`, whatever the replicas' load."""
    return index % len(outstanding)


def class_weighted(
    replica_types: Sequence[int],
    rules: Sequence[Sequence[tuple[int, float]]],
    request_class: Sequence[int],
) -> Route:
    """The M7 `class_weighted` policy for one replay.

    `replica_types[r]` is replica r's type (its `ReplicaPlan` index), `rules[k]` the
    (type, weight) pairs of class k in type order, and `request_class[i]` request i's
    class. Each request of class k goes to the type whose count so far is furthest below
    its target (`weight x requests of class k so far` minus requests sent there: the
    largest cumulative deficit, ties to the lowest type), which is deterministic and keeps
    every type within one request of its share; then to the least-outstanding replica of
    that type. A class without rules is routed least-outstanding over all replicas.
    """
    members = [
        [r for r, t in enumerate(replica_types) if t == i] for i in range(max(replica_types) + 1)
    ]
    everyone = list(range(len(replica_types)))
    sent = [[0] * len(rule) for rule in rules]
    seen = [0] * len(rules)
    targets: list[list[int]] = []
    for k in request_class:
        rule = rules[k]
        if not rule:
            targets.append(everyone)
            continue
        seen[k] += 1
        best, deficit = 0, -math.inf
        for j, (_, weight) in enumerate(rule):
            gap = weight * seen[k] - sent[k][j]
            if gap > deficit:
                best, deficit = j, gap
        sent[k][best] += 1
        targets.append(members[rule[best][0]])

    def route(outstanding: Sequence[int], index: int) -> int:
        return min(targets[index], key=outstanding.__getitem__)

    return route
