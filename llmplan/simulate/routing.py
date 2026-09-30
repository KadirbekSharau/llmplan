"""Routing policies for trace replay (M5_DESIGN.md section 6).

A policy is a pure function `(outstanding, index) -> replica`: `outstanding[r]` is the
number of requests admitted or queued on replica `r` when request `index` (its position
in the trace) arrives. Adding a policy is one decorated function here.
"""

from __future__ import annotations

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
