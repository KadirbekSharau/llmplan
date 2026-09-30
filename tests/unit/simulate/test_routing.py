from __future__ import annotations

import numpy as np
import pytest

from llmplan.errors import UnknownRegistryKey
from llmplan.simulate import routing
from llmplan.simulate.events import run_events
from llmplan.simulate.replica import ReplicaSpec

SPEC = ReplicaSpec(
    slots=1,
    kv_token_capacity=10**6,
    kv_bytes_per_token=1,
    weight_bytes=1,
    prefill_tokens_per_s=1000.0,
    tpot_s=0.01,
)


def test_least_outstanding_picks_fewest_then_lowest_index() -> None:
    policy = routing.get("least_outstanding")
    assert policy([2, 1, 1], 0) == 1
    assert policy([0, 0], 5) == 0
    assert policy([3, 2, 0], 9) == 2


def test_round_robin_ignores_load() -> None:
    policy = routing.get("round_robin")
    assert [policy([9, 0, 0], i) for i in range(5)] == [0, 1, 2, 0, 1]


def test_unknown_policy() -> None:
    with pytest.raises(UnknownRegistryKey, match="unknown routing policy 'random'"):
        routing.get("random")


def test_least_outstanding_spreads_a_burst() -> None:
    n = 6
    run = run_events(
        (SPEC, SPEC, SPEC),
        np.zeros(n),
        np.full(n, 100, dtype=np.int64),
        np.full(n, 40, dtype=np.int64),
        routing.get("least_outstanding"),
    )
    assert run.requests.frame["replica_index"].tolist() == [0, 1, 2, 0, 1, 2]
