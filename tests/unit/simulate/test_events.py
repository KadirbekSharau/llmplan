from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from llmplan.errors import ValidationError
from llmplan.simulate.events import REQUEST_COLUMNS, EngineRun, run_events
from llmplan.simulate.replica import ReplicaSpec, replica_specs
from tests.fake_planner import GPUS, sim_plan

ONE_SLOT = ReplicaSpec(
    slots=1,
    kv_token_capacity=10**9,
    kv_bytes_per_token=100,
    weight_bytes=10**9,
    prefill_tokens_per_s=1000.0,
    tpot_s=0.01,
)


def _first(outstanding: Sequence[int], index: int) -> int:
    return 0


def _run(
    arrivals: list[float],
    input_tokens: int = 100,
    output_tokens: int = 40,
    replicas: Sequence[ReplicaSpec] = (ONE_SLOT,),
) -> EngineRun:
    n = len(arrivals)
    return run_events(
        replicas,
        np.asarray(arrivals, dtype=np.float64),
        np.full(n, input_tokens, dtype=np.int64),
        np.full(n, output_tokens, dtype=np.int64),
        _first,
    )


def test_replica_specs_from_plan() -> None:
    specs = replica_specs(sim_plan(replicas=3, effective_batch=8, kv_token_capacity=1000))
    assert len(specs) == 3
    spec = specs[0]
    assert spec.slots == 8
    assert spec.kv_token_capacity == 1000
    assert spec.prefill_tokens_per_s == 1000.0
    assert spec.tpot_s == 0.01
    assert spec.kv_bytes_per_token > 0
    assert spec.weight_bytes > 0
    assert spec.vram_bytes_total is None  # no GPU catalog given


def test_replica_specs_vram_total_from_the_gpu_catalog() -> None:
    plan = sim_plan()
    (spec,) = replica_specs(plan, GPUS)
    assert spec.vram_bytes_total == GPUS["fake-a"].vram_bytes * 1  # tensor parallel 1
    (unknown,) = replica_specs(plan, {"fake-b": GPUS["fake-b"]})
    assert unknown.vram_bytes_total is None


def test_replica_specs_rejects_unservable_plans() -> None:
    with pytest.raises(ValidationError, match="kv_token_capacity 0"):
        replica_specs(sim_plan(kv_token_capacity=0))
    plan = sim_plan()
    (replica,) = plan.replicas
    no_perf = replica.model_copy(
        update={"candidate": replica.candidate.model_copy(update={"perf": None})}
    )
    with pytest.raises(ValidationError, match="no fit or perf estimate"):
        replica_specs(plan.model_copy(update={"replicas": (no_perf,)}))
    with pytest.raises(ValidationError, match="no replicas"):
        replica_specs(plan.model_copy(update={"replicas": ()}))


def test_no_queueing_service_times_are_exact() -> None:
    run = _run([float(i) for i in range(100)])
    frame = run.requests.frame
    assert tuple(frame.columns) == REQUEST_COLUMNS
    assert (frame["ttft_ms"] == 100.0).all()
    assert (frame["e2e_ms"] == 500.0).all()
    assert (frame["tpot_ms"] == 10.0).all()
    assert (frame["start_s"] == frame["arrival_s"]).all()
    assert run.steps[0].queue_depth.max() == 0
    assert run.n_kv_capped == 0


def test_overload_builds_a_fifo_queue() -> None:
    run = _run([0.25 * i for i in range(100)])
    frame = run.requests.frame
    assert frame["start_s"].tolist() == [0.5 * i for i in range(100)]
    assert run.steps[0].queue_depth.max() >= 45
    assert frame["ttft_ms"].iloc[-1] > 20_000


def test_kv_limits_admission_before_slots() -> None:
    spec = ReplicaSpec(
        slots=8,
        kv_token_capacity=1000,
        kv_bytes_per_token=100,
        weight_bytes=10**9,
        prefill_tokens_per_s=1000.0,
        tpot_s=0.01,
    )
    run = _run([0.0] * 10, input_tokens=200, output_tokens=100, replicas=(spec,))
    starts = run.requests.frame["start_s"].tolist()
    service = 0.2 + 100 * 0.01
    assert starts[:3] == [0.0, 0.0, 0.0]
    assert starts[3] == service
    assert run.steps[0].queue_depth.max() == 7
    assert run.steps[0].kv_tokens.max() == 900
    assert run.steps[0].busy_slots.max() == 3


def test_oversized_request_is_capped_and_runs_alone() -> None:
    spec = ReplicaSpec(
        slots=4,
        kv_token_capacity=100,
        kv_bytes_per_token=1,
        weight_bytes=1,
        prefill_tokens_per_s=1000.0,
        tpot_s=0.01,
    )
    run = _run([0.0, 0.0], input_tokens=100, output_tokens=40, replicas=(spec,))
    frame = run.requests.frame
    assert run.n_kv_capped == 2
    assert frame["kv_tokens"].tolist() == [100, 100]
    assert frame["start_s"].tolist() == [0.0, 0.5]


def test_completion_before_arrival_at_the_same_time() -> None:
    run = _run([0.0, 0.5])  # the first completes at exactly 0.5
    assert run.requests.frame["start_s"].tolist() == [0.0, 0.5]
    assert run.steps[0].queue_depth.max() == 0


def test_routing_callback_places_requests() -> None:
    def alternate(outstanding: Sequence[int], index: int) -> int:
        return index % len(outstanding)

    n = 6
    run = run_events(
        (ONE_SLOT, ONE_SLOT),
        np.zeros(n),
        np.full(n, 100, dtype=np.int64),
        np.full(n, 40, dtype=np.int64),
        alternate,
    )
    assert run.requests.frame["replica_index"].tolist() == [0, 1, 0, 1, 0, 1]
    assert [step.queue_depth.max() for step in run.steps] == [2, 2]
