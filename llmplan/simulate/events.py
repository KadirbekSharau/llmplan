"""Discrete-event engine for trace replay (M5_DESIGN.md section 5).

Plain Python with `heapq` and tuples in the loop; numpy only to package the results. The
heap holds completions `(time_s, seq, replica, request)`; arrivals are consumed in trace
order from the sorted arrays, so at equal times a completion is processed before an
arrival (a slot freed at `t` is available to a request arriving at `t`). `seq` breaks ties
between completions deterministically. KV accounting (M7) is `"full"` (a request reserves
and holds input + output tokens from admission) or `"incremental"` (it reserves its mean
occupancy, input + output / 2, and its KV in use grows linearly from input at admission to
input + output at completion).
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from llmplan.simulate.replica import MS_PER_S, ReplicaSpec, ReplicaState

Route = Callable[[Sequence[int], int], int]
"""(outstanding requests per replica, request index) -> replica index."""

REQUEST_COLUMNS: tuple[str, ...] = (
    "arrival_s",
    "start_s",
    "complete_s",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "replica_index",
    "kv_tokens",
    "class_index",
)


class RequestLog(BaseModel):
    """Per-request results of a replay, one row per simulated request in trace order.

    `frame` columns: `arrival_s`, `start_s` (admission), `complete_s` (float64 seconds);
    `ttft_ms = (start_s - arrival_s + input_tokens / prefill_tokens_per_s) * 1000`,
    `tpot_ms` (the replica's constant time per output token), `e2e_ms` (queueing plus
    service, float64 milliseconds); `replica_index`, `kv_tokens` (KV tokens held at
    completion, `input + output` capped at the replica's KV capacity) and (M7)
    `class_index` (the request's class in the plan, 0 without classes) as int64. The model
    is frozen but the frame is a pandas object: treat it as read-only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    frame: pd.DataFrame


@dataclass(frozen=True)
class StepLog:
    """State of one replica after each event that touched it: time, busy slots, KV tokens
    in use and the rate they change at until the next event (0 under full accounting), and
    queue depth (parallel float64 arrays, in event order). Before the first entry the
    replica is idle and empty."""

    time_s: npt.NDArray[np.float64]
    busy_slots: npt.NDArray[np.float64]
    kv_tokens: npt.NDArray[np.float64]
    kv_slope: npt.NDArray[np.float64]
    queue_depth: npt.NDArray[np.float64]


@dataclass(frozen=True)
class EngineRun:
    """Everything the event loop produced: per-request records, per-replica state logs, and
    the number of requests whose KV need was capped at their replica's capacity."""

    requests: RequestLog
    steps: tuple[StepLog, ...]
    n_kv_capped: int


def run_events(
    replicas: Sequence[ReplicaSpec],
    arrival_s: npt.NDArray[np.float64],
    input_tokens: npt.NDArray[np.int64],
    output_tokens: npt.NDArray[np.int64],
    route: Route,
    *,
    kv_accounting: Literal["incremental", "full"] = "full",
    class_index: npt.NDArray[np.int64] | None = None,
) -> EngineRun:
    """Replay requests (sorted by arrival) on `replicas`, routing each arrival with `route`.

    A request holds one slot for `input / prefill_tokens_per_s + output * tpot_s` seconds
    and reserves KV tokens: `input + output` (`"full"`) or `input + ceil(output / 2)`, its
    mean occupancy (`"incremental"`), capped at the replica's KV capacity so an oversized
    request runs alone instead of blocking forever. It starts on arrival when its replica
    has a free slot, enough unreserved KV tokens, and an empty queue; otherwise it joins
    that replica's FIFO queue, which is drained in order on every completion.
    `class_index` labels each request in the output (zeros when None).
    """
    arrivals: list[float] = arrival_s.tolist()
    inputs: list[int] = input_tokens.tolist()
    outputs: list[int] = output_tokens.tolist()
    incremental = kv_accounting == "incremental"
    n = len(arrivals)
    states = [ReplicaState(spec) for spec in replicas]
    outstanding = [0] * len(replicas)
    logs: list[list[tuple[float, int, float, float, int]]] = [[] for _ in replicas]
    start = [0.0] * n
    ttft = [0.0] * n
    e2e = [0.0] * n
    complete = [0.0] * n
    placed = [0] * n
    need = [0] * n  # KV tokens reserved for admission
    held = [0] * n  # KV tokens held at completion
    growth = [0.0] * n  # KV tokens/s while running (incremental)
    heap: list[tuple[float, int, int, int]] = []
    seq = itertools.count()
    n_capped = 0

    def begin(q: int, r: int, now: float) -> None:
        spec, state = replicas[r], states[r]
        state.admit(need[q])
        wait = now - arrivals[q]
        prefill = inputs[q] / spec.prefill_tokens_per_s
        service = prefill + outputs[q] * spec.tpot_s
        if incremental:
            initial = min(inputs[q], held[q])
            growth[q] = (held[q] - initial) / service
            state.advance(now)
            state.grow(initial, growth[q])
        start[q] = now
        ttft[q] = wait + prefill
        e2e[q] = wait + service
        complete[q] = now + service
        heapq.heappush(heap, (now + service, next(seq), r, q))

    i = 0
    while i < n or heap:
        if heap and (i == n or heap[0][0] <= arrivals[i]):
            now, _, r, q = heapq.heappop(heap)
            state = states[r]
            state.release(need[q])
            outstanding[r] -= 1
            if incremental:
                state.advance(now)
                state.shrink(held[q], growth[q], idle=state.free_slots == replicas[r].slots)
            queue = state.queue
            while queue and state.can_admit(need[queue[0]]):
                begin(queue.popleft(), r, now)
        else:
            now, q = arrivals[i], i
            i += 1
            r = route(outstanding, q)
            state = states[r]
            capacity = replicas[r].kv_token_capacity
            tokens = inputs[q] + outputs[q]
            if tokens > capacity:
                tokens = capacity
                n_capped += 1
            held[q] = tokens
            need[q] = min(inputs[q] + (outputs[q] + 1) // 2, capacity) if incremental else tokens
            placed[q] = r
            outstanding[r] += 1
            if not state.queue and state.can_admit(need[q]):
                begin(q, r, now)
            else:
                state.queue.append(q)
        if incremental:
            state.advance(now)
            kv, slope = state.kv_used, state.kv_slope
        else:
            kv, slope = float(replicas[r].kv_token_capacity - state.kv_free), 0.0
        logs[r].append((now, state.free_slots, kv, slope, len(state.queue)))

    where = np.asarray(placed, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64) if class_index is None else class_index
    frame = pd.DataFrame(
        {
            "arrival_s": arrival_s.astype(np.float64),
            "start_s": np.asarray(start, dtype=np.float64),
            "complete_s": np.asarray(complete, dtype=np.float64),
            "ttft_ms": np.asarray(ttft, dtype=np.float64) * MS_PER_S,
            "tpot_ms": np.asarray([spec.tpot_s for spec in replicas])[where] * MS_PER_S,
            "e2e_ms": np.asarray(e2e, dtype=np.float64) * MS_PER_S,
            "replica_index": where,
            "kv_tokens": np.asarray(held, dtype=np.int64),
            "class_index": labels.astype(np.int64),
        }
    )
    steps = tuple(_step_log(spec, log) for spec, log in zip(replicas, logs, strict=True))
    return EngineRun(requests=RequestLog(frame=frame), steps=steps, n_kv_capped=n_capped)


def _step_log(spec: ReplicaSpec, log: list[tuple[float, int, float, float, int]]) -> StepLog:
    table = np.asarray(log, dtype=np.float64).reshape(-1, 5)
    return StepLog(
        time_s=table[:, 0],
        busy_slots=spec.slots - table[:, 1],
        kv_tokens=table[:, 2],
        kv_slope=table[:, 3],
        queue_depth=table[:, 4],
    )
