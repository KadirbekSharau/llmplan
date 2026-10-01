"""Replica model for trace replay (M5_DESIGN.md section 4).

`ReplicaSpec` is the fixed description of one simulated replica, derived from a planned
candidate's M1 fit and M3 performance estimate. `ReplicaState` is the mutable slot and
KV-cache bookkeeping the event loop keeps for it; it never leaves `llmplan.simulate`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from llmplan.catalog.hardware import GPUSpec
from llmplan.errors import ValidationError
from llmplan.planner.result import PlanResult, label

MS_PER_S = 1000.0


@dataclass(frozen=True, slots=True)
class ReplicaSpec:
    """One replica: `slots` concurrent requests (`perf.effective_batch`), a KV cache of
    `kv_token_capacity` tokens (`fit.kv_token_capacity`) at `kv_bytes_per_token` bytes
    summed over its GPUs, `weight_bytes` summed over its GPUs, a prefill rate in tokens/s,
    and a constant time per output token (`perf.tpot_ms_p50`, the full-batch value).
    `vram_bytes_total` is the GPU's `vram_bytes` times tensor parallel; None when the GPU
    spec is unknown.
    """

    slots: int
    kv_token_capacity: int
    kv_bytes_per_token: int
    weight_bytes: int
    prefill_tokens_per_s: float
    tpot_s: float
    vram_bytes_total: int | None = None


def replica_specs(
    plan: PlanResult, gpus: Mapping[str, GPUSpec] | None = None
) -> tuple[ReplicaSpec, ...]:
    """Expand every `ReplicaPlan` of `plan` into `count` identical `ReplicaSpec`s, in plan order.

    `gpus` is the GPU catalog the plan was made with; a replica whose `gpu_id` it does not
    hold (every replica, when `gpus` is None) gets `vram_bytes_total` None. Raises
    `ValidationError` when the plan has no replicas or a planned candidate lacks a fit, a
    performance estimate, or a positive KV-cache capacity (it could not serve).
    """
    specs: list[ReplicaSpec] = []
    for replica in plan.replicas:
        candidate = replica.candidate
        fit, perf = candidate.fit, candidate.perf
        if fit is None or perf is None:
            raise ValidationError(f"replica {label(candidate)!r} has no fit or perf estimate")
        if fit.kv_token_capacity < 1:
            raise ValidationError(
                f"replica {label(candidate)!r} has kv_token_capacity {fit.kv_token_capacity}"
            )
        tp = candidate.config.tensor_parallel
        gpu = None if gpus is None else gpus.get(candidate.price_row.gpu_id)
        spec = ReplicaSpec(
            slots=perf.effective_batch,
            kv_token_capacity=fit.kv_token_capacity,
            kv_bytes_per_token=fit.kv_bytes_per_token_per_gpu * tp,
            weight_bytes=fit.per_gpu_weight_bytes * tp,
            prefill_tokens_per_s=perf.prefill_tokens_per_s,
            tpot_s=perf.tpot_ms_p50 / MS_PER_S,
            vram_bytes_total=None if gpu is None else gpu.vram_bytes * tp,
        )
        specs.extend([spec] * replica.count)
    if not specs:
        raise ValidationError("plan has no replicas to simulate")
    return tuple(specs)


class ReplicaState:
    """Free slots, free KV tokens, and the FIFO queue of request indices of one replica."""

    __slots__ = ("free_slots", "kv_free", "queue")

    def __init__(self, spec: ReplicaSpec) -> None:
        self.free_slots = spec.slots
        self.kv_free = spec.kv_token_capacity
        self.queue: deque[int] = deque()

    def can_admit(self, kv_tokens: int) -> bool:
        """A request needing `kv_tokens` starts now only with a free slot and enough KV."""
        return self.free_slots > 0 and kv_tokens <= self.kv_free

    def admit(self, kv_tokens: int) -> None:
        self.free_slots -= 1
        self.kv_free -= kv_tokens

    def release(self, kv_tokens: int) -> None:
        self.free_slots += 1
        self.kv_free += kv_tokens
