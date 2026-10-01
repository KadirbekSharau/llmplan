"""Planner inputs (M4_DESIGN.md section 3): `SLO`, `PlanOptions`, `PlanRequest`.

Validation happens here, once; the planner trusts these models. Checks that need several
inputs together (catalog foreign keys, `max_model_len` against the model) are done by
`llmplan.planner.plan` so they raise the package's typed errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec, PriceRow
from llmplan.catalog.models import ModelSpec
from llmplan.memory.engine import EngineProfile
from llmplan.types import Commitment, DType
from llmplan.workload.classes import DemandClass
from llmplan.workload.schema import WorkloadStats

SolverName = Literal["highs", "cp_sat", "scip", "gurobi"]
PositiveInt = Annotated[int, Field(gt=0)]
MAX_SEED = 2**31 - 1  # largest seed every MathOpt backend accepts unchanged
SHARE_TOLERANCE = 1e-6


class SLO(BaseModel):
    """Latency targets and the utilization derating applied to every replica's capacity.

    `ttft_ms_p95` and `tpot_ms_p95` bound the performance model's p95 service times (no
    queueing; that is M5). None means unconstrained. Capacity is multiplied by
    `utilization_target` before it is matched against demand.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ttft_ms_p95: float | None = Field(default=None, gt=0)
    tpot_ms_p95: float | None = Field(default=None, gt=0)
    utilization_target: float = Field(default=0.8, gt=0, le=1)


class PlanOptions(BaseModel):
    """The search space and solver settings for one plan.

    Candidates are every in-scope price row x `tensor_parallel_choices` x `dtype_choices`
    x `max_num_seqs_choices`. `gpu_ids=None` means every GPU with a price row; `providers`
    None means every provider. `perf_backend` is a perf registry key or `"auto"`.
    `max_instances_per_row` bounds each row's instance count (the big-M of the
    homogeneous constraint). Solves are single-threaded with `seed` and `time_limit_s`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_ids: tuple[str, ...] | None = Field(default=None, min_length=1)
    providers: tuple[str, ...] | None = Field(default=None, min_length=1)
    commitments: tuple[Commitment, ...] = Field(default=("on_demand",), min_length=1)
    tensor_parallel_choices: tuple[PositiveInt, ...] = Field(default=(1, 2, 4, 8), min_length=1)
    dtype_choices: tuple[DType, ...] = Field(default=("bf16", "fp8"), min_length=1)
    max_num_seqs_choices: tuple[PositiveInt, ...] = Field(default=(32, 64, 128, 256), min_length=1)
    max_model_len: int = Field(gt=0)
    homogeneous: bool = False
    perf_backend: str = Field(default="auto", min_length=1)
    solver: SolverName = "highs"
    time_limit_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=MAX_SEED)
    max_instances_per_row: int = Field(default=1000, gt=0)

    @pydantic.model_validator(mode="after")
    def _no_duplicate_choices(self) -> Self:
        for name in (
            "gpu_ids",
            "providers",
            "commitments",
            "tensor_parallel_choices",
            "dtype_choices",
            "max_num_seqs_choices",
        ):
            values = getattr(self, name)
            if values is not None and len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates: {list(values)}")
        return self


class PlanRequest(BaseModel):
    """Everything one plan needs: model, workload statistics, SLO, engine, options, catalogs.

    The catalogs are passed explicitly (not read from disk) so callers and tests can inject
    rows. Every in-scope price row's `gpu_id` must be a key of `gpus`. `classes` (M7) are
    the workload's request-size classes from `classify` on the same trace and window as
    `stats`; empty means one class, the whole workload, exactly as in M4. Their indices must
    be 0..K-1 in order and their shares must sum to 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelSpec
    stats: WorkloadStats
    slo: SLO
    engine: EngineProfile
    options: PlanOptions
    gpus: Mapping[str, GPUSpec]
    prices: tuple[PriceRow, ...]
    classes: tuple[DemandClass, ...] = ()

    @pydantic.model_validator(mode="after")
    def _consistent_classes(self) -> Self:
        if [c.index for c in self.classes] != list(range(len(self.classes))):
            raise ValueError("classes must be indexed 0..K-1 in order")
        if self.classes and abs(sum(c.share for c in self.classes) - 1) > SHARE_TOLERANCE:
            raise ValueError("class shares must sum to 1")
        return self
