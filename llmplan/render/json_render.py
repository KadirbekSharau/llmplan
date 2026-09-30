"""JSON renderer: every field in bytes, deterministic key order, ISO dates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, get_args

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.memory.fit import FitRequest, FitResult
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import model_info
from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.perf.config import ReplicaConfig
from llmplan.perf.estimate import INPUT_STAT_FIELDS, OUTPUT_STAT_FIELDS, PerfEstimate, StatsLike
from llmplan.planner.request import PlanRequest
from llmplan.planner.result import PlanResult
from llmplan.render import register
from llmplan.types import KVDType
from llmplan.workload import Workload, WorkloadStats

REQUEST_FIELDS = {"tensor_parallel", "dtype", "quantize_embeddings", "context_len"}


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2) + "\n"


@register("json")
class JsonRenderer:
    """`FitResult` fields at top level, plus the resolved inputs that produced them."""

    def fit(self, request: FitRequest, result: FitResult) -> str:
        return _dumps(
            {
                **result.model_dump(mode="json"),
                "request": request.model_dump(mode="json", include=REQUEST_FIELDS),
                "model": request.model.model_dump(mode="json"),
                "gpu": request.gpu.model_dump(mode="json"),
                "engine": request.engine.model_dump(mode="json"),
            }
        )

    def model_info(self, spec: ModelSpec) -> str:
        return _dumps(
            {
                "model": spec.model_dump(mode="json"),
                "derived": model_info(spec).model_dump(mode="json"),
                "kv_bytes_per_token_total": {
                    kv: kv_bytes_per_token_total(spec, kv) for kv in get_args(KVDType)
                },
            }
        )

    def gpus(self, gpus: Mapping[str, GPUSpec]) -> str:
        return _dumps([gpu.model_dump(mode="json") for gpu in gpus.values()])

    def workload_stats(self, workload: Workload, stats: WorkloadStats) -> str:
        return _dumps(
            {
                "source": workload.source,
                "format": workload.format,
                "dropped_rows": workload.dropped_rows,
                "notes": list(workload.notes),
                "stats": stats.model_dump(mode="json"),
            }
        )

    def perf_estimate(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        config: ReplicaConfig,
        stats: StatsLike,
        result: PerfEstimate,
    ) -> str:
        return _dumps(
            {
                **result.model_dump(mode="json"),
                "model": model.id,
                "gpu": gpu.id,
                "config": config.model_dump(mode="json"),
                "stats": {
                    name: getattr(stats, name) for name in (*INPUT_STAT_FIELDS, *OUTPUT_STAT_FIELDS)
                },
            }
        )

    def benchmarks(self, rows: Sequence[BenchmarkRow]) -> str:
        return _dumps([row.model_dump(mode="json") for row in rows])

    def plan(self, request: PlanRequest, result: PlanResult) -> str:
        return _dumps(
            {
                **result.model_dump(mode="json"),
                "request": {
                    "model": request.model.id,
                    "slo": request.slo.model_dump(mode="json"),
                    "engine": request.engine.model_dump(mode="json"),
                    "options": request.options.model_dump(mode="json"),
                    "stats": request.stats.model_dump(mode="json"),
                },
            }
        )
