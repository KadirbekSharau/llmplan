"""`llmplan plan`: a thin typer command over `llmplan.planner.plan` (M4_DESIGN.md section 9).

Registered on the main app in `llmplan.cli`; errors map to exit codes there (4 infeasible,
5 solver). Workload statistics come from a trace (`--trace`) or from a JSON file
(`--stats-json`): either the output of `llmplan workload stats --format-out json` or a bare
`WorkloadStats` object.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import typer

from llmplan import render
from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import load_model
from llmplan.errors import ValidationError
from llmplan.memory.engine import EngineProfile
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.render.vllm_cmd import plan_commands
from llmplan.workload import WorkloadStats, compute_stats, load_workload

MAX_STATS_JSON_BYTES = 1_000_000

_T = TypeVar("_T")


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # deferred: llmplan.cli imports this module

    run(produce)


def _choices(flag: str, value: str | None, cast: Callable[[str], _T]) -> tuple[_T, ...] | None:
    if value is None:
        return None
    try:
        return tuple(cast(part.strip()) for part in value.split(","))
    except ValueError:
        raise ValidationError(f"{flag} must be a comma-separated list, got {value!r}") from None


def _stats_from_json(path: Path) -> WorkloadStats:
    try:
        if path.stat().st_size > MAX_STATS_JSON_BYTES:
            raise ValidationError(
                f"--stats-json {path} is larger than {MAX_STATS_JSON_BYTES} bytes"
            )
        doc: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"--stats-json {path} is not readable: {exc.strerror}") from None
    except json.JSONDecodeError as exc:
        raise ValidationError(f"--stats-json {path} is not valid JSON: {exc.msg}") from None
    if isinstance(doc, dict) and isinstance(doc.get("stats"), dict):
        doc = doc["stats"]  # the `llmplan workload stats --format-out json` layout
    return WorkloadStats.model_validate(doc)


def _stats(trace: Path | None, stats_json: Path | None) -> WorkloadStats:
    if trace is not None and stats_json is None:
        return compute_stats(load_workload(trace))
    if stats_json is not None and trace is None:
        return _stats_from_json(stats_json)
    raise ValidationError("pass exactly one of --trace or --stats-json")


def plan_command(
    model: Annotated[str, typer.Option("--model", help="HF repo id (org/name) or fixture:<name>.")],
    max_model_len: Annotated[int, typer.Option("--max-model-len", help="vLLM max_model_len.")],
    trace: Annotated[Path | None, typer.Option("--trace", help="Workload trace file.")] = None,
    stats_json: Annotated[
        Path | None, typer.Option("--stats-json", help="WorkloadStats JSON (workload stats).")
    ] = None,
    ttft_p95_ms: Annotated[
        float | None, typer.Option("--ttft-p95-ms", help="p95 TTFT bound (service time).")
    ] = None,
    tpot_p95_ms: Annotated[
        float | None, typer.Option("--tpot-p95-ms", help="p95 TPOT bound (service time).")
    ] = None,
    utilization: Annotated[
        float, typer.Option("--utilization", help="Capacity derating (utilization target).")
    ] = 0.8,
    gpus: Annotated[str | None, typer.Option("--gpus", help="GPU ids, comma-separated.")] = None,
    providers: Annotated[
        str | None, typer.Option("--providers", help="Providers, comma-separated.")
    ] = None,
    tp: Annotated[str, typer.Option("--tp", help="Tensor parallel choices.")] = "1,2,4,8",
    dtypes: Annotated[str, typer.Option("--dtypes", help="Weight dtype choices.")] = "bf16,fp8",
    max_num_seqs: Annotated[
        str, typer.Option("--max-num-seqs", help="max_num_seqs choices.")
    ] = "32,64,128,256",
    homogeneous: Annotated[
        bool, typer.Option("--homogeneous", help="Use a single price row.")
    ] = False,
    perf_backend: Annotated[
        Literal["auto", "roofline", "table"],
        typer.Option("--perf-backend", help="Performance backend."),
    ] = "auto",
    solver: Annotated[
        Literal["highs", "cp_sat", "scip", "gurobi"], typer.Option("--solver", help="MILP solver.")
    ] = "highs",
    time_limit: Annotated[
        float, typer.Option("--time-limit", help="Solver time limit, seconds.")
    ] = 60.0,
    gpu_catalog: Annotated[
        Path | None, typer.Option("--gpu-catalog", help="GPU catalog YAML (default: shipped).")
    ] = None,
    prices: Annotated[
        Path | None, typer.Option("--prices", help="Price catalog YAML (default: shipped).")
    ] = None,
    fmt: Annotated[
        Literal["text", "json", "vllm"], typer.Option("--format", help="Output format.")
    ] = "text",
) -> None:
    """Cheapest fleet and replica configs that meet peak demand and the latency SLO."""

    def produce() -> str:
        stats = _stats(trace, stats_json)
        catalog = load_gpus(gpu_catalog)
        options = PlanOptions.model_validate(
            {
                "gpu_ids": _choices("--gpus", gpus, str),
                "providers": _choices("--providers", providers, str),
                "tensor_parallel_choices": _choices("--tp", tp, int),
                "dtype_choices": _choices("--dtypes", dtypes, str),
                "max_num_seqs_choices": _choices("--max-num-seqs", max_num_seqs, int),
                "max_model_len": max_model_len,
                "homogeneous": homogeneous,
                "perf_backend": perf_backend,
                "solver": solver,
                "time_limit_s": time_limit,
            }
        )
        request = PlanRequest(
            model=load_model(model),
            stats=stats,
            slo=SLO(
                ttft_ms_p95=ttft_p95_ms, tpot_ms_p95=tpot_p95_ms, utilization_target=utilization
            ),
            engine=EngineProfile(),
            options=options,
            gpus=catalog,
            prices=load_prices(prices, gpus=catalog),
        )
        result = plan(request)
        if fmt == "vllm":
            return plan_commands(request, result)
        return render.get(fmt).plan(request, result)

    _run(produce)
