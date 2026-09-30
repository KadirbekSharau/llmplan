"""`llmplan perf ...` commands: a thin typer sub-app over `llmplan.perf` (M3_DESIGN.md 6).

Registered on the main app in `llmplan.cli`; errors map to exit codes there.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field

from llmplan import render
from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import load_model
from llmplan.errors import CatalogError, ValidationError
from llmplan.perf import ReplicaConfig, StatsLike, estimate
from llmplan.perf.benchmarks import load_benchmarks
from llmplan.types import DType
from llmplan.workload import compute_stats, load_workload

perf_app = typer.Typer(
    name="perf",
    help="Throughput and latency estimates for one replica.",
    no_args_is_help=True,
)

Format = Literal["text", "json"]
Backend = Literal["auto", "roofline", "table"]
FormatOpt = Annotated[Format, typer.Option("--format", help="Output format.")]


class ExplicitStats(BaseModel):
    """Token statistics given on the command line (the `StatsLike` fields)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens_mean: float = Field(ge=1)
    input_tokens_p50: float = Field(ge=1)
    input_tokens_p95: float = Field(ge=1)
    output_tokens_mean: float = Field(ge=0)
    output_tokens_p50: float = Field(ge=0)
    output_tokens_p95: float = Field(ge=0)


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # deferred: llmplan.cli imports this module

    run(produce)


def _gpu(gpu_id: str) -> GPUSpec:
    catalog = load_gpus()
    if gpu_id not in catalog:
        raise CatalogError(f"unknown gpu id {gpu_id!r}; known: {', '.join(catalog)}")
    return catalog[gpu_id]


STAT_FLAGS = {
    "input_tokens_mean": "--in-mean",
    "input_tokens_p50": "--in-p50",
    "input_tokens_p95": "--in-p95",
    "output_tokens_mean": "--out-mean",
    "output_tokens_p50": "--out-p50",
    "output_tokens_p95": "--out-p95",
}


def _stats(trace: Path | None, values: dict[str, float | None]) -> StatsLike:
    given = [STAT_FLAGS[name] for name, value in values.items() if value is not None]
    if trace is not None:
        if given:
            raise ValidationError(
                f"pass either --trace or the token statistics, not both (got {', '.join(given)})"
            )
        return compute_stats(load_workload(trace))
    missing = [STAT_FLAGS[name] for name, value in values.items() if value is None]
    if missing:
        raise ValidationError(f"missing workload statistics: {', '.join(missing)} (or --trace)")
    return ExplicitStats.model_validate(values)


@perf_app.command("estimate")
def estimate_command(
    model: Annotated[str, typer.Option("--model", help="HF repo id or fixture:<name>.")],
    gpu: Annotated[str, typer.Option("--gpu", help="GPU id from the catalog.")],
    tp: Annotated[int, typer.Option("--tp", help="Tensor parallel degree.")] = 1,
    dtype: Annotated[DType, typer.Option("--dtype", help="Weight dtype.")] = "bf16",
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs", help="vLLM max_num_seqs.")] = 256,
    max_model_len: Annotated[
        int, typer.Option("--max-model-len", help="vLLM max_model_len.")
    ] = 8192,
    trace: Annotated[
        Path | None, typer.Option("--trace", help="Workload trace file (instead of the stats).")
    ] = None,
    in_mean: Annotated[float | None, typer.Option("--in-mean", help="Mean input tokens.")] = None,
    in_p50: Annotated[float | None, typer.Option("--in-p50", help="p50 input tokens.")] = None,
    in_p95: Annotated[float | None, typer.Option("--in-p95", help="p95 input tokens.")] = None,
    out_mean: Annotated[
        float | None, typer.Option("--out-mean", help="Mean output tokens.")
    ] = None,
    out_p50: Annotated[float | None, typer.Option("--out-p50", help="p50 output tokens.")] = None,
    out_p95: Annotated[float | None, typer.Option("--out-p95", help="p95 output tokens.")] = None,
    backend: Annotated[Backend, typer.Option("--backend", help="Performance backend.")] = "auto",
    fmt: FormatOpt = "text",
) -> None:
    """Estimate throughput, TTFT, and TPOT of one replica of MODEL on GPU."""

    def produce() -> str:
        stats = _stats(
            trace,
            {
                "input_tokens_mean": in_mean,
                "input_tokens_p50": in_p50,
                "input_tokens_p95": in_p95,
                "output_tokens_mean": out_mean,
                "output_tokens_p50": out_p50,
                "output_tokens_p95": out_p95,
            },
        )
        spec, gpu_spec = load_model(model), _gpu(gpu)
        config = ReplicaConfig(
            tensor_parallel=tp, dtype=dtype, max_num_seqs=max_num_seqs, max_model_len=max_model_len
        )
        result = estimate(spec, gpu_spec, config, stats, backend=backend)
        return render.get(fmt).perf_estimate(spec, gpu_spec, config, stats, result)

    _run(produce)


@perf_app.command("benchmarks")
def benchmarks_command(
    gpu: Annotated[str | None, typer.Option("--gpu", help="Only rows for this GPU id.")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Only rows for this model (aliases apply).")
    ] = None,
    fmt: FormatOpt = "text",
) -> None:
    """List the shipped benchmark rows (data/benchmarks), optionally filtered."""

    def produce() -> str:
        if gpu is not None:
            _gpu(gpu)
        table = load_benchmarks()
        rows = [
            row
            for row in table.rows
            if (gpu is None or row.gpu_id == gpu)
            and (model is None or table.canonical(row.model_id) == table.canonical(model))
        ]
        return render.get(fmt).benchmarks(rows)

    _run(produce)
