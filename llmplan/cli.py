"""`llmplan` command line: a thin typer adapter over the library.

Exit codes (ARCHITECTURE.md section 7): 0 success (including "does not fit"), 2 usage or
validation, 3 catalog or fetch, 4 infeasible plan, 5 solver.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import typer

from llmplan import render
from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.cli_perf import perf_app
from llmplan.cli_plan import plan_command
from llmplan.cli_workload import traces_app, workload_app
from llmplan.errors import (
    CatalogError,
    InfeasiblePlan,
    LLMPlanError,
    SolverError,
    UnknownRegistryKey,
    ValidationError,
)
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, fit
from llmplan.types import DType, KVDType

app = typer.Typer(
    name="llmplan",
    help="CPU-only capacity planner for self-hosted LLM inference.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

Format = Literal["text", "json"]
ModelOpt = Annotated[str, typer.Option("--model", help="HF repo id (org/name) or fixture:<name>.")]
FormatOpt = Annotated[Format, typer.Option("--format", help="Output format.")]
GpusOpt = Annotated[
    Path | None, typer.Option("--gpus", help="GPU catalog YAML (default: data/gpus.yaml).")
]


def exit_code(exc: LLMPlanError) -> int:
    """Map an llmplan error to its CLI exit code."""
    if isinstance(exc, ValidationError | UnknownRegistryKey):
        return 2
    if isinstance(exc, CatalogError):
        return 3
    if isinstance(exc, InfeasiblePlan):
        return 4
    if isinstance(exc, SolverError):
        return 5
    return 1


def _run(produce: Callable[[], str]) -> None:
    try:
        output = produce()
    except LLMPlanError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exit_code(exc)) from None
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "input"
        typer.echo(f"error: invalid {field}: {err['msg']}", err=True)
        raise typer.Exit(2) from None
    typer.echo(output, nl=False)


@app.command("fit")
def fit_command(
    model: ModelOpt,
    gpu: Annotated[str, typer.Option("--gpu", help="GPU id from the catalog.")],
    tp: Annotated[int, typer.Option("--tp", help="Tensor parallel degree.")] = 1,
    dtype: Annotated[DType, typer.Option("--dtype", help="Weight dtype.")] = "bf16",
    kv_dtype: Annotated[KVDType, typer.Option("--kv-dtype", help="KV-cache dtype.")] = "bf16",
    context: Annotated[int, typer.Option("--context", help="Tokens per sequence.")] = 8192,
    gpu_mem_util: Annotated[
        float, typer.Option("--gpu-mem-util", help="vLLM gpu_memory_utilization.")
    ] = 0.9,
    max_num_batched_tokens: Annotated[
        int, typer.Option("--max-num-batched-tokens", help="vLLM max_num_batched_tokens.")
    ] = 8192,
    param_count_override: Annotated[
        int | None, typer.Option("--param-count-override", help="Use this parameter count.")
    ] = None,
    gpus: GpusOpt = None,
    fmt: FormatOpt = "text",
) -> None:
    """Does MODEL fit on GPU with tensor parallel TP, and how many sequences fit?"""

    def produce() -> str:
        spec = load_model(model)
        if param_count_override is not None:
            spec = ModelSpec.model_validate(
                {**spec.model_dump(), "param_count_override": param_count_override}
            )
        catalog = load_gpus(gpus)
        if gpu not in catalog:
            raise CatalogError(f"unknown gpu id {gpu!r}; known: {', '.join(catalog)}")
        request = FitRequest(
            model=spec,
            gpu=catalog[gpu],
            engine=EngineProfile(
                gpu_memory_utilization=gpu_mem_util,
                max_num_batched_tokens=max_num_batched_tokens,
                kv_dtype=kv_dtype,
            ),
            tensor_parallel=tp,
            dtype=dtype,
            context_len=context,
        )
        return render.get(fmt).fit(request, fit(request))

    _run(produce)


@app.command("model-info")
def model_info_command(model: ModelOpt, fmt: FormatOpt = "text") -> None:
    """Show the architecture integers, parameter count, and weight sizes of MODEL."""
    _run(lambda: render.get(fmt).model_info(load_model(model)))


@app.command("gpus")
def gpus_command(gpus: GpusOpt = None, fmt: FormatOpt = "text") -> None:
    """List the GPU catalog."""
    _run(lambda: render.get(fmt).gpus(load_gpus(gpus)))


app.command("plan")(plan_command)
app.add_typer(perf_app, name="perf")
app.add_typer(workload_app, name="workload")
app.add_typer(traces_app, name="traces")
