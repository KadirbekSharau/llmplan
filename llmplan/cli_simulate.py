"""`llmplan simulate`: a thin typer command over `llmplan.simulate.replay` (M5_DESIGN.md
section 8).

Registered on the main app in `llmplan.cli`. `--plan` is the `llmplan plan --format json`
output. Latency budgets come from `--ttft-p95-ms` / `--tpot-p95-ms`, else from the SLO the
plan was made with (`request.slo` in the plan file); with neither, violations are not
counted and the output says so. `--png` also writes the timeline figure. `--gpu-catalog`
(default: the shipped catalog) supplies each replica's total VRAM. M7: `--routing auto`
(default) is `class_weighted` for a plan with routing weights and `least_outstanding`
otherwise; `--kv-accounting` picks incremental (default) or full KV reservation;
`--requests-csv` also writes the per-request records.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from llmplan import render
from llmplan.catalog.hardware import load_gpus
from llmplan.errors import ValidationError
from llmplan.planner.result import load_plan_json
from llmplan.simulate import SimOptions, replay, replay_requests
from llmplan.simulate.timeline import RoutingName
from llmplan.workload import load_workload


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # deferred: llmplan.cli imports this module

    run(produce)


def simulate_command(
    plan_path: Annotated[
        Path, typer.Option("--plan", help="Plan JSON from `llmplan plan --format json`.")
    ],
    trace: Annotated[Path, typer.Option("--trace", help="Workload trace file to replay.")],
    window: Annotated[float, typer.Option("--window", help="Window length, seconds.")] = 60.0,
    routing: Annotated[
        Literal["auto", "least_outstanding", "round_robin", "class_weighted"],
        typer.Option("--routing", help="Routing policy (auto: class_weighted for class plans)."),
    ] = "auto",
    kv_accounting: Annotated[
        Literal["incremental", "full"],
        typer.Option("--kv-accounting", help="KV reservation: mean occupancy or input+output."),
    ] = "incremental",
    requests_csv: Annotated[
        Path | None, typer.Option("--requests-csv", help="Also write per-request rows (CSV).")
    ] = None,
    ttft_p95_ms: Annotated[
        float | None,
        typer.Option("--ttft-p95-ms", help="TTFT budget incl. queueing (default: plan SLO)."),
    ] = None,
    tpot_p95_ms: Annotated[
        float | None, typer.Option("--tpot-p95-ms", help="TPOT budget (default: plan SLO).")
    ] = None,
    png: Annotated[Path | None, typer.Option("--png", help="Also write the timeline PNG.")] = None,
    gpu_catalog: Annotated[
        Path | None,
        typer.Option("--gpu-catalog", help="GPU catalog YAML the plan used (default: shipped)."),
    ] = None,
    fmt: Annotated[
        Literal["text", "json"], typer.Option("--format", help="Output format.")
    ] = "text",
) -> None:
    """Replay a trace on a planned fleet: utilization, KV, queueing, and SLO violations."""

    def produce() -> str:
        plan, slo = load_plan_json(plan_path)
        chosen: RoutingName = routing if routing != "auto" else "least_outstanding"
        if routing == "auto" and plan.routing:
            chosen = "class_weighted"
        options = SimOptions(
            window_s=window,
            routing=chosen,
            ttft_budget_ms=ttft_p95_ms,
            tpot_budget_ms=tpot_p95_ms,
            kv_accounting=kv_accounting,
        )
        workload = load_workload(trace)
        timeline = replay(plan, workload, slo=slo, options=options, gpus=load_gpus(gpu_catalog))
        if requests_csv is not None:
            frame = replay_requests(plan, workload, options=options).frame
            try:
                frame.to_csv(requests_csv, index=False)
            except OSError as exc:
                raise ValidationError(
                    f"--requests-csv {requests_csv} is not writable: {exc.strerror}"
                ) from None
        if png is not None:
            from llmplan.render.plots import save_png  # matplotlib only when asked for

            save_png(timeline, png)
        return render.get(fmt).timeline(timeline)

    _run(produce)
