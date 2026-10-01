"""`llmplan simulate`: a thin typer command over `llmplan.simulate.replay` (M5_DESIGN.md
section 8).

Registered on the main app in `llmplan.cli`. `--plan` is the `llmplan plan --format json`
output. Latency budgets come from `--ttft-p95-ms` / `--tpot-p95-ms`, else from the SLO the
plan was made with (`request.slo` in the plan file); with neither, violations are not
counted and the output says so. `--png` also writes the timeline figure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from llmplan import render
from llmplan.planner.result import load_plan_json
from llmplan.simulate import SimOptions, replay
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
        Literal["least_outstanding", "round_robin"],
        typer.Option("--routing", help="Routing policy."),
    ] = "least_outstanding",
    ttft_p95_ms: Annotated[
        float | None,
        typer.Option("--ttft-p95-ms", help="TTFT budget incl. queueing (default: plan SLO)."),
    ] = None,
    tpot_p95_ms: Annotated[
        float | None, typer.Option("--tpot-p95-ms", help="TPOT budget (default: plan SLO).")
    ] = None,
    png: Annotated[Path | None, typer.Option("--png", help="Also write the timeline PNG.")] = None,
    fmt: Annotated[
        Literal["text", "json"], typer.Option("--format", help="Output format.")
    ] = "text",
) -> None:
    """Replay a trace on a planned fleet: utilization, KV, queueing, and SLO violations."""

    def produce() -> str:
        plan, slo = load_plan_json(plan_path)
        options = SimOptions(
            window_s=window,
            routing=routing,
            ttft_budget_ms=ttft_p95_ms,
            tpot_budget_ms=tpot_p95_ms,
        )
        timeline = replay(plan, load_workload(trace), slo=slo, options=options)
        if png is not None:
            from llmplan.render.plots import save_png  # matplotlib only when asked for

            save_png(timeline, png)
        return render.get(fmt).timeline(timeline)

    _run(produce)
