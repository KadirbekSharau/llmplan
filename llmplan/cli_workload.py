"""`llmplan workload` and `llmplan traces` subcommands (M2): thin adapters over `llmplan.workload`.

Registered on the main app in `llmplan.cli`. Errors go through the main app's handler, so
exit codes match ARCHITECTURE.md section 7. Output formats come from `llmplan.render`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from llmplan import render
from llmplan.workload import compute_stats, generate, load_workload
from llmplan.workload.fetch import fetch_trace
from llmplan.workload.formats.generic_csv import write_csv
from llmplan.workload.synth import parse_distribution

workload_app = typer.Typer(help="Characterize and synthesize request traces.", no_args_is_help=True)
traces_app = typer.Typer(help="Download public request traces.", no_args_is_help=True)

TraceFormatOpt = Literal["auto", "csv", "azure2023", "azure2024", "burstgpt"]


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # lazy: llmplan.cli imports this module

    run(produce)


@workload_app.command("stats")
def stats_command(
    trace: Annotated[Path, typer.Option("--trace", help="Local trace file (CSV).")],
    trace_format: Annotated[
        TraceFormatOpt, typer.Option("--format", help="Trace format; auto reads the header.")
    ] = "auto",
    window: Annotated[float, typer.Option("--window", help="Rate window in seconds.")] = 60.0,
    format_out: Annotated[
        Literal["text", "json"], typer.Option("--format-out", help="Output format.")
    ] = "text",
) -> None:
    """Summarize a trace: request and token rates, token percentiles, diurnal profile."""

    def produce() -> str:
        workload = load_workload(trace, format=None if trace_format == "auto" else trace_format)
        stats = compute_stats(workload, window_s=window)
        return render.get(format_out).workload_stats(workload, stats)

    _run(produce)


@workload_app.command("synth")
def synth_command(
    rps: Annotated[float, typer.Option("--rps", help="Mean arrival rate, requests/s.")],
    duration: Annotated[float, typer.Option("--duration", help="Trace length in seconds.")],
    out: Annotated[Path, typer.Option("--out", help="CSV file to write (generic csv).")],
    in_tokens: Annotated[
        str, typer.Option("--in-tokens", help="Input tokens: fixed:N, lognormal:M:S[:LO:HI], ...")
    ] = "lognormal:6.2:0.8",
    out_tokens: Annotated[
        str, typer.Option("--out-tokens", help="Output tokens, same syntax as --in-tokens.")
    ] = "lognormal:5.5:0.9",
    seed: Annotated[int, typer.Option("--seed", help="Random seed (>= 0).")] = 0,
) -> None:
    """Generate a Poisson trace and write it in the generic csv format."""

    def produce() -> str:
        workload = generate(
            rate_rps=rps,
            duration_s=duration,
            input_tokens=parse_distribution(in_tokens),
            output_tokens=parse_distribution(out_tokens),
            seed=seed,
        )
        write_csv(workload, out)
        return f"wrote {len(workload.frame):,} requests to {out} (seed {seed})\n"

    _run(produce)


@traces_app.command("fetch")
def fetch_command(
    name: Annotated[str, typer.Argument(help="Trace name from the shipped trace manifest.")],
    dest: Annotated[Path, typer.Option("--dest", help="Existing directory to save into.")],
    yes: Annotated[
        bool, typer.Option("--yes", help="Consent to download the file (required).")
    ] = False,
) -> None:
    """Download trace NAME into DEST and verify its SHA-256."""

    def produce() -> str:
        path = fetch_trace(name, dest, yes=yes)
        return f"saved {name} to {path} (SHA-256 verified)\n"

    _run(produce)
