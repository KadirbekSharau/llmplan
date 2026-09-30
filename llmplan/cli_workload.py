"""`llmplan workload` and `llmplan traces` subcommands (M2): thin adapters over `llmplan.workload`.

Registered on the main app in `llmplan.cli`. Errors go through the main app's handler, so
exit codes match ARCHITECTURE.md section 7. The two stats renderers live here rather than
in `llmplan.render` to keep M2 out of the renderer registry while M3 extends it in parallel
(TODO(M4): move them into `render/` next to the plan renderers).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from llmplan.workload import Workload, WorkloadStats, compute_stats, generate, load_workload
from llmplan.workload.fetch import fetch_trace
from llmplan.workload.formats.generic_csv import write_csv
from llmplan.workload.synth import parse_distribution

workload_app = typer.Typer(help="Characterize and synthesize request traces.", no_args_is_help=True)
traces_app = typer.Typer(help="Download public request traces.", no_args_is_help=True)

TraceFormatOpt = Literal["auto", "csv", "azure2023", "azure2024", "burstgpt"]
LABEL_WIDTH = 16


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # lazy: llmplan.cli imports this module

    run(produce)


def _num(value: float, digits: int = 2) -> str:
    text = f"{value:,.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def _tokens(stats: WorkloadStats, kind: Literal["input", "output"]) -> str:
    p50, p95, p99, mean, top = (
        getattr(stats, f"{kind}_tokens_{part}") for part in ("p50", "p95", "p99", "mean", "max")
    )
    return f"p50 {_num(p50)}  p95 {_num(p95)}  p99 {_num(p99)}  mean {_num(mean)}  max {top:,}"


def stats_text(workload: Workload, stats: WorkloadStats) -> str:
    """Render `llmplan workload stats` as aligned plain text."""
    lines = [
        _line("trace", f"{workload.source}  (format {workload.format})"),
        _line("requests", f"{stats.n_requests:,}  ({workload.dropped_rows:,} rows dropped)"),
        _line("duration", f"{_num(stats.duration_s, 3)} s"),
        _line("windows", f"{stats.n_windows:,} x {_num(stats.window_s, 3)} s"),
        _line("mean rate", f"{_num(stats.mean_rps, 3)} req/s"),
        _line(
            "peak rate",
            f"{_num(stats.peak_window_rps, 3)} req/s  (window {stats.peak_window_index})",
        ),
        _line("input tokens", _tokens(stats, "input")),
        _line("output tokens", _tokens(stats, "output")),
        _line("peak input", f"{_num(stats.peak_input_tokens_per_s)} tokens/s"),
        _line("peak output", f"{_num(stats.peak_output_tokens_per_s)} tokens/s"),
    ]
    if stats.hourly_rps is None:
        lines.append(_line("hourly req/s", "- (trace covers fewer than 24 hour windows)"))
    else:
        for start in range(0, 24, 6):
            values = "  ".join(_num(v, 3) for v in stats.hourly_rps[start : start + 6])
            label = "hourly req/s" if start == 0 else ""
            lines.append(_line(label, f"{start:02d}-{start + 5:02d}h  {values}"))
    lines.extend(_line("note" if i == 0 else "", n) for i, n in enumerate(workload.notes))
    return "\n".join(lines) + "\n"


def stats_json(workload: Workload, stats: WorkloadStats) -> str:
    """Render `llmplan workload stats` as JSON; byte-identical for identical inputs."""
    doc = {
        "source": workload.source,
        "format": workload.format,
        "dropped_rows": workload.dropped_rows,
        "notes": list(workload.notes),
        "stats": stats.model_dump(mode="json"),
    }
    return json.dumps(doc, indent=2) + "\n"


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
        render = stats_json if format_out == "json" else stats_text
        return render(workload, stats)

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
    name: Annotated[str, typer.Argument(help="Trace name from data/traces/manifest.yaml.")],
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
