"""`llmplan traces` subcommands (M2): thin typer adapters over `llmplan.workload`.

Registered on the main app in `llmplan.cli`. Errors go through the main app's handler, so
exit codes match ARCHITECTURE.md section 7.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from llmplan.workload.fetch import fetch_trace

traces_app = typer.Typer(help="Download public request traces.", no_args_is_help=True)


def _run(produce: Callable[[], str]) -> None:
    from llmplan.cli import _run as run  # lazy: llmplan.cli imports this module

    run(produce)


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
