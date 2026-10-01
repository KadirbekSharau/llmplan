"""`llmplan ui`: start the Streamlit web UI (M6_DESIGN.md section 2) in this process.

Registered on the main app in `llmplan.cli`. Streamlit is imported inside the command only,
so `llmplan --help` and every other command never load it. The server runs with the
section 6 upload cap (50 MB) and Streamlit's own usage telemetry turned off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

APP_PATH = Path(__file__).resolve().parent / "ui" / "app.py"
MAX_UPLOAD_MB = 50  # Streamlit's server-side cap; the app checks 50,000,000 bytes itself


def server_flags(*, port: int, address: str, headless: bool) -> dict[str, Any]:
    """Streamlit config options for `llmplan ui`, in `streamlit run` flag form."""
    return {
        "server_port": port,
        "server_address": address,
        "server_headless": headless,
        "server_maxUploadSize": MAX_UPLOAD_MB,
        "browser_gatherUsageStats": False,
    }


def ui_command(
    port: Annotated[int, typer.Option("--port", help="Port to serve on.")] = 8501,
    address: Annotated[
        str, typer.Option("--address", help="Address to bind (0.0.0.0 in a container).")
    ] = "localhost",
    headless: Annotated[
        bool, typer.Option("--headless/--browser", help="Do not open a browser window.")
    ] = False,
) -> None:
    """Start the web UI (Streamlit) on PORT; stop it with Ctrl+C."""
    from streamlit.web import bootstrap  # deferred: only this command needs Streamlit

    flags = server_flags(port=port, address=address, headless=headless)
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(str(APP_PATH), False, [], flags)
