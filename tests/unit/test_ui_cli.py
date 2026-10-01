"""`llmplan ui`: starts Streamlit in-process on the bundled app with the section 6 flags."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from streamlit.web import bootstrap
from typer.testing import CliRunner

from llmplan.cli import app
from llmplan.cli_ui import APP_PATH, MAX_UPLOAD_MB

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()
ALL = "0.0.0.0"  # noqa: S104  # what the container binds; nothing is bound in this test


def test_ui_runs_the_bundled_app_with_the_upload_cap_and_no_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        bootstrap, "load_config_options", lambda flag_options: calls.append(("cfg", flag_options))
    )
    monkeypatch.setattr(bootstrap, "run", lambda *args: calls.append(("run", *args)))
    result = runner.invoke(app, ["ui", "--port", "8600", "--address", ALL, "--headless"])
    assert result.exit_code == 0, result.output
    flags = {
        "server_port": 8600,
        "server_address": ALL,
        "server_headless": True,
        "server_maxUploadSize": 50,
        "browser_gatherUsageStats": False,
    }
    assert calls == [("cfg", flags), ("run", str(APP_PATH), False, [], flags)]
    assert APP_PATH.is_file()


def test_ui_defaults_and_help() -> None:
    assert "ui" in runner.invoke(app, ["--help"]).stdout
    help_text = runner.invoke(app, ["ui", "--help"]).stdout
    assert "--port" in help_text
    assert "--headless" in help_text


def test_streamlit_config_file_matches_the_cli_flags() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == MAX_UPLOAD_MB
    assert config["browser"]["gatherUsageStats"] is False
