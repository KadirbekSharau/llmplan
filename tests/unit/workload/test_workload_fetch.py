from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from llmplan import cli_workload
from llmplan.cli import app
from llmplan.errors import CatalogError, FetchError, ValidationError
from llmplan.workload.fetch import TraceSource, fetch_trace, load_manifest

PAYLOAD = b"arrival_s,input_tokens,output_tokens\n0,1,1\n"
URL = "https://example.org/files/trace.csv"


def source(**overrides: Any) -> dict[str, TraceSource]:
    fields: dict[str, Any] = {
        "name": "demo",
        "url": URL,
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "size_bytes": len(PAYLOAD),
        "as_of": date(2026, 9, 30),
        "license_url": "https://example.org/license",
    }
    fields.update(overrides)
    return {"demo": TraceSource(**fields)}


def serving(content: bytes = PAYLOAD, status: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status, content=content))


def failing(request: httpx.Request) -> httpx.Response:
    raise AssertionError("no request expected")


def test_shipped_manifest_is_complete() -> None:
    manifest = load_manifest()
    assert set(manifest) == {
        "azure2023-code",
        "azure2023-conv",
        "azure2024-code",
        "azure2024-conv",
        "burstgpt-1",
    }
    for entry in manifest.values():
        assert entry.url is not None and entry.url.endswith(".csv")
        assert entry.sha256 is not None and entry.size_bytes is not None
        assert entry.size_bytes < 2 * 2**30


def test_successful_download(tmp_path: Path) -> None:
    path = fetch_trace("demo", tmp_path, yes=True, manifest=source(), transport=serving())
    assert path == tmp_path / "trace.csv"
    assert path.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [path]


def test_existing_identical_file_is_reused(tmp_path: Path) -> None:
    (tmp_path / "trace.csv").write_bytes(PAYLOAD)
    transport = httpx.MockTransport(failing)
    path = fetch_trace("demo", tmp_path, yes=True, manifest=source(), transport=transport)
    assert path.read_bytes() == PAYLOAD


def test_existing_different_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "trace.csv").write_bytes(b"mine")
    with pytest.raises(FetchError, match="already exists"):
        fetch_trace("demo", tmp_path, yes=True, manifest=source(), transport=serving())
    assert (tmp_path / "trace.csv").read_bytes() == b"mine"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"yes": False}, ValidationError, "--yes"),
        ({"name": "other"}, ValidationError, "unknown trace 'other'; known: demo"),
        ({"manifest": source(url=None)}, FetchError, "no direct download URL"),
        ({"manifest": source(sha256=None)}, FetchError, "no direct download URL"),
        ({"manifest": source(url="https://example.org/")}, FetchError, "unsafe file name"),
        ({"manifest": source(url="https://example.org/.hidden")}, FetchError, "unsafe"),
    ],
)
def test_refusals_before_any_request(
    tmp_path: Path, kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    args: dict[str, Any] = {"name": "demo", "yes": True, "manifest": source()}
    args.update(kwargs)
    name = args.pop("name")
    with pytest.raises(error, match=message):
        fetch_trace(name, tmp_path, transport=httpx.MockTransport(failing), **args)
    assert list(tmp_path.iterdir()) == []


def test_dest_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not an existing directory"):
        fetch_trace("demo", tmp_path / "missing", yes=True, manifest=source())


@pytest.mark.parametrize(
    ("transport", "kwargs", "message"),
    [
        (serving(status=404), {}, "HTTP 404"),
        (serving(PAYLOAD + b"x"), {}, f"exceeds {len(PAYLOAD)} bytes"),
        (serving(), {"max_bytes": 10}, "exceeds 10 bytes"),
        (
            httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))),
            {},
            "failed: ConnectError",
        ),
        (
            httpx.MockTransport(
                lambda r: httpx.Response(302, headers={"Location": "http://example.org/x.csv"})
            ),
            {},
            "non-https",
        ),
    ],
)
def test_download_failures_leave_no_file(
    tmp_path: Path, transport: httpx.MockTransport, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(FetchError, match=message):
        fetch_trace("demo", tmp_path, yes=True, manifest=source(), transport=transport, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- name: [unclosed\n", "cannot load"),
        ("name: x\n", "expected a list"),
        ("- name: x\n", "row 0: field 'url'"),
    ],
)
def test_bad_manifests(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(text)
    with pytest.raises(CatalogError, match=message):
        load_manifest(path)


def test_duplicate_manifest_names(tmp_path: Path) -> None:
    row = (
        "- {name: a, url: null, sha256: null, size_bytes: null, as_of: 2026-09-30, "
        "license_url: 'https://example.org/l'}\n"
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(row * 2)
    with pytest.raises(CatalogError, match="row 1: duplicate trace name 'a'"):
        load_manifest(path)


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="cannot load"):
        load_manifest(tmp_path / "absent.yaml")


def test_cli_fetch_reports_saved_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, bool]] = []

    def fake(name: str, dest: Path, *, yes: bool) -> Path:
        calls.append((name, dest, yes))
        return dest / "file.csv"

    monkeypatch.setattr(cli_workload, "fetch_trace", fake)
    args = ["traces", "fetch", "burstgpt-1", "--dest", str(tmp_path), "--yes"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout == f"saved burstgpt-1 to {tmp_path / 'file.csv'} (SHA-256 verified)\n"
    assert calls == [("burstgpt-1", tmp_path, True)]


def test_cli_fetch_unknown_name_exits_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["traces", "fetch", "nope", "--dest", str(tmp_path), "--yes"])
    assert result.exit_code == 2
    assert "unknown trace 'nope'" in result.stderr
