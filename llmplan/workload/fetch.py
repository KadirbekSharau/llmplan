"""Consented, checksum-verified download of public traces (M2_DESIGN.md section 7).

The shipped trace manifest (`llmplan/data/traces/manifest.yaml`) lists each downloadable
trace with its URL, SHA-256, size, and license. `fetch_trace` is the only code in
`llmplan.workload` that touches the network; the trace parsers only read local files.
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from collections.abc import Mapping
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from urllib.parse import urlsplit

import httpx
import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmplan.errors import CatalogError, FetchError, ValidationError
from llmplan.paths import TRACE_MANIFEST

log = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = TRACE_MANIFEST  # package data (llmplan/paths.py)
MAX_DOWNLOAD_BYTES = 2 * 2**30
_URL = r"^https://\S+$"
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class TraceSource(BaseModel):
    """One public trace file. `url` is None when the dataset forbids direct download."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    url: str | None = Field(pattern=_URL)
    sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(gt=0)
    as_of: date
    license_url: str = Field(pattern=_URL)


def load_manifest(path: Path | None = None) -> Mapping[str, TraceSource]:
    """Load the trace manifest (default: the shipped `traces/manifest.yaml`) keyed by name.

    Raises `CatalogError` naming the row index and field for any invalid row or duplicate.
    """
    path = path or DEFAULT_MANIFEST_PATH
    try:
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"cannot load trace manifest {path}: {exc}") from None
    if not isinstance(rows, list):
        raise CatalogError(f"{path.name}: expected a list of traces at top level")
    sources: dict[str, TraceSource] = {}
    for index, row in enumerate(rows):
        try:
            source = TraceSource.model_validate(row)
        except pydantic.ValidationError as exc:
            err = exc.errors()[0]
            field = ".".join(str(p) for p in err["loc"]) or "row"
            raise CatalogError(f"{path.name} row {index}: field {field!r}: {err['msg']}") from None
        if source.name in sources:
            raise CatalogError(f"{path.name} row {index}: duplicate trace name {source.name!r}")
        sources[source.name] = source
    return MappingProxyType(sources)


def _https_only(request: httpx.Request) -> None:
    """httpx request hook: every hop, including redirects, must use https."""
    if request.url.scheme != "https":
        raise FetchError(f"refusing non-https request to {request.url.host!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_trace(
    name: str,
    dest: Path,
    *,
    yes: bool,
    manifest: Mapping[str, TraceSource] | None = None,
    transport: httpx.BaseTransport | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    timeout_s: float = 60.0,
) -> Path:
    """Download trace `name` from the manifest into the existing directory `dest`.

    Refuses (`ValidationError`, before any network access) unless `yes` is True. Streams to
    a temporary file in `dest`, stops past `max_bytes` (default 2 GiB) or the manifest size,
    verifies the manifest SHA-256, then renames the file into place and returns its path.
    A file already at the target path is returned when its checksum matches and refused
    otherwise. Raises `FetchError` on HTTP errors, size overflow, or checksum mismatch; the
    temporary file is always removed on failure.
    """
    if not yes:
        raise ValidationError(f"downloading trace {name!r} needs explicit consent: pass --yes")
    sources = load_manifest() if manifest is None else manifest
    if name not in sources:
        raise ValidationError(f"unknown trace {name!r}; known: {', '.join(sorted(sources))}")
    source = sources[name]
    if source.url is None or source.sha256 is None:
        raise FetchError(f"trace {name!r} has no direct download URL and checksum; see its page")
    if not dest.is_dir():
        raise ValidationError(f"--dest {dest} is not an existing directory")
    filename = PurePosixPath(urlsplit(source.url).path).name
    if not _FILENAME_RE.fullmatch(filename) or filename.startswith("."):
        raise FetchError(f"trace {name!r}: unsafe file name {filename!r} in manifest URL")
    target = dest / filename
    if target.exists():
        if target.is_file() and _sha256_file(target) == source.sha256:
            return target
        raise FetchError(f"{target} already exists with other contents; remove it first")

    cap = min(max_bytes, source.size_bytes or max_bytes)
    log.info("downloading trace name=%s expected_bytes=%s", name, source.size_bytes)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115  closed by the with-block below
        dir=dest, prefix=f".{filename}.", suffix=".part", delete=False
    )
    temp = Path(handle.name)
    try:
        digest = hashlib.sha256()
        received = 0
        with (
            handle,
            httpx.Client(
                transport=transport,
                timeout=timeout_s,
                follow_redirects=True,
                event_hooks={"request": [_https_only]},
            ) as client,
            client.stream("GET", source.url) as response,
        ):
            if response.status_code != 200:
                raise FetchError(f"HTTP {response.status_code} downloading trace {name!r}")
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > cap:
                    raise FetchError(f"trace {name!r} download exceeds {cap} bytes")
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != source.sha256:
            raise FetchError(
                f"SHA-256 mismatch for trace {name!r}: expected {source.sha256}, "
                f"got {digest.hexdigest()}"
            )
        temp.replace(target)
    except httpx.HTTPError as exc:
        raise FetchError(f"downloading trace {name!r} failed: {type(exc).__name__}") from None
    finally:
        temp.unlink(missing_ok=True)
    log.info("saved trace name=%s bytes=%d", name, received)
    return target
