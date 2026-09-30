"""Trace format registry (ARCHITECTURE.md section 6): `csv`, `azure2023`, `azure2024`, `burstgpt`.

Each format recognizes its header and parses a local file into a `Workload`. Adding a
format is a new module plus an import line at the bottom of this file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from llmplan.errors import UnknownRegistryKey, WorkloadFormatError
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES, read_header
from llmplan.workload.schema import Workload


class TraceFormat(Protocol):
    """Interface every trace format module implements."""

    def matches(self, header: Sequence[str], first_row: Sequence[str] | None) -> bool: ...

    def parse(self, path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Workload: ...


_REGISTRY: dict[str, TraceFormat] = {}


def register(key: str) -> Callable[[type[TraceFormat]], type[TraceFormat]]:
    """Class decorator: instantiate the class and register it under `key`."""

    def decorator(cls: type[TraceFormat]) -> type[TraceFormat]:
        _REGISTRY[key] = cls()
        return cls

    return decorator


def get(key: str) -> TraceFormat:
    """Return the trace format registered under `key`."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRegistryKey(f"unknown trace format {key!r}; known: {known}") from None


def keys() -> tuple[str, ...]:
    """Registered format keys, sorted."""
    return tuple(sorted(_REGISTRY))


def detect(path: Path) -> str:
    """Return the key of the one format whose header matches the file at `path`.

    Only the header row and the first data row are read. Raises `WorkloadFormatError` when
    no format or more than one format matches (pass the format explicitly then).
    """
    header, first_row = read_header(path)
    hits = [key for key, fmt in sorted(_REGISTRY.items()) if fmt.matches(header, first_row)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        shown = ", ".join(header[:8]) + (", ..." if len(header) > 8 else "")
        raise WorkloadFormatError(
            f"{path.name}: header ({shown}) matches no trace format ({', '.join(keys())}); "
            "pass the format explicitly"
        )
    raise WorkloadFormatError(
        f"{path.name}: header matches several formats ({', '.join(hits)}); "
        "pass the format explicitly"
    )


from llmplan.workload.formats import generic_csv  # noqa: E402

__all__ = ["TraceFormat", "detect", "generic_csv", "get", "keys", "register"]
