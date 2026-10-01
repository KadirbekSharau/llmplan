"""Workload ingestion and characterization (M2): traces in, `Workload` and `WorkloadStats` out.

Public API: `load_workload` (ARCHITECTURE.md section 5), `compute_stats`, `generate`, and
the models `Workload`, `WorkloadStats`, `Distribution`, plus `InMemoryTrace` for traces
that must not touch the disk (M6 web uploads), and request-size classes (M7): `classify`,
`DemandClass`.
"""

from __future__ import annotations

from pathlib import Path

from llmplan.errors import ValidationError
from llmplan.workload import formats
from llmplan.workload.classes import DemandClass, classify
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES, InMemoryTrace
from llmplan.workload.schema import Distribution, Workload, WorkloadStats
from llmplan.workload.stats import compute_stats
from llmplan.workload.synth import generate


def load_workload(
    source: str | Path | InMemoryTrace,
    *,
    format: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Workload:
    """Parse the local trace file `source` (or an `InMemoryTrace`) into a validated `Workload`.

    `format` is a registry key (`csv`, `azure2023`, `azure2024`, `burstgpt`); None detects
    it from the header. Traces larger than `max_bytes` (default 2 GiB) are rejected with
    `ValidationError`. Nothing is downloaded, and an `InMemoryTrace` is parsed from memory
    without being written anywhere. Raises `UnknownRegistryKey` for an unknown format and
    `WorkloadFormatError` for a trace that does not match its format.
    """
    if max_bytes <= 0:
        raise ValidationError(f"max_bytes must be positive, got {max_bytes}")
    trace = source if isinstance(source, InMemoryTrace) else Path(source)
    key = formats.detect(trace) if format is None else format
    return formats.get(key).parse(trace, max_bytes=max_bytes)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DemandClass",
    "Distribution",
    "InMemoryTrace",
    "Workload",
    "WorkloadStats",
    "classify",
    "compute_stats",
    "generate",
    "load_workload",
]
