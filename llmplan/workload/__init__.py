"""Workload ingestion and characterization (M2): traces in, `Workload` and `WorkloadStats` out.

Public API: `load_workload` (ARCHITECTURE.md section 5), `compute_stats`, `generate`, and
the models `Workload`, `WorkloadStats`, `Distribution`.
"""

from __future__ import annotations

from pathlib import Path

from llmplan.errors import ValidationError
from llmplan.workload import formats
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES
from llmplan.workload.schema import Distribution, Workload, WorkloadStats
from llmplan.workload.stats import compute_stats
from llmplan.workload.synth import generate


def load_workload(
    source: str | Path, *, format: str | None = None, max_bytes: int = DEFAULT_MAX_BYTES
) -> Workload:
    """Parse the local trace file `source` into a validated `Workload`.

    `format` is a registry key (`csv`, `azure2023`, `azure2024`, `burstgpt`); None detects
    it from the header. Files larger than `max_bytes` (default 2 GiB) are rejected with
    `ValidationError`. Nothing is downloaded. Raises `UnknownRegistryKey` for an unknown
    format and `WorkloadFormatError` for a file that does not match its format.
    """
    if max_bytes <= 0:
        raise ValidationError(f"max_bytes must be positive, got {max_bytes}")
    path = Path(source)
    key = formats.detect(path) if format is None else format
    return formats.get(key).parse(path, max_bytes=max_bytes)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "Distribution",
    "Workload",
    "WorkloadStats",
    "compute_stats",
    "generate",
    "load_workload",
]
