"""Generic CSV trace format (M2_DESIGN.md section 4.1), key `csv`.

Columns: `arrival_s` (float seconds) or `timestamp` (ISO-8601), `input_tokens`,
`output_tokens`, and optionally `model`, `tenant`. `write_csv` emits this format, so
synthetic workloads round-trip through `load_workload`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llmplan.errors import WorkloadFormatError
from llmplan.workload.formats import register
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES, TimeKind, read_header, read_trace
from llmplan.workload.schema import Workload

_TOKENS = ("input_tokens", "output_tokens")
_TIMES = ("arrival_s", "timestamp")
_LABELS = ("model", "tenant")


@register("csv")
class GenericCsv:
    """The generic `csv` format."""

    def matches(self, header: Sequence[str], first_row: Sequence[str] | None) -> bool:
        columns = set(header)
        return columns.issuperset(_TOKENS) and not columns.isdisjoint(_TIMES)

    def parse(self, path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Workload:
        """Parse a generic CSV; raises `WorkloadFormatError` if both time columns exist."""
        header, _ = read_header(path)
        present = [c for c in _TIMES if c in header]
        if len(present) == 2:
            raise WorkloadFormatError(
                f"{path.name}: has both 'arrival_s' and 'timestamp' columns; keep exactly one"
            )
        if not present:
            raise WorkloadFormatError(f"{path.name}: needs an 'arrival_s' or 'timestamp' column")
        columns = {
            "time": present[0],
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
        }
        columns.update({label: label for label in _LABELS if label in header})
        kind: TimeKind = "seconds" if present[0] == "arrival_s" else "datetime"
        return read_trace(path, fmt="csv", columns=columns, time_kind=kind, max_bytes=max_bytes)


def write_csv(workload: Workload, path: Path) -> None:
    """Write `workload` to `path` in the generic `csv` format.

    Always writes `arrival_s`, `input_tokens`, `output_tokens`; writes `model` and `tenant`
    only when they hold at least one value. Floats use their shortest round-trip repr and
    lines end in `\\n`, so equal workloads produce byte-identical files.
    """
    frame = workload.frame
    columns = ["arrival_s", *_TOKENS, *(c for c in _LABELS if frame[c].notna().any())]
    frame.to_csv(path, columns=columns, index=False, lineterminator="\n")
