"""Azure LLM inference traces (M2_DESIGN.md section 4.2), keys `azure2023` and `azure2024`.

Both releases share the header `TIMESTAMP,ContextTokens,GeneratedTokens` (verified against
the dataset files, see M2_NOTES.md). They differ in timestamp style: 2023 rows are naive
(`2023-11-16 18:17:03.9799600`), 2024 rows carry a UTC offset
(`2024-05-10 00:00:00.009930+00:00`); detection reads the first data row to tell them apart.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

from llmplan.workload.formats import register
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES, TraceSource, read_trace
from llmplan.workload.schema import Workload

_COLUMNS = {
    "time": "TIMESTAMP",
    "input_tokens": "ContextTokens",
    "output_tokens": "GeneratedTokens",
}
_UTC_OFFSET = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


class _AzureTrace:
    key: ClassVar[str]
    offset_timestamps: ClassVar[bool]

    def matches(self, header: Sequence[str], first_row: Sequence[str] | None) -> bool:
        if not set(_COLUMNS.values()).issubset(header):
            return False
        if first_row is None:  # no data row: both releases match, detect() reports ambiguity
            return True
        index = list(header).index(_COLUMNS["time"])
        stamp = first_row[index].strip() if index < len(first_row) else ""
        return bool(_UTC_OFFSET.search(stamp)) == self.offset_timestamps

    def parse(self, source: TraceSource, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Workload:
        """Parse an Azure trace; timestamps without an offset are read as UTC."""
        return read_trace(
            source, fmt=self.key, columns=_COLUMNS, time_kind="datetime", max_bytes=max_bytes
        )


@register("azure2023")
class Azure2023(_AzureTrace):
    """Azure LLM inference trace 2023 (Splitwise, ISCA 2024)."""

    key = "azure2023"
    offset_timestamps = False


@register("azure2024")
class Azure2024(_AzureTrace):
    """Azure LLM inference trace 2024 (DynamoLLM, HPCA 2025), one-week files."""

    key = "azure2024"
    offset_timestamps = True
