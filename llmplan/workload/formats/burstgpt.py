"""BurstGPT trace (M2_DESIGN.md section 4.3), key `burstgpt`.

Header verified against release v2.0 `BurstGPT_1.csv`:
`Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type` (newer files may add
`Session ID` and `Elapsed time`, which are ignored). `Timestamp` is seconds from the start
of the collection; `Model` maps to `model`.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmplan.workload.formats import register
from llmplan.workload.formats.reader import DEFAULT_MAX_BYTES, TraceSource, read_trace
from llmplan.workload.schema import Workload

_HEADER = frozenset(
    ("Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type")
)
_COLUMNS = {
    "time": "Timestamp",
    "model": "Model",
    "input_tokens": "Request tokens",
    "output_tokens": "Response tokens",
}


@register("burstgpt")
class BurstGPT:
    """The BurstGPT trace format."""

    def matches(self, header: Sequence[str], first_row: Sequence[str] | None) -> bool:
        return _HEADER.issubset(header)

    def parse(self, source: TraceSource, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Workload:
        """Parse a BurstGPT file; rows with zero response tokens are kept and counted."""
        workload = read_trace(
            source, fmt="burstgpt", columns=_COLUMNS, time_kind="seconds", max_bytes=max_bytes
        )
        zero = int((workload.frame["output_tokens"] == 0).sum())
        if zero == 0:
            return workload
        note = f"zero_output_rows={zero}: rows with 'Response tokens' == 0 kept (failed or empty)"
        return workload.model_copy(update={"notes": (*workload.notes, note)})
