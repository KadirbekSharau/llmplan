"""Plain-text rendering of `llmplan workload stats` (M2_DESIGN.md section 8).

Moved here from `llmplan/cli_workload.py` in M4; the output is unchanged byte for byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from llmplan.workload import Workload, WorkloadStats

LABEL_WIDTH = 16


def _num(value: float, digits: int = 2) -> str:
    text = f"{value:,.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def _tokens(stats: WorkloadStats, kind: Literal["input", "output"]) -> str:
    p50, p95, p99, mean, top = (
        getattr(stats, f"{kind}_tokens_{part}") for part in ("p50", "p95", "p99", "mean", "max")
    )
    return f"p50 {_num(p50)}  p95 {_num(p95)}  p99 {_num(p99)}  mean {_num(mean)}  max {top:,}"


def workload_stats_text(workload: Workload, stats: WorkloadStats) -> str:
    """Render `llmplan workload stats` as aligned plain text."""
    lines = [
        _line("trace", f"{workload.source}  (format {workload.format})"),
        _line("requests", f"{stats.n_requests:,}  ({workload.dropped_rows:,} rows dropped)"),
        _line("duration", f"{_num(stats.duration_s, 3)} s"),
        _line("windows", f"{stats.n_windows:,} x {_num(stats.window_s, 3)} s"),
        _line("mean rate", f"{_num(stats.mean_rps, 3)} req/s"),
        _line(
            "peak rate",
            f"{_num(stats.peak_window_rps, 3)} req/s  (window {stats.peak_window_index})",
        ),
        _line("input tokens", _tokens(stats, "input")),
        _line("output tokens", _tokens(stats, "output")),
        _line("peak input", f"{_num(stats.peak_input_tokens_per_s)} tokens/s"),
        _line("peak output", f"{_num(stats.peak_output_tokens_per_s)} tokens/s"),
    ]
    if stats.hourly_rps is None:
        lines.append(_line("hourly req/s", "- (trace covers fewer than 24 hour windows)"))
    else:
        for start in range(0, 24, 6):
            values = "  ".join(_num(v, 3) for v in stats.hourly_rps[start : start + 6])
            label = "hourly req/s" if start == 0 else ""
            lines.append(_line(label, f"{start:02d}-{start + 5:02d}h  {values}"))
    lines.extend(_line("note" if i == 0 else "", n) for i, n in enumerate(workload.notes))
    return "\n".join(lines) + "\n"
