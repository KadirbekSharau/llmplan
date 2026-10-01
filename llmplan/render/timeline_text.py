"""Plain-text rendering of a `Timeline` (M5_DESIGN.md section 8).

A summary block, with request-size classes (M7) one line per class, a compact table with
one row per window (demand, completions, mean utilization over replicas, maximum queue
depth over replicas, violations), and the assumptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmplan.simulate import ClassSummary, Timeline, WindowRecord

LABEL_WIDTH = 10


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def _budget(budget_ms: float | None, violation_pct: float) -> str:
    if budget_ms is None:
        return "no budget"
    return f"budget {budget_ms:g} ms: {violation_pct:.2f}% violations"


def _replicas(n: int) -> str:
    return f"{n} replica" if n == 1 else f"{n} replicas"


def _row(w: WindowRecord) -> str:
    util = sum(r.utilization for r in w.replicas) / len(w.replicas)
    queue = max(r.queue_depth_max for r in w.replicas)
    return (
        f"  {w.index:>6}{w.start_s:>11,.0f}{w.demand_rps:>12,.3f}{w.completions:>13,}"
        f"{util * 100:>10.1f}%{queue:>11,}{w.ttft_violations:>11,}{w.tpot_violations:>11,}"
    )


def _class(c: ClassSummary) -> str:
    if c.ttft_ms_p95 is None:
        return _line(f"Class {c.class_index}", "0 requests")
    return _line(
        f"Class {c.class_index}",
        f"{c.n_requests:,} requests, TTFT p95 {c.ttft_ms_p95:,.1f} ms, "
        f"{c.ttft_violation_pct:.2f}% TTFT and {c.tpot_violation_pct:.2f}% TPOT violations",
    )


def timeline_text(timeline: Timeline) -> str:
    """Render `llmplan simulate` output as aligned plain text."""
    s, options = timeline.summary, timeline.options
    tpot_budget = _budget(options.tpot_budget_ms, s.tpot_violation_pct)
    lines = [
        _line("Requests", f"{s.n_requests:,} simulated ({s.n_truncated:,} truncated)"),
        _line(
            "TTFT",
            f"p50 {s.ttft_ms_p50:,.1f} ms, p95 {s.ttft_ms_p95:,.1f} ms  "
            f"({_budget(options.ttft_budget_ms, s.ttft_violation_pct)})",
        ),
        _line(
            "TPOT",
            f"p95 {s.tpot_ms_p95:,.2f} ms  ({tpot_budget})",
        ),
        _line("E2E", f"p95 {s.e2e_ms_p95:,.1f} ms"),
        _line(
            "Fleet",
            f"{_replicas(len(timeline.windows[0].replicas))}, mean utilization "
            f"{s.mean_utilization * 100:.1f}%, max queue depth {s.max_queue_depth:,}",
        ),
        _line("Cost", f"${timeline.plan_cost_usd_per_day:,.2f}/day"),
        _line(
            "Windows",
            f"{len(timeline.windows):,} x {options.window_s:g} s, routing {options.routing}",
        ),
        *(_class(c) for c in timeline.classes),
        "",
        f"  {'window':>6}{'start_s':>11}{'demand_rps':>12}{'completions':>13}{'util':>11}"
        f"{'queue_max':>11}{'ttft_viol':>11}{'tpot_viol':>11}",
        *(_row(w) for w in timeline.windows),
        "",
        "Assumptions",
        *(f"  - {note}" for note in timeline.assumptions),
    ]
    return "\n".join(lines) + "\n"
