"""Timeline figure (M5_DESIGN.md section 2): four stacked panels saved as a PNG.

Panels, all per window: demand against the plan's capacity (req/s); utilization per
replica; KV cache in use per replica (max, GB); and queue depth (max over replicas) with
SLO violations, both counted in requests. Uses matplotlib's Agg canvas directly (no
pyplot, no display), so it runs headless. Only this module imports matplotlib, and the
package does not import it at start-up.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from llmplan.errors import ValidationError
from llmplan.simulate import Timeline

# Reference categorical palette (fixed order, never cycled); text in ink tokens.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CRITICAL = "#e34948"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
DPI = 100


def save_png(timeline: Timeline, path: Path) -> None:
    """Draw `timeline` as a four-panel PNG at `path` (overwritten if it exists).

    With up to eight replicas each gets its own line; beyond that the utilization and KV
    panels show the mean and maximum across replicas. Raises `ValidationError` when
    `path` cannot be written.
    """
    windows = timeline.windows
    x = [w.start_s for w in windows] + [windows[-1].start_s + timeline.options.window_s]
    figure = Figure(figsize=(10, 10), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(figure)
    demand, util, kv, queue = figure.subplots(4, 1, sharex=True)
    for axes in (demand, util, kv, queue):
        _style(axes)

    _steps(demand, x, [w.demand_rps for w in windows], color=SERIES[0])
    _steps(demand, x, [w.capacity_rps for w in windows], color=INK_MUTED, ls="--")
    _legend(demand, ["demand", "plan capacity (derated)"])
    demand.set_ylim(bottom=0)
    demand.set_ylabel("req/s", color=INK_MUTED)

    n_replicas = len(windows[0].replicas)
    utilization = [[r.utilization for r in w.replicas] for w in windows]
    kv_gb = [[r.kv_bytes_in_use_max / 1e9 for r in w.replicas] for w in windows]
    _per_replica(util, x, utilization, n_replicas)
    util.set_ylim(0, 1.05)
    util.set_ylabel("utilization", color=INK_MUTED)
    _per_replica(kv, x, kv_gb, n_replicas)
    kv.set_ylim(bottom=0)
    kv.set_ylabel("KV in use, max (GB)", color=INK_MUTED)

    _steps(queue, x, [max(r.queue_depth_max for r in w.replicas) for w in windows], color=SERIES[0])
    violations = [w.ttft_violations + w.tpot_violations for w in windows]
    width = timeline.options.window_s * 0.9
    queue.bar(x[:-1], violations, width=width, align="edge", color=CRITICAL, alpha=0.6)
    _legend(queue, ["queue depth (max)", "SLO violations (TTFT + TPOT)"])
    queue.set_ylim(bottom=0)
    queue.set_ylabel("requests", color=INK_MUTED)
    queue.set_xlabel("seconds since first arrival", color=INK_MUTED)

    summary = timeline.summary
    figure.suptitle(
        f"{summary.n_requests:,} requests, TTFT p95 {summary.ttft_ms_p95:,.0f} ms, "
        f"{summary.ttft_violation_pct:.1f}% TTFT violations, "
        f"${timeline.plan_cost_usd_per_day:,.2f}/day",
        color=INK,
    )
    try:
        figure.savefig(path, format="png", facecolor=SURFACE, metadata={"Software": None})
    except OSError as exc:
        raise ValidationError(f"cannot write PNG {path}: {exc.strerror}") from None


def _style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, lw=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_MUTED)


def _legend(axes: Axes, labels: list[str]) -> None:
    """Legend in one row above the panel, clear of the data."""
    axes.legend(labels, frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncols=4)


def _steps(
    axes: Axes, edges: list[float], values: list[float], *, color: str, ls: str = "-"
) -> None:
    """One value per window, drawn flat across the window (edges has one more entry)."""
    axes.step(edges, [*values, values[-1]], where="post", lw=2, color=color, linestyle=ls)


def _per_replica(axes: Axes, x: list[float], rows: list[list[float]], n_replicas: int) -> None:
    if n_replicas <= len(SERIES):
        for r in range(n_replicas):
            _steps(axes, x, [row[r] for row in rows], color=SERIES[r])
        _legend(axes, [f"replica {r}" for r in range(n_replicas)])
        return
    _steps(axes, x, [sum(row) / len(row) for row in rows], color=SERIES[0])
    _steps(axes, x, [max(row) for row in rows], color=SERIES[1])
    _legend(axes, [f"mean of {n_replicas} replicas", "max"])
