"""Timeline figure (M5_DESIGN.md section 2): four stacked panels as PNG bytes or a file.

Panels, all per window: demand against the plan's capacity (req/s); utilization per
replica; the fleet's VRAM split into weights, KV cache in use (max) and free (GB, M6
carry-over); and queue depth (max over replicas) with SLO violations, both counted in
requests. Uses matplotlib's Agg canvas directly (no pyplot, no display), so it runs
headless. Only this module imports matplotlib, and the package does not import it at
start-up.
"""

from __future__ import annotations

import io
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from llmplan.errors import ValidationError
from llmplan.simulate import Timeline, WindowRecord

# Reference categorical palette (fixed order, never cycled); text in ink tokens.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CRITICAL = "#e34948"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FREE = "#d9d8d4"
DPI = 100
GB = 1e9


def vram_split_gb(window: WindowRecord) -> tuple[float, float, float | None]:
    """Fleet VRAM in one window, in GB: (weights, KV in use (max), free).

    Sums over replicas. Free is total VRAM minus weights minus KV in use, floored at 0;
    None when any replica's `vram_bytes_total` is unknown.
    """
    weights = sum(r.weight_bytes for r in window.replicas)
    kv = sum(r.kv_bytes_in_use_max for r in window.replicas)
    totals = [r.vram_bytes_total for r in window.replicas]
    if any(t is None for t in totals):
        return weights / GB, kv / GB, None
    free = max(0, sum(t for t in totals if t is not None) - weights - kv)
    return weights / GB, kv / GB, free / GB


def render_png(timeline: Timeline) -> bytes:
    """Draw `timeline` as a four-panel PNG and return its bytes (no file is written).

    With up to eight replicas each gets its own utilization line; beyond that the panel
    shows the mean and maximum across replicas. The output carries no software metadata,
    so identical timelines give identical bytes on one matplotlib version.
    """
    windows = timeline.windows
    x = [w.start_s for w in windows] + [windows[-1].start_s + timeline.options.window_s]
    figure = Figure(figsize=(10, 10), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(figure)
    demand, util, vram, queue = figure.subplots(4, 1, sharex=True)
    for axes in (demand, util, vram, queue):
        _style(axes)

    _steps(demand, x, [w.demand_rps for w in windows], color=SERIES[0])
    _steps(demand, x, [w.capacity_rps for w in windows], color=INK_MUTED, ls="--")
    _legend(demand, ["demand", "plan capacity (derated)"])
    demand.set_ylim(bottom=0)
    demand.set_ylabel("req/s", color=INK_MUTED)

    n_replicas = len(windows[0].replicas)
    utilization = [[r.utilization for r in w.replicas] for w in windows]
    _per_replica(util, x, utilization, n_replicas)
    util.set_ylim(0, 1.05)
    util.set_ylabel("utilization", color=INK_MUTED)

    _vram(vram, x, [vram_split_gb(w) for w in windows])

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
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=SURFACE, metadata={"Software": None})
    return buffer.getvalue()


def save_png(timeline: Timeline, path: Path) -> None:
    """Write `render_png(timeline)` to `path` (overwritten if it exists).

    Raises `ValidationError` when `path` cannot be written.
    """
    try:
        path.write_bytes(render_png(timeline))
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


def _vram(axes: Axes, x: list[float], split: list[tuple[float, float, float | None]]) -> None:
    """Stacked weights / KV in use / free per window (free only when total VRAM is known)."""
    weights = [s[0] for s in split]
    kv_top = [s[0] + s[1] for s in split]
    layers = [([0.0] * len(split), weights, SERIES[0]), (weights, kv_top, SERIES[1])]
    labels = ["weights", "KV in use (max)"]
    if all(s[2] is not None for s in split):
        free_top = [top + (s[2] or 0.0) for top, s in zip(kv_top, split, strict=True)]
        layers.append((kv_top, free_top, FREE))
        labels.append("free")
    for low, high, color in layers:
        axes.fill_between(
            x, [*low, low[-1]], [*high, high[-1]], step="post", color=color, lw=0, alpha=0.9
        )
    _legend(axes, labels)
    axes.set_ylim(bottom=0)
    axes.set_ylabel("fleet VRAM (GB)", color=INK_MUTED)
