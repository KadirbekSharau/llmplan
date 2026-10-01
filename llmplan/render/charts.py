"""Interactive charts (M9_DESIGN.md section 3) as Altair (Vega-Lite) specs, drawn by the web
UI; Altair ships with Streamlit, so there is no new dependency. The interactive counterpart
of `render/plots.py` (same four timeline panels), imported only by the UI.

`timeline_chart` stacks four panels of a replay (demand against the plan's capacity,
replica utilization, the fleet's VRAM split, queue depth with latency-target violations)
on one shared time axis that zooms and pans together, with hover tooltips. Replays longer
than `MAX_WINDOWS` windows are thinned to every k-th window so the browser stays fast.
`routing_chart` shows each request-size class's routing weights as one stacked bar.
Values are read from the library's results; nothing is recomputed.
"""

from __future__ import annotations

import math
import warnings

import altair as alt
import pandas as pd

from llmplan.planner.result import PlanResult, label
from llmplan.render.plots import vram_split_gb
from llmplan.simulate import Timeline

MAX_WINDOWS = 500
PANELS = ("req/s", "utilization", "VRAM (GB)", "requests")


def timeline_frame(timeline: Timeline) -> pd.DataFrame:
    """The replay in long form, one row per window, panel and series: `minute` (window
    start), `panel`, `series`, `value`. Utilization is the mean and maximum over replicas;
    VRAM free is left out when a replica's total is unknown."""
    rows = []
    for w in timeline.windows[:: math.ceil(len(timeline.windows) / MAX_WINDOWS)]:
        utilization = [r.utilization for r in w.replicas]
        weights, kv, free = vram_split_gb(w)
        series = (
            ("demand", w.demand_rps),
            ("capacity (derated)", w.capacity_rps),
            ("mean", sum(utilization) / len(utilization)),
            ("max", max(utilization)),
            ("weights", weights),
            ("KV cache in use (max)", kv),
            ("free", free),
            ("queue depth (max)", max(r.queue_depth_max for r in w.replicas)),
            ("over the latency target", w.ttft_violations + w.tpot_violations),
        )
        panels = (0, 0, 1, 1, 2, 2, 2, 3, 3)
        rows += [
            {"minute": w.start_s / 60, "panel": PANELS[p], "series": name, "value": value}
            for p, (name, value) in zip(panels, series, strict=True)
            if value is not None
        ]
    return pd.DataFrame(rows)


def timeline_chart(timeline: Timeline) -> alt.VConcatChart:
    """Four stacked panels over a shared, zoomable minute axis (VRAM as stacked areas)."""
    data = timeline_frame(timeline)
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    domain = alt.Scale(domain=[0, float(data["minute"].max())], nice=False)
    panels = []
    for panel in PANELS:
        base = alt.Chart(data).transform_filter(alt.datum.panel == panel)
        stacked = panel == PANELS[2]
        mark = (
            base.mark_area(interpolate="step-after")
            if stacked
            else base.mark_line(interpolate="step-after", point=alt.OverlayMarkDef(opacity=0))
        )
        panels.append(
            mark.encode(
                x=alt.X("minute:Q", title="minutes since the first arrival", scale=domain),
                y=alt.Y("value:Q", title=panel, stack=True if stacked else None),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[
                    alt.Tooltip("minute:Q", format=",.1f"),
                    alt.Tooltip("series:N"),
                    alt.Tooltip("value:Q", format=",.3~f"),
                ],
            )
            .properties(height=130)
            .add_params(zoom)
        )
    with warnings.catch_warnings():  # Altair notes it merged the one zoom shared by all panels
        warnings.simplefilter("ignore", UserWarning)
        chart: alt.VConcatChart = alt.vconcat(*panels)
    shared: alt.VConcatChart = chart.resolve_scale(x="shared", color="independent")
    return shared  # annotated: Altair's API returns Any


def routing_chart(result: PlanResult) -> alt.Chart:
    """Each class's routing weights over the replica types, one horizontal stacked bar per
    class (weights sum to 1 per class)."""
    bounds = {
        c.index: f"class {c.index}: in {c.input_lo:,}-{c.input_hi:,}, out {c.output_lo:,}-"
        f"{c.output_hi:,}"
        for c in result.classes
    }
    data = pd.DataFrame(
        [
            {"class": bounds[r.class_index], "replica type": label(r.candidate), "weight": r.weight}
            for r in result.routing
        ]
    )
    chart: alt.Chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="share of the class's requests", axis=alt.Axis(format="%")),
            y=alt.Y("class:N", title=None),
            color=alt.Color("replica type:N", legend=alt.Legend(orient="bottom", columns=1)),
            tooltip=["class:N", "replica type:N", alt.Tooltip("weight:Q", format=".1%")],
        )
    )
    return chart
