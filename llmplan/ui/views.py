"""Result views of the web UI (M6_DESIGN.md section 3, main area).

Each function renders one section from library results; values come from the plan,
the timeline, and the existing renderers (`render.get("json")`, `serve_command`,
`render_png`). Nothing is computed here that the library does not already expose.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from llmplan import render
from llmplan.planner.result import CandidateEval, PlanResult, label
from llmplan.render.plots import render_png
from llmplan.render.timeline_json import timeline_json
from llmplan.render.vllm_cmd import serve_command
from llmplan.ui import state
from llmplan.workload import Workload, WorkloadStats

TOP_CANDIDATES = 15
REJECTED_STYLE = "color: #8a8986"
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!$|<>~])")


def escape(text: str) -> str:
    """Backslash-escape Markdown (and `$` math) so library text renders verbatim."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def stats_caption(workload: Workload, stats: WorkloadStats) -> None:
    """One-line summary of a traffic source (format, size, peak, token means)."""
    dropped = f", {workload.dropped_rows:,} rows dropped" if workload.dropped_rows else ""
    st.caption(
        escape(
            f"{workload.format} format, {stats.n_requests:,} requests over "
            f"{stats.duration_s / 60:,.1f} min{dropped}; peak {stats.peak_window_rps:,.2f} "
            f"req/s (60 s window); mean tokens in {stats.input_tokens_mean:,.0f} / out "
            f"{stats.output_tokens_mean:,.0f}"
        )
    )
    for note in workload.notes:
        st.caption(escape(note))


def answer_card(result: PlanResult) -> None:
    """Cost, baseline, saving, binding constraint, and solver status."""
    baseline = None if result.baseline is None else result.baseline.cost_usd_per_day
    saving = result.baseline_saving_pct
    columns = st.columns(3)
    columns[0].metric("Cost per day", _usd(result.cost_usd_per_day))
    columns[1].metric("Baseline per day", _usd(baseline), help="Best single-row fleet")
    columns[2].metric("Saving", "n/a" if saving is None else f"{saving:.1f}%")
    st.markdown(
        f"Binding constraint: **{escape(result.binding)}** · Solver: "
        f"**{escape(result.solver.backend)} {escape(result.solver.status)}**"
    )
    st.caption(
        escape(
            f"Peak demand {result.demand_rps:,.2f} req/s and "
            f"{result.demand_output_tokens_per_s:,.0f} output tokens/s; fleet capacity "
            f"{result.capacity_rps:,.2f} req/s and {result.capacity_output_tokens_per_s:,.0f} "
            "output tokens/s after derating."
        )
    )


def fleet_and_replicas(run: state.PlanRun) -> None:
    """Fleet table, replica table, and one copyable `vllm serve` line per replica plan."""
    result = run.result
    st.subheader("Fleet")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "provider": item.price_row.provider,
                    "instance": item.price_row.instance,
                    "GPUs per instance": f"{item.price_row.gpu_count} x {item.price_row.gpu_id}",
                    "instances": item.instances,
                    "$/hour each": item.price_row.price_usd_per_hour,
                    "$/day": round(item.usd_per_day, 2),
                    "price as of": item.price_row.as_of.isoformat(),
                }
                for item in result.fleet
            ]
        ),
        hide_index=True,
    )
    st.subheader("Replicas")
    rows = []
    for replica in result.replicas:
        c, perf = replica.candidate, replica.candidate.perf
        rows.append(
            {
                "replicas": replica.count,
                "on": f"{c.price_row.provider} {c.price_row.instance}",
                "tp": c.config.tensor_parallel,
                "dtype": c.config.dtype,
                "max_num_seqs": c.config.max_num_seqs,
                "max_model_len": c.config.max_model_len,
                "perf": "-" if perf is None else f"{perf.backend}/{perf.confidence}",
                "TTFT p95 ms": None if perf is None else round(perf.ttft_ms_p95, 1),
                "TPOT p95 ms": None if perf is None else round(perf.tpot_ms_p95, 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for replica in result.replicas:
        row = replica.candidate.price_row
        st.caption(
            escape(
                f"{replica.count} replica(s) on {replica.instances} x {row.provider} "
                f"{row.instance} ({row.gpu_count} x {row.gpu_id} per instance)"
            )
        )
        st.code(serve_command(run.request.model, replica.candidate.config), language="bash")


def _saving(run: state.PlanRun) -> str:
    saving, base = run.class_saving_pct, run.single_class_cost_usd_per_day
    if saving is None or base is None:
        return "n/a (no fleet without classes meets the target)"
    return f"{saving:.1f}% (sized for the mean request: {_usd(base)}/day)"


def request_size_routing(run: state.PlanRun) -> None:
    """With request-size classes (M7): the saving against a single-class plan, the class
    table with each class's replayed TTFT, and the routing weights."""
    result = run.result
    if not result.classes:
        return
    st.subheader("Request-size routing")
    st.markdown(f"Saving from request-size routing: **{escape(_saving(run))}**")
    st.caption(
        "Each class is sized at its own request shape, so the saving can be negative when the "
        "mean request understates the long ones."
    )
    replayed = {c.class_index: c for c in run.timeline.classes}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "class": c.index,
                    "input tokens": f"{c.input_lo:,}..{c.input_hi:,}",
                    "output tokens": f"{c.output_lo:,}..{c.output_hi:,}",
                    "share": f"{c.share * 100:.1f}%",
                    "peak req/s": round(c.peak_rps, 3),
                    "binding": binding,
                    "replayed TTFT p95 ms": replayed[c.index].ttft_ms_p95,
                    "TTFT violations": f"{replayed[c.index].ttft_violation_pct:.1f}%",
                }
                for c, binding in zip(result.classes, result.class_binding, strict=True)
            ]
        ),
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "class": r.class_index,
                    "weight": f"{r.weight * 100:.1f}%",
                    "replica type": label(r.candidate),
                    "replica-equivalents": round(r.replicas, 3),
                    "req/s": round(r.capacity_rps, 2),
                }
                for r in result.routing
            ]
        ),
        hide_index=True,
    )


def timeline(run: state.PlanRun) -> None:
    """The four-panel timeline figure of the replay (M5 renderer, PNG bytes)."""
    summary, options = run.timeline.summary, run.timeline.options
    st.subheader("Timeline")
    st.image(
        render_png(run.timeline),
        caption=(
            f"Replay of {summary.n_requests:,} requests in {len(run.timeline.windows)} windows "
            f"of {options.window_s:g} s: TTFT p95 {summary.ttft_ms_p95:,.0f} ms (queueing "
            f"included), {summary.ttft_violation_pct:.1f}% over the TTFT target, mean "
            f"utilization {summary.mean_utilization * 100:.0f}%."
        ),
    )


def _candidate_row(c: CandidateEval) -> dict[str, object]:
    return {
        "status": c.status,
        "$/hour per req/s": None if c.usd_per_hour_per_rps is None else c.usd_per_hour_per_rps,
        "provider": c.price_row.provider,
        "instance": c.price_row.instance,
        "GPU": c.price_row.gpu_id,
        "tp": c.config.tensor_parallel,
        "dtype": c.config.dtype,
        "max_num_seqs": c.config.max_num_seqs,
        "reason": c.reason,
    }


def candidates(result: PlanResult) -> None:
    """The top 15 eligible candidates by $/hour per req/s, then the rejected ones greyed."""
    eligible = [c for c in result.candidates if c.status == "eligible"]
    rejected = [c for c in result.candidates if c.status != "eligible"]
    shown = eligible[:TOP_CANDIDATES] + rejected
    st.subheader("Candidates")
    st.caption(
        f"Top {min(len(eligible), TOP_CANDIDATES)} of {len(eligible)} eligible by $/hour per "
        f"req/s, then {len(rejected)} rejected (greyed) with the reason."
    )
    frame = pd.DataFrame([_candidate_row(c) for c in shown])
    styled = frame.style.apply(
        lambda row: [REJECTED_STYLE if row["status"] != "eligible" else ""] * len(row), axis=1
    )
    st.dataframe(styled, hide_index=True)


def assumptions(run: state.PlanRun) -> None:
    """Every assumption string of the plan, the perf estimates and the replay, verbatim."""
    st.subheader("Assumptions")
    for title, notes in state.assumptions(run).items():
        st.markdown(f"**{escape(title)}**")
        st.markdown("\n".join(f"- {escape(note)}" for note in notes) or "- none")


def downloads(run: state.PlanRun) -> None:
    """The plan JSON (`llmplan plan --format json` layout) and the timeline JSON."""
    left, right = st.columns(2)
    left.download_button(
        "Download plan JSON",
        data=render.get("json").plan(run.request, run.result),
        file_name="llmplan-plan.json",
        mime="application/json",
    )
    right.download_button(
        "Download timeline JSON",
        data=timeline_json(run.timeline),
        file_name="llmplan-timeline.json",
        mime="application/json",
    )


def results(run: state.PlanRun) -> None:
    """The whole main area after a successful plan."""
    answer_card(run.result)
    fleet_and_replicas(run)
    request_size_routing(run)
    timeline(run)
    candidates(run.result)
    assumptions(run)
    downloads(run)
