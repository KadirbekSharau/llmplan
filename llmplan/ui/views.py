"""Result views of the web UI (M6 section 3; M9: header and footer, the answer card, five
result tabs).

Each function renders one part from library results: the plan, the timeline, the existing
renderers (`render.get("json")`, `serve_command`, `render_png`), `charts` for the
interactive charts and `wording` for numbers and sentences. Nothing is computed here that
the library does not already expose. Tables show five key columns unless "Show all
columns" is on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import streamlit as st

from llmplan import __version__, render
from llmplan.perf.confidence import ROOFLINE_CONSTANTS, confidence_sentence
from llmplan.perf.contribute import REPOSITORY_URL
from llmplan.planner.result import CandidateEval, PlanResult
from llmplan.render.plan_text import baseline_saving, class_comparison_sentence
from llmplan.render.plots import render_png
from llmplan.render.timeline_json import timeline_json
from llmplan.render.vllm_cmd import serve_command
from llmplan.simulate import Timeline
from llmplan.ui import calibrate, charts, state, usage_log, wording
from llmplan.workload import Workload, WorkloadStats

PAGE_TITLE = "llmplan — GPU fleet planner for LLM inference"
FAVICON = (  # inline SVG: three bars of a fleet, in the theme's primary colour
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" '
    'rx="3" fill="#1F6FEB"/><path d="M4 13V7M8 13V3M12 13V9" stroke="#fff" stroke-width="2"/></svg>'
)
LEGEND = (
    "How far to trust the performance estimates: **measured** (a published or uploaded "
    "benchmark row), **interpolated** (between measured rows), **roofline** (first-principles "
    "model, about ±30% on throughput)."
)
REPLICA_HELP = "A replica is one vLLM server holding one copy of the model; with tensor "
REPLICA_HELP += "parallelism it spans several GPUs of one instance."
BADGE_COLORS = {"measured": "green", "interpolated": "blue", "roofline": "orange"}
TOP_CANDIDATES = 15
KEY_COLUMNS = 5
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!$|<>~])")


def escape(text: str) -> str:
    """Backslash-escape Markdown (and `$` math) so library text renders verbatim."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def header() -> None:
    """Product name, one-line promise, version, GitHub link, and the confidence legend (the
    help icon next to the title)."""
    st.title("llmplan", help=LEGEND)
    st.caption(
        "The cheapest GPU fleet and vLLM settings that meet your latency target, proven by "
        f"replaying your traffic. CPU-only: nothing connects to a GPU or your cluster. "
        f"v{__version__} · [GitHub]({REPOSITORY_URL})"
    )


def footer() -> None:
    """The usage-log sentence (M6 section 7), the license, and where the docs live."""
    st.divider()
    st.caption(usage_log.FOOTER)
    st.caption(f"Apache-2.0 · [Docs]({REPOSITORY_URL}#readme) · [Source]({REPOSITORY_URL})")


def stats_caption(workload: Workload, stats: WorkloadStats) -> None:
    """The traffic chip: format, size, peak, token means (and the source's notes)."""
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


def answer_card(run: state.PlanRun, *, stale: bool, share: tuple[str | None, bool]) -> None:
    """The answer first: cost per day against the baseline, the fleet in one sentence, the
    confidence badge and banner (M8), the first replica's vLLM command, the share link and
    the plan JSON; a chip when the inputs changed since this plan."""
    result, confidence = run.result, run.result.perf_confidence
    saving, base = result.baseline_saving_pct, result.baseline
    with st.container(border=True):
        if stale:
            st.badge("Inputs changed since this plan: plan again", icon=":material/refresh:")
        st.metric(
            "Cost per day",
            wording.usd(result.cost_usd_per_day),
            delta=None
            if saving is None or base is None
            else f"{baseline_saving(saving)} against the best single-row fleet "
            f"({wording.usd(base.cost_usd_per_day)})",
            delta_color="normal" if (saving or 0.0) >= 0 else "inverse",
            delta_arrow="off",
        )
        fleet = wording.fleet_sentence(result, run.request.gpus)
        st.markdown(f"**{escape(fleet)}**", help=REPLICA_HELP)
        st.badge(
            f"Confidence: {confidence}",
            color=BADGE_COLORS.get(confidence, "orange"),  # type: ignore[arg-type]  # colour name
            help=f"{LEGEND} [How to calibrate](#{calibrate.ANCHOR})",
        )
        sentence = confidence_sentence(confidence, result.perf_sources)
        alert = st.warning if confidence in ("roofline", "mixed") else st.info
        alert(escape(f"Performance model: {sentence}"))
        st.caption("vLLM command of the first replica type (every type is in the Fleet tab):")
        st.code(serve_command(run.request.model, result.replicas[0].candidate.config), "bash")
        url, shareable = share
        st.caption(
            ("Share link: opens this page with these inputs and plans them." if url else "")
            + ("" if url else "The inputs are too long for a share link.")
            + ("" if shareable else " Uploads are not shareable: it opens with the default preset.")
        )
        if url:
            st.code(url, language=None)
        st.download_button(
            "Download plan JSON",
            data=render.get("json").plan(run.request, result, run.comparison),
            file_name="llmplan-plan.json",
            mime="application/json",
            on_click="ignore",
        )


def _table(rows: Sequence[Mapping[str, object]], wide: bool) -> None:
    frame = pd.DataFrame(rows)
    st.dataframe(frame if wide else frame.iloc[:, :KEY_COLUMNS], hide_index=True)


def fleet_tab(run: state.PlanRun, wide: bool) -> None:
    """The fleet table, then one card per replica type with its `vllm serve` line."""
    fleet = [
        {
            "provider": item.price_row.provider,
            "instance": item.price_row.instance,
            "instances": item.instances,
            "GPUs each": f"{item.price_row.gpu_count} x {item.price_row.gpu_id}",
            "$/day": round(item.usd_per_day, 2),
            "$/hour each": item.price_row.price_usd_per_hour,
            "price as of": item.price_row.as_of.isoformat(),
        }
        for item in run.result.fleet
    ]
    _table(fleet, wide)
    for replica in run.result.replicas:
        row, config = replica.candidate.price_row, replica.candidate.config
        perf = replica.candidate.perf
        with st.container(border=True):
            st.markdown(
                f"**{replica.count} replica(s)** "
                + escape(f"on {replica.instances} x {row.provider} {row.instance}"),
                help=REPLICA_HELP,
            )
            details = (
                f"tensor parallel {config.tensor_parallel} · {config.dtype} · max_num_seqs "
                f"{config.max_num_seqs} · max_model_len {config.max_model_len:,}"
            )
            if perf is not None:
                details += (
                    f" · TTFT p95 {wording.duration(perf.ttft_ms_p95)} · TPOT p95 "
                    f"{wording.duration(perf.tpot_ms_p95)} ({perf.backend}, {perf.confidence})"
                )
            st.caption(escape(details))
            st.code(serve_command(run.request.model, config), language="bash")


def routing_tab(run: state.PlanRun, wide: bool) -> None:
    """Request-size classes (M7): the comparison with sizing for the mean request (M8), the
    class table with each class's replayed TTFT, and (two or more classes) the weights."""
    result, comparison = run.result, run.comparison
    if comparison is None:
        st.markdown(
            "One request-size class: every replica serves the whole traffic. Choose 2x2 or "
            "3x3 classes under Advanced to size short and long requests separately."
        )
        return
    st.subheader("Request-size routing")
    st.markdown(f"**{escape(class_comparison_sentence(comparison))}**")
    st.caption("Each class is sized and checked against the latency target at its own shape.")
    replayed = {c.class_index: c for c in run.timeline.classes}
    classes = [
        {
            "class": c.index,
            "input tokens": f"{c.input_lo:,}..{c.input_hi:,}",
            "output tokens": f"{c.output_lo:,}..{c.output_hi:,}",
            "share": f"{c.share * 100:.1f}%",
            "TTFT violations": f"{replayed[c.index].ttft_violation_pct:.1f}%",
            "replayed TTFT p95 ms": replayed[c.index].ttft_ms_p95,
            "peak req/s": round(c.peak_rps, 3),
            "binding": binding,
        }
        for c, binding in zip(result.classes, result.class_binding, strict=True)
    ]
    _table(classes, wide)
    if len(result.classes) > 1:
        st.altair_chart(charts.routing_chart(result), width="stretch")


def timeline_tab(run: state.PlanRun, rewindow: Callable[[float], Timeline]) -> None:
    """Interactive charts of the replay with a window selector (another window replays the
    same plan again), and the PNG (M5 renderer) and JSON downloads."""
    auto = run.timeline.options.window_s
    choices = state.window_choices(auto)
    window = st.radio(
        "Timeline window",
        choices,
        index=choices.index(auto),
        horizontal=True,
        format_func=lambda w: f"{w:g} s" + (" (auto)" if w == auto else ""),
    )
    timeline = run.timeline if window == auto else rewindow(window)
    st.altair_chart(charts.timeline_chart(timeline), width="stretch")
    summary = timeline.summary
    st.caption(
        f"Replay of {summary.n_requests:,} requests in {len(timeline.windows)} windows of "
        f"{window:g} s: TTFT p95 {wording.duration(summary.ttft_ms_p95)} (queueing included), "
        f"{summary.ttft_violation_pct:.1f}% over the TTFT target, mean utilization "
        f"{summary.mean_utilization * 100:.0f}%. Drag to zoom the time axis; hover for values."
    )
    left, right = st.columns(2)
    left.download_button(
        "Download PNG",
        data=lambda: render_png(timeline),
        file_name="llmplan-timeline.png",
        mime="image/png",
        on_click="ignore",
    )
    right.download_button(
        "Download timeline JSON",
        data=timeline_json(timeline),
        file_name="llmplan-timeline.json",
        mime="application/json",
        on_click="ignore",
    )


def _candidate_row(c: CandidateEval) -> dict[str, object]:
    return {
        "status": c.status,
        "on": f"{c.price_row.provider} {c.price_row.instance}",
        "tp": c.config.tensor_parallel,
        "$/hour per req/s": c.usd_per_hour_per_rps,
        "reason": c.reason,
        "GPU": c.price_row.gpu_id,
        "dtype": c.config.dtype,
        "max_num_seqs": c.config.max_num_seqs,
    }


def candidates_tab(result: PlanResult, wide: bool) -> None:
    """The 15 cheapest eligible candidates by $/hour per req/s; rejected ones on request."""
    eligible = [c for c in result.candidates if c.status == "eligible"]
    rejected = [c for c in result.candidates if c.status != "eligible"]
    show = st.toggle(f"Show the {len(rejected)} rejected candidates", key="show_rejected")
    st.caption(
        f"The {min(len(eligible), TOP_CANDIDATES)} cheapest of {len(eligible)} eligible "
        f"candidates by $/hour per req/s{', then the rejected ones' if show else ''}. Click a "
        "column header to sort; hover a cell for its full text."
    )
    shown = eligible[:TOP_CANDIDATES] + (rejected if show else [])
    _table([_candidate_row(c) for c in shown], wide)


def assumptions_tab(run: state.PlanRun) -> None:
    """Every assumption of the plan, the perf estimates and the replay, verbatim and grouped;
    then the roofline constants (M8) and how to replace them with measurements."""
    for title, notes in state.assumptions(run).items():
        st.markdown(f"**{escape(title)}**")
        st.markdown("\n".join(f"- {escape(note)}" for note in notes) or "- none")
    st.markdown("**Roofline constants** (assumptions, not measurements)")
    st.markdown("\n".join(f"- `{n}` = {v:g}: {escape(text)}" for n, v, text in ROOFLINE_CONSTANTS))
    st.markdown(
        f"[How to calibrate](#{calibrate.ANCHOR}): upload your own `vllm bench serve` results "
        "(or an llmplan CSV) under Advanced; matching rows replace the model for this session "
        "and the confidence then says *measured*."
    )


def results(
    run: state.PlanRun,
    rewindow: Callable[[float], Timeline],
    *,
    stale: bool,
    share: tuple[str | None, bool],
) -> None:
    """The answer card, then Fleet | Routing | Timeline | Candidates | Assumptions."""
    answer_card(run, stale=stale, share=share)
    wide = st.toggle("Show all columns", key="all_columns", help="Tables show five key columns.")
    fleet, routing, timeline, candidates, assumptions = st.tabs(
        ["Fleet", "Routing", "Timeline", "Candidates", "Assumptions"]
    )
    with fleet:
        fleet_tab(run, wide)
    with routing:
        routing_tab(run, wide)
    with timeline:
        timeline_tab(run, rewindow)
    with candidates:
        candidates_tab(run.result, wide)
    with assumptions:
        assumptions_tab(run)
