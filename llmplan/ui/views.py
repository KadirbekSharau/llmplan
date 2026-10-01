"""What the web UI shows (M6; M9: header, empty state, answer card, five tabs), rendered from
library results and renderers; nothing is computed here. Tables show five columns unless
"Show all columns" is on."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from llmplan import __version__, render
from llmplan.perf.confidence import ROOFLINE_CONSTANTS, confidence_sentence
from llmplan.perf.contribute import REPOSITORY_URL as REPO
from llmplan.planner.result import PlanResult
from llmplan.render import charts
from llmplan.render.plan_text import baseline_saving, class_comparison_sentence, fleet_sentence
from llmplan.render.plots import render_png
from llmplan.render.timeline_json import timeline_json
from llmplan.render.vllm_cmd import serve_command
from llmplan.simulate import Timeline
from llmplan.ui import calibrate, state, usage_log
from llmplan.workload import Workload, WorkloadStats

PAGE_TITLE = "llmplan — GPU fleet planner for LLM inference"
FAVICON = (  # inline SVG: three bars of a fleet, in the theme's primary colour
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" '
    'rx="3" fill="#1F6FEB"/><path d="M4 13V7M8 13V3M12 13V9" stroke="#fff" stroke-width="2"/></svg>'
)
LEGEND = (
    "Confidence: **measured** (a published or uploaded benchmark row), **interpolated** "
    "(between measured rows), **roofline** (first-principles model, about ±30%)."
)
REPLICA = "A replica is one vLLM server with one copy of the model (several GPUs with tensor "
REPLICA += "parallelism)."
TOP_CANDIDATES = 15
Share = tuple[str | None, bool]  # the share link (None: too long) and whether it is complete
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!$|<>~])")


def escape(text: str) -> str:
    """Backslash-escape Markdown (and `$` math) so library text renders verbatim."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def header() -> None:
    """Name, promise, version, GitHub link; the title's help icon is the confidence legend."""
    st.title("llmplan", help=LEGEND)
    st.caption(
        "The cheapest GPU fleet and vLLM settings for your latency target, proven by replaying "
        f"your traffic. CPU-only planning. v{__version__} · [GitHub]({REPO})"
    )


def footer() -> None:
    """The usage-log sentence (M6 section 7), the license, and where the docs live."""
    st.divider()
    st.caption(usage_log.FOOTER)
    st.caption(f"Apache-2.0 · [Docs]({REPO}#readme) · [Source]({REPO})")


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


def _download(label: str, data: Any, name: str, where: Any = st) -> None:
    mime = "image/png" if name.endswith(".png") else "application/json"
    where.download_button(label, data, f"llmplan-{name}", mime, on_click="ignore")


def _table(columns: str, rows: Sequence[Sequence[Any]], wide: bool) -> None:
    frame = pd.DataFrame(rows, columns=columns.split("|"))
    st.dataframe(frame if wide else frame.iloc[:, :5], hide_index=True)


def _answer(run: state.PlanRun, stale: bool, share: Share) -> None:
    result, confidence, saving = (
        run.result,
        run.result.perf_confidence,
        run.result.baseline_saving_pct,
    )
    base = "" if result.baseline is None else f" vs {state.usd(result.baseline.cost_usd_per_day)}"
    url, shareable = share
    with st.container(border=True):
        if stale:
            st.badge("Inputs changed since this plan: plan again", icon=":material/refresh:")
        delta = None if saving is None else f"{baseline_saving(saving)}{base} on one GPU type"
        color: Any = "normal" if (saving or 0) >= 0 else "inverse"
        st.metric(
            "Cost per day", state.usd(result.cost_usd_per_day), delta, color, delta_arrow="off"
        )
        st.markdown(f"**{escape(fleet_sentence(result, run.request.gpus))}**", help=REPLICA)
        badge: Any = {"measured": "green", "interpolated": "blue"}.get(confidence, "orange")
        st.badge(f"Confidence: {confidence}", color=badge, help=f"{LEGEND} [Calibrate](#calibrate)")
        sentence = confidence_sentence(confidence, result.perf_sources)
        (st.warning if confidence in ("roofline", "mixed") else st.info)(
            escape(f"Performance model: {sentence}")
        )
        st.caption("vLLM command of the first replica type (every type is in the Fleet tab):")
        st.code(serve_command(run.request.model, result.replicas[0].candidate.config), "bash")
        note = "" if shareable else " Uploads are not shareable: it opens with the default preset."
        st.caption(
            ("Share link: opens this page with these inputs." if url else "Too long to share.")
            + note
        )
        if url:
            st.code(url, language=None)
        plan_json = render.get("json").plan(run.request, result, run.comparison)
        _download("Download plan JSON", plan_json, "plan.json")


def _fleet(run: state.PlanRun, wide: bool) -> None:
    rows = [
        (
            r.provider,
            r.instance,
            i.instances,
            f"{r.gpu_count} x {r.gpu_id}",
            round(i.usd_per_day, 2),
            r.price_usd_per_hour,
            r.as_of.isoformat(),
        )
        for i in run.result.fleet
        for r in [i.price_row]
    ]
    _table("provider|instance|instances|GPUs each|$/day|$/hour each|as of", rows, wide)
    for replica in run.result.replicas:
        row, config, perf = (
            replica.candidate.price_row,
            replica.candidate.config,
            replica.candidate.perf,
        )
        with st.container(border=True):
            where = escape(f"on {replica.instances} x {row.provider} {row.instance}")
            st.markdown(f"**{replica.count} replica(s)** {where}", help=REPLICA)
            details = f"tensor parallel {config.tensor_parallel} · {config.dtype} · max_num_seqs "
            details += f"{config.max_num_seqs} · max_model_len {config.max_model_len:,}"
            if perf is not None:
                details += f" · TTFT p95 {state.duration(perf.ttft_ms_p95)} · TPOT p95 "
                details += f"{state.duration(perf.tpot_ms_p95)} ({perf.backend}, {perf.confidence})"
            st.caption(escape(details))
            st.code(serve_command(run.request.model, config), language="bash")


def _routing(run: state.PlanRun, wide: bool) -> None:
    result, comparison = run.result, run.comparison
    if comparison is None:
        st.markdown("One request-size class. Choose 2x2 or 3x3 under Advanced to split them.")
        return
    st.subheader("Request-size routing")
    st.markdown(f"**{escape(class_comparison_sentence(comparison))}**")
    st.caption("Each class is sized and checked against the latency target at its own shape.")
    replay = {c.class_index: c for c in run.timeline.classes}
    rows = [
        (
            c.index,
            f"{c.input_lo:,}..{c.input_hi:,}",
            f"{c.output_lo:,}..{c.output_hi:,}",
            f"{c.share:.1%}",
            f"{replay[c.index].ttft_violation_pct:.1f}%",
            replay[c.index].ttft_ms_p95,
            round(c.peak_rps, 3),
            binding,
        )
        for c, binding in zip(result.classes, result.class_binding, strict=True)
    ]
    columns = "class|input tokens|output tokens|share|TTFT violations|replayed TTFT p95 ms"
    _table(f"{columns}|peak req/s|binding", rows, wide)
    if len(result.classes) > 1:
        st.altair_chart(charts.routing_chart(result), width="stretch")


def _timeline(run: state.PlanRun, rewindow: Callable[[float], Timeline]) -> None:
    auto = run.timeline.options.window_s
    choices = state.window_choices(auto)
    label = {w: f"{w:g} s" + (" (auto)" if w == auto else "") for w in choices}
    window = st.radio("Timeline window", choices, choices.index(auto), label.get, horizontal=True)
    timeline = run.timeline if window == auto else rewindow(window)
    st.altair_chart(charts.timeline_chart(timeline), width="stretch")
    summary = timeline.summary
    st.caption(
        f"{summary.n_requests:,} requests in {len(timeline.windows)} windows of {window:g} s: "
        f"TTFT p95 {state.duration(summary.ttft_ms_p95)} with queueing, "
        f"{summary.ttft_violation_pct:.1f}% over target. Drag to zoom; hover for values."
    )
    left, right = st.columns(2)
    _download("Download PNG", lambda: render_png(timeline), "timeline.png", left)
    _download("Download timeline JSON", timeline_json(timeline), "timeline.json", right)


def _candidates(result: PlanResult, wide: bool) -> None:
    eligible = [c for c in result.candidates if c.status == "eligible"]
    rejected = [c for c in result.candidates if c.status != "eligible"]
    show = st.toggle(f"Show the {len(rejected)} rejected candidates", key="show_rejected")
    shown = min(len(eligible), TOP_CANDIDATES)
    st.caption(
        f"The {shown} cheapest of {len(eligible)} eligible, by $/hour per req/s. Click to sort."
    )
    rows = [
        (
            c.status,
            f"{c.price_row.provider} {c.price_row.instance}",
            c.config.tensor_parallel,
            c.usd_per_hour_per_rps,
            c.reason,
            c.price_row.gpu_id,
            c.config.dtype,
            c.config.max_num_seqs,
        )
        for c in eligible[:TOP_CANDIDATES] + (rejected if show else [])
    ]
    _table("status|on|tp|$/hour per req/s|reason|GPU|dtype|max_num_seqs", rows, wide)


def _assumptions(run: state.PlanRun) -> None:
    for title, notes in state.assumptions(run).items():
        st.markdown(f"**{escape(title)}**")
        st.markdown("\n".join(f"- {escape(note)}" for note in notes) or "- none")
    st.markdown("**Roofline constants** (assumptions, not measurements)")
    st.markdown("\n".join(f"- `{n}` = {v:g}: {escape(text)}" for n, v, text in ROOFLINE_CONSTANTS))
    st.markdown(
        f"[Calibrate](#{calibrate.ANCHOR}) under Advanced with your own `vllm bench serve` "
        "results: matching rows replace the model for this session (confidence *measured*)."
    )


def results(
    run: state.PlanRun, rewindow: Callable[[float], Timeline], stale: bool, share: Share
) -> None:
    """The answer card (cost, fleet, confidence, vLLM command, share link, JSON; a chip when
    the inputs changed), then Fleet, Routing, Timeline (`rewindow` replays the plan at another
    window), Candidates (15 cheapest eligible; rejected on request) and Assumptions tabs."""
    _answer(run, stale, share)
    wide = st.toggle("Show all columns", key="all_columns", help="Tables show five key columns.")
    tabs = st.tabs(["Fleet", "Routing", "Timeline", "Candidates", "Assumptions"])
    with tabs[0]:
        _fleet(run, wide)
    with tabs[1]:
        _routing(run, wide)
    with tabs[2]:
        _timeline(run, rewindow)
    with tabs[3]:
        _candidates(run.result, wide)
    with tabs[4]:
        _assumptions(run)
