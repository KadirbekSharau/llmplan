"""llmplan web UI (`llmplan ui`; M6, M9): four input steps, the answer first, five tabs.

No planning logic: `state` calls the library, only on Plan, an example or a share link.
Inputs live in session state under their widget keys (`presets.INPUTS`). Limits (M6
section 6): 50 MB uploads checked before parsing, 30 s solver limit, 30 plans per hour.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, get_args

import pandas as pd
import streamlit as st

from llmplan import render
from llmplan.catalog.hardware import GPUSpec, PriceRow, load_gpus, load_prices
from llmplan.catalog.models import FIXTURE_PREFIX, ModelSpec, load_model
from llmplan.errors import InfeasiblePlan, LLMPlanError, ValidationError
from llmplan.perf.uploads import upload_backends
from llmplan.planner import SLO, PlanRequest, PlanResult
from llmplan.render.text import model_summary
from llmplan.simulate import SimOptions, Timeline
from llmplan.types import Commitment
from llmplan.ui import calibrate, presets, share, state, usage_log, views
from llmplan.workload import (
    InMemoryTrace,
    Workload,
    WorkloadStats,
    compute_stats,
    generate,
    load_workload,
)
from llmplan.workload.classes import classify_spec
from llmplan.workload.synth import parse_distribution

log = logging.getLogger("llmplan.ui")
ss = st.session_state
Traffic = Callable[[], Workload]
Job = tuple[PlanRequest, Workload, SimOptions, calibrate.Rows, Callable[[str, float], None]]
STEPS = ("Resolving model", "Loading traffic", "Estimating performance", "Solving", "Replaying")
HELP = {
    "ttft": "Time to first token (p95): until the first word appears. Empty: no target.",
    "tpot": "Time per output token (p95): between words as the answer streams. Empty: no target.",
    "utilization": "Capacity is derated to this share of each replica, keeping headroom.",
    "tokens": "fixed:N, lognormal:MEAN:SIGMA[:LO:HI] (underlying normal), or uniform:LO:HI",
    "classes": "Input x output token bins (median splits for 2x2); 1 sizes for the mean request",
    "compact": "Inputs above the results, for phones (Streamlit cannot see the screen width).",
}


@st.cache_data(ttl=86_400, show_spinner=False)
def _catalogs() -> tuple[dict[str, GPUSpec], tuple[PriceRow, ...]]:
    gpus = dict(load_gpus())  # a plain dict: cached values are pickled
    return gpus, load_prices(gpus=gpus)


@st.cache_data(ttl=86_400, show_spinner=False)
def _model(model_id: str) -> ModelSpec:
    return load_model(model_id)  # HF ids are validated by the M1 fetcher before any request


@st.cache_data(ttl=86_400, show_spinner=False)
def _preset(key: str) -> Workload:
    return presets.load_preset(presets.preset(key))


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _run(key: str, _job: Job) -> state.PlanRun:  # `key` (state.cache_key) identifies `_job`
    request, trace, options, rows, progress = _job
    return state.run_plan(request, trace, options, request.gpus, upload_backends(rows), progress)


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _rewindow(key: str, window_s: float, _run: state.PlanRun, _trace: Workload) -> Timeline:
    return state.replay_window(_run, _trace, window_s)


def _defaults(shipped: tuple[PriceRow, ...]) -> dict[str, Any]:
    catalog = {"gpu_ids": {r.gpu_id for r in shipped}, "providers": {r.provider for r in shipped}}
    return presets.DEFAULTS | {key: sorted(values) for key, values in catalog.items()}


def _seed(gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]) -> list[str]:
    """Defaults for inputs missing from session state (every run: Streamlit drops undrawn
    widgets' state); a session's first run applies a share link. Returns ignored params."""
    defaults, ignored = _defaults(shipped), list[str]()
    if "seeded" not in ss:
        ss["seeded"], ss["compact"] = True, st.query_params.get("layout") == "compact"
        choices = presets.CHOICES | {"gpu_ids": list(gpus), "providers": defaults["providers"]}
        params = {k: v for k, v in st.query_params.to_dict().items() if k != "layout"}
        values, ignored = share.decode(params, choices)
        _set(values, run=bool(values))
    for key, value in defaults.items():
        ss.setdefault(key, list(value) if isinstance(value, list) else value)
    return ignored


def _set(values: Mapping[str, Any], *, run: bool = False) -> None:
    ss.update(values, run_now=run)


def _number(label: str, key: str, **kwargs: Any) -> None:
    st.number_input(label, *presets.BOUNDS[key], key=key, help=HELP.get(key), **kwargs)


def _fail(error: LLMPlanError) -> Traffic:
    def raise_it() -> Workload:
        raise error

    return raise_it


def _model_id() -> str:
    custom = ss["model_choice"] == presets.CUSTOM_MODEL
    return str(ss["model_custom"]).strip() if custom else str(ss["model_choice"])


def _model_step() -> None:
    label = {m: f"{presets.model_group(m)} · {m}" for m in presets.MODEL_CHOICES}
    choices = [*presets.MODEL_CHOICES, presets.CUSTOM_MODEL]
    st.selectbox("Model", choices, key="model_choice", format_func=lambda m: label.get(m, m))
    if ss["model_choice"] == presets.CUSTOM_MODEL:
        st.text_input("Hugging Face id (org/name)", key="model_custom")
    st.caption("Fetched from Hugging Face on Plan; gated models need HF_TOKEN on the server.")
    model_id = _model_id()  # fixtures resolve offline; Hugging Face ids once a plan fetched them
    spec = _model(model_id) if model_id.startswith(FIXTURE_PREFIX) else None
    if (spec := spec or ss.get("resolved_models", {}).get(model_id)) is not None:
        st.caption(views.escape(model_summary(spec)))
        with st.popover("Model info"):
            st.code(render.get("text").model_info(spec), language=None)


def _upload_source() -> Traffic:
    uploaded = st.file_uploader("Trace CSV (up to 50 MB)", type=["csv"], key="upload")
    st.caption("Formats: generic csv, Azure 2023/2024, BurstGPT. Parsed in memory, never stored.")
    if uploaded is None:
        ss.pop("upload_parsed", None)
        return _fail(ValidationError("upload a CSV trace, or choose a preset or synthetic traffic"))
    parsed: Workload | LLMPlanError = ValidationError(
        f"Upload refused: {uploaded.size:,} bytes is over the 50 MB "
        f"({presets.MAX_UPLOAD_BYTES:,} bytes) limit; nothing was parsed."
    )
    if uploaded.size <= presets.MAX_UPLOAD_BYTES:
        if ss.get("upload_parsed", ("",))[0] != uploaded.file_id:  # parse once per file
            try:
                data = InMemoryTrace("uploaded.csv", uploaded.getvalue())
                parsed = load_workload(data, max_bytes=presets.MAX_UPLOAD_BYTES)
            except LLMPlanError as exc:
                parsed = exc
            ss["upload_parsed"] = (uploaded.file_id, parsed)
        parsed = ss["upload_parsed"][1]
    if isinstance(parsed, LLMPlanError):
        st.error(str(parsed))
        return _fail(parsed)
    views.stats_caption(parsed, compute_stats(workload := parsed))
    return lambda: workload


def _synthetic_source() -> Traffic:
    _number("Rate (req/s)", "syn_rate")
    _number("Duration (s)", "syn_duration", step=60.0)
    st.text_input("Input tokens", key="syn_in", help=HELP["tokens"])
    st.text_input("Output tokens", key="syn_out", help=HELP["tokens"])
    _number("Seed", "syn_seed")
    rate, duration, seed = ss["syn_rate"], ss["syn_duration"], int(ss["syn_seed"])
    lengths = ss["syn_in"], ss["syn_out"]

    def build() -> Workload:
        if (expected := rate * duration) > presets.MAX_SIM_REQUESTS:
            raise ValidationError(
                f"rate x duration = {expected:,.0f} expected requests; the web UI allows up "
                f"to {presets.MAX_SIM_REQUESTS:,}"
            )
        inputs, outputs = map(parse_distribution, lengths)
        return generate(
            rate_rps=rate,
            duration_s=duration,
            input_tokens=inputs,
            output_tokens=outputs,
            seed=seed,
        )

    return build


def _traffic_step() -> Traffic:
    modes = presets.TRAFFIC_MODES
    mode = st.radio("Source", modes, key="traffic_mode", captions=presets.TRAFFIC_HELP)
    if mode != modes[0]:
        return _upload_source() if mode == modes[1] else _synthetic_source()
    labels = {p.key: p.label for p in presets.PRESETS}
    key = st.selectbox("Preset", list(labels), format_func=labels.__getitem__, key="preset")
    workload = _preset(key)
    views.stats_caption(workload, compute_stats(workload))
    return lambda: workload


def _target_step() -> None:
    for column, (name, values) in zip(st.columns(3), presets.TARGET_PRESETS.items(), strict=True):
        hint = "TTFT {:g} ms, TPOT {:g} ms".format(*values) if values[0] else "no latency target"
        targets = dict(zip(("ttft", "tpot"), values, strict=True))
        column.button(name, key=f"target_{name}", help=hint, on_click=_set, args=(targets,))
    _number("TTFT p95 (ms)", "ttft", value=None)
    _number("TPOT p95 (ms)", "tpot", value=None)
    _number("Utilization target", "utilization", step=0.05)


def _reset_prices() -> None:
    ss["price_table"] = state.prices_frame(_catalogs()[1])
    ss["price_version"] = ss.get("price_version", 0) + 1


def _hardware_step(gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]) -> pd.DataFrame:
    if "price_table" not in ss:
        _reset_prices()
    filters, column = st.container(), st.column_config
    with st.popover("Edit prices", width="stretch"):
        table: pd.DataFrame = st.data_editor(
            ss["price_table"],
            key=f"price_editor_{ss['price_version']}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "gpu_id": column.SelectboxColumn(options=list(gpus)),
                "gpu_count": column.NumberColumn(min_value=1, step=1),
                "price_usd_per_hour": column.NumberColumn(format="%.4f", min_value=0.0),
                "commitment": column.SelectboxColumn(options=get_args(Commitment)),
                "as_of": column.DateColumn(),
            },
        )
        dates = "{} to {}".format(*state.as_of_range(shipped))
        st.caption(f"USD per instance-hour as of {dates}; edits are checked on Plan.")
        st.button("Reset prices", on_click=_reset_prices)
    providers = sorted({str(p) for p in table["provider"].dropna()} | {r.provider for r in shipped})
    filters.multiselect("Providers", providers, key="providers")
    filters.multiselect("GPUs", list(gpus), key="gpu_ids")
    return table


def _advanced(gpus: Mapping[str, GPUSpec]) -> calibrate.BenchmarkFile | None:
    st.multiselect("Tensor parallel", presets.TP_CHOICES, key="tensor_parallel")
    st.multiselect("dtype", presets.DTYPE_CHOICES, key="dtypes")
    st.multiselect("max_num_seqs", presets.MAX_NUM_SEQS_CHOICES, key="max_num_seqs")
    _number("max_model_len", "max_model_len", step=256)
    st.selectbox("Request-size classes", presets.CLASS_CHOICES, key="classes", help=HELP["classes"])
    st.selectbox("Perf backend", presets.PERF_BACKENDS, key="perf_backend")
    st.selectbox("Solver", presets.SOLVERS, key="solver")
    _number("Solver time limit (s)", "time_limit_s")
    return calibrate.sidebar(gpus, presets.DTYPE_CHOICES)


def _inputs(
    gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]
) -> tuple[Traffic, pd.DataFrame, calibrate.BenchmarkFile | None, bool]:
    """The four steps, Advanced and Plan: in the sidebar, or (compact) an accordion on top."""
    compact = bool(ss["compact"])
    with st.container() if compact else st.sidebar:
        with st.expander("1. Model", expanded=True):
            _model_step()
        with st.expander("2. Traffic", expanded=not compact):
            traffic = _traffic_step()
        with st.expander("3. Latency target", expanded=not compact):
            _target_step()
        with st.expander("4. Hardware and prices", expanded=not compact):
            table = _hardware_step(gpus, shipped)
        with st.expander("Advanced"):
            benchmarks = _advanced(gpus)
        clicked = st.button("Plan", type="primary", key="plan", width="stretch")
    return traffic, table, benchmarks, clicked


def _slo() -> SLO:
    return SLO(ttft_ms_p95=ss["ttft"], tpot_ms_p95=ss["tpot"], utilization_target=ss["utilization"])


def _done(step: str, seconds: float | None) -> None:
    st.write(f"{step}: {'cached result' if seconds is None else state.duration(seconds * 1000)}")


def _on_plan(
    traffic: Traffic, table: pd.DataFrame, benchmarks: calibrate.BenchmarkFile | None
) -> tuple[str, object]:
    """One plan inside the five progress steps, logged once it reached the planner; returns
    ("run", PlanRun), or ("error" or "warning", the text to show)."""
    admitted, ss["plan_times"] = state.admit_plan(ss.get("plan_times", ()), time.time())
    if not admitted:
        return "warning", (
            f"You have run {presets.MAX_PLANS_PER_HOUR} plans in the last hour. Please wait a "
            "few minutes before planning again."
        )
    request_id, started, gpus = uuid.uuid4().hex[:12], time.perf_counter(), _catalogs()[0]
    stats: WorkloadStats | None = None
    result: PlanResult | None = None
    outcome: tuple[str, object]
    status: str
    try:
        with st.status("Planning...", expanded=False) as progress:
            if not (model_id := _model_id()):
                raise ValidationError("enter a Hugging Face model id (org/name)")
            model = ss.setdefault("resolved_models", {})[model_id] = _model(model_id)
            rows = calibrate.rows_for_plan(benchmarks, model, gpus)
            _done(STEPS[0], (resolved := time.perf_counter()) - started)
            stats = compute_stats(workload := traffic())
            _done(STEPS[1], time.perf_counter() - resolved)
            request = state.build_request(
                model=model,
                stats=stats,
                slo=_slo(),
                gpus=gpus,
                prices=state.price_rows(table, gpus),
                classes=classify_spec(workload, ss["classes"]),
                **{key: ss[key] for key in presets.OPTION_KEYS},
            )
            options = state.sim_options(stats)
            key = state.cache_key(request, options, workload, rows)
            limit = f"time limit {request.options.time_limit_s:g} s"
            progress.update(label=f"Estimating performance, solving ({limit}) and replaying...")
            timings: dict[str, float] = {}  # run_plan's stage seconds; empty for a cached run
            run = _run(key, (request, workload, options, rows, timings.__setitem__))
            for stage, step in zip(("estimate", "solve", "replay"), STEPS[2:], strict=True):
                _done(f"{step} ({limit})" if stage == "solve" else step, timings.get(stage))
            took = state.duration((time.perf_counter() - started) * 1000)
            progress.update(label=f"Planned in {took}", state="complete")
        ss["plan_key"], ss["plan_workload"] = key, workload  # for the Timeline window selector
        result, status, outcome = run.result, run.result.solver.status, ("run", run)
    except LLMPlanError as exc:
        status = "infeasible" if isinstance(exc, InfeasiblePlan) else "error"
        outcome = ("error", state.error_text(exc))
    except Exception:
        log.exception("plan failed request_id=%s", request_id)
        status, outcome = "error", ("error", f"Something went wrong (request id {request_id}).")
    if stats is not None:  # the planner was reached: log the input shapes (M6 section 7)
        usage_log.append(
            usage_log.usage_record(
                request_id=request_id,
                model_id=_model_id(),
                gpu_ids=ss["gpu_ids"],
                stats=stats,
                slo=_slo(),
                result=result,
                solver_status=status,
                duration_s=time.perf_counter() - started,
            )
        )
    return outcome


def main() -> None:
    """The page: header, inputs, Plan, then the answer and its tabs (or the empty state)."""
    st.set_page_config(views.PAGE_TITLE, views.FAVICON, "wide", "auto")
    gpus, shipped = _catalogs()
    ignored = _seed(gpus, shipped)
    views.header()
    if st.toggle("Compact layout", key="compact", help=HELP["compact"]):  # kept in the URL
        st.query_params["layout"] = "compact"
    else:
        st.query_params.pop("layout", None)
    if ignored:
        st.warning(f"Ignored link parameters (unknown or invalid): {', '.join(ignored)}.")
    traffic, table, benchmarks, clicked = _inputs(gpus, shipped)
    keys = (*presets.DEFAULTS, "upload", "bench_upload", *calibrate.RUN_KEYS)
    inputs = repr([getattr(ss.get(k), "file_id", ss.get(k)) for k in keys]) + table.to_csv()
    stale = ss.get("outcome", ("",))[0] == "run" and ss.get("planned_inputs") != inputs
    again = st.empty()  # the main area's copy of Plan while the plan shown is stale
    clicked |= stale and again.button("Plan again", key="plan_again", type="primary")
    if clicked or ss.pop("run_now", False):
        ss["outcome"], ss["planned_inputs"] = _on_plan(traffic, table, benchmarks), inputs
        params, shareable = share.encode(ss.to_dict())  # the plan's inputs into the page URL
        st.query_params.from_dict(params | ({"layout": "compact"} if ss["compact"] else {}))
        ss["share"], stale = (share.link(st.context.url or "", params), shareable), False
        again.empty()
    kind, value = ss.get("outcome", ("none", None))
    if kind == "run" and isinstance(value, state.PlanRun):
        key, trace = ss["plan_key"], ss["plan_workload"]
        views.results(value, lambda w: _rewindow(key, w, value, trace), stale, ss["share"])
        calibrate.report()
    elif kind in ("error", "warning"):
        (st.error if kind == "error" else st.warning)(str(value))
    else:
        views.empty_state(
            bool(ss["compact"]), lambda example: _set(_defaults(shipped) | example, run=True)
        )
    views.footer()


main()
