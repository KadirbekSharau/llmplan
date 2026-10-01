"""llmplan web UI (M6; M9: four input steps, the answer first, five result tabs).

Run with `llmplan ui`. The page holds no planning logic: results come from `llmplan.*`
through `llmplan.ui.state` (M8: own benchmarks through `llmplan.ui.calibrate`), computed
only on Plan, and logged by `usage_log` when enabled. Every input lives in session state
under its widget key, seeded from `presets.DEFAULTS`. Limits (M6_DESIGN.md section 6): 50 MB
uploads checked before parsing, 30 s solver limit, 200,000 requests, 30 plans per hour.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any

import pandas as pd
import streamlit as st

from llmplan import render
from llmplan.catalog.hardware import GPUSpec, PriceRow, load_gpus, load_prices
from llmplan.catalog.models import FIXTURE_PREFIX, ModelSpec, load_model
from llmplan.errors import InfeasiblePlan, LLMPlanError, ValidationError
from llmplan.perf.uploads import upload_backends
from llmplan.planner import SLO, PlanRequest, PlanResult
from llmplan.simulate import SimOptions, Timeline
from llmplan.types import Commitment
from llmplan.ui import calibrate, presets, share, state, usage_log, views, wording
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
STEPS = ("Resolving model", "Loading traffic", "Estimating performance", "Solving", "Replaying")
ss = st.session_state
Traffic = Callable[[], Workload]
COMMITMENTS: tuple[Commitment, ...] = ("on_demand", "reserved_1y", "reserved_3y", "spot")
DISTRIBUTION_HELP = "fixed:N, lognormal:MEAN:SIGMA[:LO:HI] (underlying normal), or uniform:LO:HI"
CLASSES_HELP = "Input x output token bins (median splits for 2x2); 1 sizes for the mean request"
COMPACT_HELP = (
    "Inputs above the results instead of in the sidebar, for phones. Streamlit cannot detect "
    "the screen width, so this is a switch; the page URL remembers it."
)
TARGET_HELP = {
    "ttft": "Time to first token (95th percentile): how long until the first word appears.",
    "tpot": "Time per output token (95th percentile): the gap between words as the answer streams.",
    "utilization": "Capacity is derated to this share of each replica, keeping headroom.",
}


@st.cache_data(ttl=86_400, show_spinner=False)
def _catalogs() -> tuple[dict[str, GPUSpec], tuple[PriceRow, ...]]:
    gpus = dict(load_gpus())  # a plain dict: cached values are pickled
    return gpus, load_prices(gpus=gpus)


@st.cache_data(ttl=86_400, show_spinner=False)
def _model(model_id: str) -> ModelSpec:
    return load_model(model_id)  # HF ids are validated by the M1 fetcher before any request


@st.cache_data(ttl=86_400, show_spinner=False)
def _preset(key: str) -> tuple[Workload, WorkloadStats]:
    workload = presets.load_preset(presets.preset(key))
    return workload, compute_stats(workload)


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _run(
    key: str,
    _request: PlanRequest,
    _workload: Workload,
    _options: SimOptions,
    _gpus: Mapping[str, GPUSpec],
    _rows: calibrate.Rows,
    _progress: Callable[[str, float], None],
) -> state.PlanRun:
    return state.run_plan(_request, _workload, _options, _gpus, upload_backends(_rows), _progress)


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _rewindow(key: str, window_s: float, _run: state.PlanRun, _workload: Workload) -> Timeline:
    return state.replay_window(_run, _workload, window_s, _catalogs()[0])


def _seed(gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]) -> list[str]:
    """Default inputs, set before any widget is drawn; every run, because Streamlit drops the
    state of a widget that the previous run did not draw. On a session's first run a share
    link's valid parameters go first (and plan); returns the names of those ignored."""
    providers = sorted({row.provider for row in shipped})
    ignored: list[str] = []
    if "seeded" not in ss:
        ss["seeded"] = True
        choices = {
            **presets.CHOICES,
            **{"model_choice": presets.MODEL_CHOICES, "gpu_ids": list(gpus)},
            **{"preset": [p.key for p in presets.PRESETS], "providers": providers},
        }
        params = {k: v for k, v in st.query_params.to_dict().items() if k != "layout"}
        values, ignored = share.decode(params, choices)
        ss.update(values)
        ss["run_now"] = bool(values)
    ss.setdefault("compact", st.query_params.get("layout") == "compact")
    defaults = {**presets.DEFAULTS, "gpu_ids": sorted({r.gpu_id for r in shipped})}
    for key, value in {**defaults, "providers": providers}.items():
        ss.setdefault(key, list(value) if isinstance(value, list) else value)
    return ignored


def _fingerprint(table: pd.DataFrame) -> str:
    """Every input as text, to tell when the inputs changed after the last plan."""
    keys = (*presets.DEFAULTS, "upload", "bench_upload", *calibrate.RUN_KEYS)
    return repr([getattr(ss.get(k), "file_id", ss.get(k)) for k in keys]) + table.to_csv()


def _share_after_plan() -> tuple[str | None, bool]:
    """Put the plan's inputs in the page URL; returns the share link and whether the
    traffic was shareable (an upload is not)."""
    params, shareable = share.encode(ss.to_dict())
    st.query_params.from_dict({**params, **({"layout": "compact"} if ss["compact"] else {})})
    return share.link(st.context.url or "", params), shareable


def _apply(values: Mapping[str, Any]) -> None:
    ss.update(values)


def _remember_layout() -> None:
    if ss["compact"]:
        st.query_params["layout"] = "compact"
    else:
        st.query_params.pop("layout", None)


def _number(label: str, key: str, **kwargs: Any) -> None:
    low, high = presets.BOUNDS[key]
    st.number_input(label, min_value=low, max_value=high, key=key, **kwargs)


def _fail(error: LLMPlanError) -> Traffic:
    def raise_it() -> Workload:
        raise error

    return raise_it


def _model_id() -> str:
    custom = ss["model_choice"] == presets.CUSTOM_MODEL
    return str(ss["model_custom"]).strip() if custom else str(ss["model_choice"])


def _known_model(model_id: str) -> ModelSpec | None:
    """Fixtures resolve offline right away; Hugging Face ids once a plan has fetched them."""
    if model_id.startswith(FIXTURE_PREFIX):
        return _model(model_id)  # the picker offers shipped fixtures only
    resolved: dict[str, ModelSpec] = ss.get("resolved_models", {})
    return resolved.get(model_id)


def _model_step() -> None:
    choices = [*presets.MODEL_CHOICES, presets.CUSTOM_MODEL]
    label = {m: f"{presets.model_group(m)} · {m}" for m in presets.MODEL_CHOICES}
    st.selectbox("Model", choices, key="model_choice", format_func=lambda m: label.get(m, m))
    if ss["model_choice"] == presets.CUSTOM_MODEL:
        st.text_input("Hugging Face id (org/name)", key="model_custom")
    st.caption(
        "Hugging Face configs are fetched when you click Plan. Gated models need HF_TOKEN "
        "set on the server; it is never shown or logged."
    )
    spec = _known_model(_model_id())
    if spec is not None:
        st.caption(views.escape(wording.model_summary(spec)))
        with st.popover("Model info"):
            st.code(render.get("text").model_info(spec), language=None)


def _upload_source() -> Traffic:
    uploaded = st.file_uploader("Trace CSV (up to 50 MB)", type=["csv"], key="upload")
    st.caption("Formats: generic csv, Azure 2023/2024, BurstGPT. Parsed in memory, never stored.")
    if uploaded is None:
        ss.pop("upload_parsed", None)
        return _fail(ValidationError("upload a CSV trace, or choose a preset or synthetic traffic"))
    if uploaded.size > presets.MAX_UPLOAD_BYTES:
        refused = ValidationError(
            f"Upload refused: {uploaded.size:,} bytes is over the 50 MB "
            f"({presets.MAX_UPLOAD_BYTES:,} bytes) limit; nothing was parsed."
        )
        st.error(str(refused))
        return _fail(refused)
    cached = ss.get("upload_parsed")
    if cached is None or cached[0] != uploaded.file_id:
        try:
            parsed: Workload | LLMPlanError = load_workload(
                InMemoryTrace("uploaded.csv", uploaded.getvalue()),
                max_bytes=presets.MAX_UPLOAD_BYTES,
            )
        except LLMPlanError as exc:
            parsed = exc
        cached = ss["upload_parsed"] = (uploaded.file_id, parsed)
    parsed = cached[1]
    if isinstance(parsed, LLMPlanError):
        st.error(str(parsed))
        return _fail(parsed)
    views.stats_caption(parsed, compute_stats(parsed))
    workload = parsed
    return lambda: workload


def _synthetic_source() -> Traffic:
    _number("Rate (req/s)", "syn_rate")
    _number("Duration (s)", "syn_duration", step=60.0)
    st.text_input("Input tokens", key="syn_in", help=DISTRIBUTION_HELP)
    st.text_input("Output tokens", key="syn_out", help=DISTRIBUTION_HELP)
    _number("Seed", "syn_seed")
    rate, duration, in_spec, out_spec, seed = (
        ss[k] for k in ("syn_rate", "syn_duration", "syn_in", "syn_out", "syn_seed")
    )

    def build() -> Workload:
        expected = rate * duration
        if expected > presets.MAX_SIM_REQUESTS:
            raise ValidationError(
                f"rate x duration = {expected:,.0f} expected requests; the web UI allows up "
                f"to {presets.MAX_SIM_REQUESTS:,}"
            )
        return generate(
            rate_rps=rate,
            duration_s=duration,
            input_tokens=parse_distribution(in_spec),
            output_tokens=parse_distribution(out_spec),
            seed=int(seed),
        )

    return build


def _traffic_step() -> Traffic:
    modes = presets.TRAFFIC_MODES
    mode = st.radio("Source", modes, key="traffic_mode", captions=presets.TRAFFIC_HELP)
    if mode == modes[1]:
        return _upload_source()
    if mode == modes[2]:
        return _synthetic_source()
    labels = {p.key: p.label for p in presets.PRESETS}
    key = st.selectbox("Preset", list(labels), format_func=labels.__getitem__, key="preset")
    workload, stats = _preset(key)
    views.stats_caption(workload, stats)
    return lambda: workload


def _target_step() -> None:
    columns = st.columns(len(presets.TARGET_PRESETS))
    for column, (name, (ttft, tpot)) in zip(columns, presets.TARGET_PRESETS.items(), strict=True):
        hint = "no target" if ttft is None else f"TTFT {ttft:g} ms, TPOT {tpot:g} ms"
        values = {"ttft": ttft, "tpot": tpot}
        column.button(name, key=f"target_{name}", help=hint, on_click=_apply, args=(values,))
    _number("TTFT p95 (ms)", "ttft", value=None, help=f"{TARGET_HELP['ttft']} Empty: no target.")
    _number("TPOT p95 (ms)", "tpot", value=None, help=f"{TARGET_HELP['tpot']} Empty: no target.")
    _number("Utilization target", "utilization", step=0.05, help=TARGET_HELP["utilization"])


def _reset_prices() -> None:
    ss["price_table"] = state.prices_frame(_catalogs()[1])
    ss["price_version"] = ss.get("price_version", 0) + 1


def _hardware_step(gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]) -> pd.DataFrame:
    if "price_table" not in ss:
        _reset_prices()
    filters = st.container()
    with st.popover("Edit prices", width="stretch"):
        table: pd.DataFrame = st.data_editor(
            ss["price_table"],
            key=f"price_editor_{ss['price_version']}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "gpu_id": st.column_config.SelectboxColumn(options=list(gpus)),
                "gpu_count": st.column_config.NumberColumn(min_value=1, step=1),
                "price_usd_per_hour": st.column_config.NumberColumn(format="%.4f", min_value=0.0),
                "commitment": st.column_config.SelectboxColumn(options=list(COMMITMENTS)),
                "as_of": st.column_config.DateColumn(),
            },
        )
        oldest, newest = state.as_of_range(shipped)
        st.caption(
            f"Shipped prices as of {oldest} to {newest} (USD per instance-hour). Edits are "
            "validated row by row when you click Plan."
        )
        st.button("Reset prices", on_click=_reset_prices)
    providers = sorted({str(p) for p in table["provider"].dropna()} | {r.provider for r in shipped})
    filters.multiselect("Providers", providers, key="providers")
    filters.multiselect("GPUs", list(gpus), key="gpu_ids")
    return table


def _advanced(gpus: Mapping[str, GPUSpec]) -> calibrate.BenchmarkFile | None:
    st.multiselect("Tensor parallel", presets.TP_CHOICES, key="tp")
    st.multiselect("dtype", presets.DTYPE_CHOICES, key="dtypes")
    st.multiselect("max_num_seqs", presets.MAX_NUM_SEQS_CHOICES, key="seqs")
    _number("max_model_len", "max_model_len", step=256)
    st.selectbox("Request-size classes", presets.CLASS_CHOICES, key="classes", help=CLASSES_HELP)
    st.selectbox("Perf backend", presets.PERF_BACKENDS, key="perf_backend")
    st.selectbox("Solver", presets.SOLVERS, key="solver")
    _number("Solver time limit (s)", "time_limit")
    return calibrate.sidebar(gpus, presets.DTYPE_CHOICES)


def _step(title: str, compact: bool) -> AbstractContextManager[Any]:
    """A numbered input step: open in the sidebar; an accordion (step 1 open) when compact."""
    return st.expander(title, expanded=not compact or title.startswith("1."))


def _inputs(
    gpus: Mapping[str, GPUSpec], shipped: tuple[PriceRow, ...]
) -> tuple[Traffic, pd.DataFrame, calibrate.BenchmarkFile | None, bool]:
    """The four steps, Advanced and the Plan button, in the sidebar or (compact) on top."""
    compact = bool(ss["compact"])
    with st.container() if compact else st.sidebar:
        with _step("1. Model", compact):
            _model_step()
        with _step("2. Traffic", compact):
            traffic = _traffic_step()
        with _step("3. Latency target", compact):
            _target_step()
        with _step("4. Hardware and prices", compact):
            table = _hardware_step(gpus, shipped)
        with st.expander("Advanced"):
            benchmarks = _advanced(gpus)
        clicked = st.button("Plan", type="primary", key="plan", width="stretch")
    return traffic, table, benchmarks, clicked


def _slo() -> SLO:
    return SLO(ttft_ms_p95=ss["ttft"], tpot_ms_p95=ss["tpot"], utilization_target=ss["utilization"])


def _resolve_model(model_id: str) -> ModelSpec:
    if not model_id:
        raise ValidationError("enter a Hugging Face model id (org/name)")
    spec = _model(model_id)
    ss.setdefault("resolved_models", {})[model_id] = spec
    return spec


def _done(name: str, seconds: float | None) -> None:
    st.write(f"{name}: {'cached result' if seconds is None else wording.duration(seconds * 1000)}")


def _plan(
    traffic: Traffic,
    table: pd.DataFrame,
    benchmarks: calibrate.BenchmarkFile | None,
    seen: dict[str, WorkloadStats],
    status: Any,
) -> state.PlanRun:
    gpus, _ = _catalogs()
    started = time.perf_counter()
    model = _resolve_model(_model_id())
    rows = calibrate.rows_for_plan(benchmarks, model, gpus)
    _done(STEPS[0], (resolved := time.perf_counter()) - started)
    workload = traffic()
    stats = seen["stats"] = compute_stats(workload)
    _done(STEPS[1], time.perf_counter() - resolved)
    request = state.build_request(
        model=model,
        stats=stats,
        slo=_slo(),
        gpus=gpus,
        prices=state.price_rows(table, gpus),
        gpu_ids=ss["gpu_ids"],
        providers=ss["providers"],
        tensor_parallel=ss["tp"],
        dtypes=ss["dtypes"],
        max_num_seqs=ss["seqs"],
        max_model_len=int(ss["max_model_len"]),
        perf_backend=ss["perf_backend"],
        solver=ss["solver"],
        time_limit_s=ss["time_limit"],
        classes=classify_spec(workload, ss["classes"]),
    )
    options = state.sim_options(stats)
    key = state.cache_key(request, options, workload, rows)
    limit = f"time limit {request.options.time_limit_s:g} s"
    status.update(label=f"Estimating performance, solving ({limit}) and replaying...")
    timings: dict[str, float] = {}  # stays empty when the run comes from the cache
    run = _run(key, request, workload, options, gpus, rows, timings.__setitem__)
    for stage, step in zip(("estimate", "solve", "replay"), STEPS[2:], strict=True):
        _done(f"{step} ({limit})" if stage == "solve" else step, timings.get(stage))
    took = wording.duration((time.perf_counter() - started) * 1000)
    status.update(label=f"Planned in {took}", state="complete")
    ss["plan_key"], ss["plan_workload"] = key, workload  # for the Timeline window selector
    return run


def _on_plan(
    traffic: Traffic, table: pd.DataFrame, benchmarks: calibrate.BenchmarkFile | None
) -> tuple[str, object]:
    """Run one plan; returns ("run", PlanRun), ("error", message) or ("warning", message)."""
    admitted, times = state.admit_plan(ss.get("plan_times", ()), time.time())
    ss["plan_times"] = times
    if not admitted:
        return "warning", (
            f"You have run {presets.MAX_PLANS_PER_HOUR} plans in the last hour. Please wait a "
            "few minutes before planning again."
        )
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    seen: dict[str, WorkloadStats] = {}
    outcome: tuple[str, object]
    status: str
    result: PlanResult | None = None
    try:
        with st.status("Planning...", expanded=False) as progress:
            run = _plan(traffic, table, benchmarks, seen, progress)
        result, status, outcome = run.result, run.result.solver.status, ("run", run)
    except LLMPlanError as exc:
        status = "infeasible" if isinstance(exc, InfeasiblePlan) else "error"
        outcome = ("error", wording.error_text(exc))
    except Exception:
        log.exception("plan failed request_id=%s", request_id)
        message = f"Something went wrong (request id {request_id}); it has been logged."
        status, outcome = "error", ("error", message)
    if "stats" in seen:  # the planner was reached: log the input shapes (section 7)
        usage_log.append(
            usage_log.usage_record(
                request_id=request_id,
                model_id=_model_id(),
                gpu_ids=ss["gpu_ids"],
                stats=seen["stats"],
                slo=_slo(),
                result=result,
                solver_status=status,
                duration_s=time.perf_counter() - started,
            )
        )
    return outcome


def main() -> None:
    """The page: header, inputs (sidebar or compact), one Plan button, the last outcome."""
    st.set_page_config(
        page_title=views.PAGE_TITLE,
        page_icon=views.FAVICON,
        layout="wide",
        initial_sidebar_state="auto",
    )
    gpus, shipped = _catalogs()
    ignored = _seed(gpus, shipped)
    views.header()
    st.toggle("Compact layout", key="compact", on_change=_remember_layout, help=COMPACT_HELP)
    if ignored:
        st.warning(f"Ignored link parameters (unknown or invalid): {', '.join(ignored)}.")
    traffic, table, benchmarks, clicked = _inputs(gpus, shipped)
    inputs = _fingerprint(table)
    stale = ss.get("outcome", ("",))[0] == "run" and ss.get("planned_inputs") != inputs
    again = st.empty()  # the main area's copy of Plan while the shown plan is stale
    clicked |= stale and again.button("Plan again", key="plan_again", type="primary")
    if clicked or ss.pop("run_now", False):
        ss["outcome"], ss["planned_inputs"] = _on_plan(traffic, table, benchmarks), inputs
        ss["share"], stale = _share_after_plan(), False
        again.empty()
    kind, value = ss.get("outcome", ("none", None))
    if kind == "run" and isinstance(value, state.PlanRun):
        key, workload = ss["plan_key"], ss["plan_workload"]
        rewindow = lambda window: _rewindow(key, window, value, workload)  # noqa: E731
        views.results(value, rewindow, stale=stale, share=ss["share"])
        calibrate.report()
    elif kind == "error":
        st.error(str(value))
    elif kind == "warning":
        st.warning(str(value))
    else:
        where = "above" if ss["compact"] else "in the sidebar"
        st.info(f"Choose the inputs {where}, then click Plan.")
    views.footer()


main()
