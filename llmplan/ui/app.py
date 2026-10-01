"""llmplan web UI (M6): pick a model, traffic, a latency target and prices; get the cheapest
fleet, its vLLM command lines and the utilization timeline.

Run with `llmplan ui`. The page holds no planning logic: results come from `llmplan.*`
through `llmplan.ui.state` (M8: own benchmarks through `llmplan.ui.calibrate`), recomputed
only on Plan, and logged by `usage_log` when enabled. Limits (M6_DESIGN.md section 6): 50 MB
uploads checked before parsing, 30 s solver limit, 200,000 requests, 30 plans per hour.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

import pandas as pd
import streamlit as st

from llmplan import render
from llmplan.catalog.hardware import GPUSpec, PriceRow, load_gpus, load_prices
from llmplan.catalog.models import FIXTURE_PREFIX, ModelSpec, load_model
from llmplan.errors import InfeasiblePlan, LLMPlanError, ValidationError
from llmplan.perf.uploads import upload_backends
from llmplan.planner import SLO, PlanRequest, PlanResult
from llmplan.simulate import SimOptions
from llmplan.types import Commitment, DType
from llmplan.ui import calibrate, presets, state, usage_log, views
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

CUSTOM_MODEL = "Other Hugging Face id"
TRAFFIC_MODES = ("Preset sample", "Upload CSV", "Synthetic")
COMMITMENTS: tuple[Commitment, ...] = ("on_demand", "reserved_1y", "reserved_3y", "spot")
DISTRIBUTION_HELP = "fixed:N, lognormal:MEAN:SIGMA[:LO:HI] (underlying normal), or uniform:LO:HI"
CLASSES_HELP = "Input x output token bins (median splits for 2x2); 1 sizes for the mean request"


@dataclass(frozen=True)
class Inputs:
    """Everything the sidebar collected; `workload` builds or returns the traffic."""

    model_id: str
    workload: Callable[[], Workload]
    slo: SLO
    price_table: pd.DataFrame
    gpu_ids: Sequence[str]
    providers: Sequence[str]
    tensor_parallel: Sequence[int]
    dtypes: Sequence[DType]
    max_num_seqs: Sequence[int]
    max_model_len: int
    perf_backend: str
    solver: str
    time_limit_s: float
    classes: str
    benchmarks: calibrate.BenchmarkFile | None


@st.cache_resource(show_spinner=False)
def _catalogs() -> tuple[Mapping[str, GPUSpec], tuple[PriceRow, ...]]:
    gpus = load_gpus()
    return gpus, load_prices(gpus=gpus)


@st.cache_data(ttl=86_400, show_spinner=False)
def _model(model_id: str) -> ModelSpec:
    return load_model(model_id)  # HF ids are validated by the M1 fetcher before any request


@st.cache_resource(show_spinner=False)
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
) -> state.PlanRun:
    return state.run_plan(_request, _workload, _options, _gpus, upload_backends(_rows))


def _fail(message: str) -> Callable[[], Workload]:
    def raise_it() -> NoReturn:
        raise ValidationError(message)

    return raise_it


def _known_model(model_id: str) -> ModelSpec | None:
    """Fixtures resolve offline right away; Hugging Face ids once a plan has fetched them."""
    if model_id.startswith(FIXTURE_PREFIX):
        return _model(model_id)  # the picker offers shipped fixtures only
    resolved: dict[str, ModelSpec] = st.session_state.get("resolved_models", {})
    return resolved.get(model_id)


def _model_section() -> str:
    st.header("Model")
    choices = [*presets.FIXTURE_MODELS, *presets.POPULAR_MODELS, CUSTOM_MODEL]
    choice = st.selectbox(
        "Model", choices, index=choices.index(presets.DEFAULT_MODEL), key="model_choice"
    )
    model_id = choice
    if choice == CUSTOM_MODEL:
        model_id = st.text_input("Hugging Face id (org/name)", key="model_custom").strip()
    st.caption(
        "Hugging Face configs are fetched when you click Plan. Gated models need HF_TOKEN "
        "set on the server; it is never shown or logged."
    )
    spec = _known_model(model_id)
    if spec is not None:
        with st.expander("Model info"):
            st.code(render.get("text").model_info(spec), language=None)
    return model_id


def _upload_source() -> Callable[[], Workload]:
    uploaded = st.file_uploader("Trace CSV (up to 50 MB)", type=["csv"], key="upload")
    st.caption("Formats: generic csv, Azure 2023/2024, BurstGPT. Parsed in memory, never stored.")
    if uploaded is None:
        st.session_state.pop("upload_parsed", None)
        return _fail("upload a CSV trace, or choose a preset or synthetic traffic")
    if uploaded.size > presets.MAX_UPLOAD_BYTES:
        message = (
            f"Upload refused: {uploaded.size:,} bytes is over the 50 MB "
            f"({presets.MAX_UPLOAD_BYTES:,} bytes) limit; nothing was parsed."
        )
        st.error(message)
        return _fail(message)
    cached = st.session_state.get("upload_parsed")
    if cached is None or cached[0] != uploaded.file_id:
        try:
            parsed: Workload | str = load_workload(
                InMemoryTrace("uploaded.csv", uploaded.getvalue()),
                max_bytes=presets.MAX_UPLOAD_BYTES,
            )
        except LLMPlanError as exc:
            parsed = str(exc)
        cached = (uploaded.file_id, parsed)
        st.session_state["upload_parsed"] = cached
    parsed = cached[1]
    if isinstance(parsed, str):
        st.error(parsed)
        return _fail(parsed)
    views.stats_caption(parsed, compute_stats(parsed))
    workload = parsed
    return lambda: workload


def _synthetic_source() -> Callable[[], Workload]:
    rate = st.number_input("Rate (req/s)", 0.01, 1000.0, 2.0, key="syn_rate")
    duration = st.number_input("Duration (s)", 1.0, 86_400.0, 3600.0, 60.0, key="syn_duration")
    in_spec = st.text_input("Input tokens", presets.DEFAULT_IN_TOKENS, help=DISTRIBUTION_HELP)
    out_spec = st.text_input("Output tokens", presets.DEFAULT_OUT_TOKENS, help=DISTRIBUTION_HELP)
    seed = st.number_input("Seed", 0, 2**31 - 1, 0, 1, key="syn_seed")

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


def _traffic_section() -> Callable[[], Workload]:
    st.header("Traffic")
    mode = st.radio("Source", TRAFFIC_MODES, key="traffic_mode")
    if mode == "Upload CSV":
        return _upload_source()
    if mode == "Synthetic":
        return _synthetic_source()
    labels = {p.key: p.label for p in presets.PRESETS}
    key = st.selectbox("Preset", list(labels), format_func=labels.__getitem__, key="preset")
    workload, stats = _preset(key)
    views.stats_caption(workload, stats)
    return lambda: workload


def _slo_section() -> SLO:
    st.header("Latency target")
    default = presets.DEFAULT_SLO
    ttft = st.number_input("TTFT p95 (ms)", 0.1, 600_000.0, default.ttft_ms_p95, key="ttft")
    tpot = st.number_input("TPOT p95 (ms)", 0.1, 60_000.0, default.tpot_ms_p95, key="tpot")
    utilization = st.number_input(
        "Utilization target", 0.05, 1.0, default.utilization_target, 0.05, key="utilization"
    )
    return SLO(ttft_ms_p95=ttft, tpot_ms_p95=tpot, utilization_target=utilization)


def _reset_prices() -> None:
    st.session_state["price_table"] = state.prices_frame(_catalogs()[1])
    st.session_state["price_version"] = st.session_state.get("price_version", 0) + 1


def _hardware_section() -> tuple[pd.DataFrame, list[str], list[str]]:
    st.header("Hardware and prices")
    gpus, shipped = _catalogs()
    if "price_table" not in st.session_state:
        _reset_prices()
    priced = sorted({row.gpu_id for row in shipped})
    gpu_ids = st.multiselect("GPUs", list(gpus), default=priced, key="gpu_ids")
    table = st.data_editor(
        st.session_state["price_table"],
        key=f"price_editor_{st.session_state['price_version']}",
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
    chosen = st.multiselect("Providers", providers, default=providers, key="providers")
    return table, gpu_ids, chosen


def _sidebar() -> Inputs:
    with st.sidebar:
        model_id = _model_section()
        workload = _traffic_section()
        slo = _slo_section()
        table, gpu_ids, providers = _hardware_section()
        with st.expander("Advanced"):
            tp = st.multiselect("Tensor parallel", presets.TP_CHOICES, presets.TP_CHOICES)
            dtypes = st.multiselect("dtype", presets.DTYPE_CHOICES, presets.DEFAULT_DTYPES)
            seqs = st.multiselect(
                "max_num_seqs", presets.MAX_NUM_SEQS_CHOICES, presets.DEFAULT_MAX_NUM_SEQS
            )
            max_model_len = st.number_input(
                "max_model_len", 256, 1_048_576, presets.DEFAULT_MAX_MODEL_LEN, 256
            )
            classes = st.selectbox(
                "Request-size classes", presets.CLASS_CHOICES, key="classes", help=CLASSES_HELP
            )
            perf_backend = st.selectbox("Perf backend", presets.PERF_BACKENDS)
            solver = st.selectbox("Solver", presets.SOLVERS)
            time_limit = st.number_input(
                "Solver time limit (s)", 1.0, presets.MAX_TIME_LIMIT_S, presets.DEFAULT_TIME_LIMIT_S
            )
        benchmarks = calibrate.sidebar(_catalogs()[0], presets.DTYPE_CHOICES)
    return Inputs(
        model_id=model_id,
        workload=workload,
        slo=slo,
        price_table=table,
        gpu_ids=gpu_ids,
        providers=providers,
        tensor_parallel=tp,
        dtypes=dtypes,
        max_num_seqs=seqs,
        max_model_len=int(max_model_len),
        perf_backend=perf_backend,
        solver=solver,
        time_limit_s=time_limit,
        classes=classes,
        benchmarks=benchmarks,
    )


def _resolve_model(model_id: str) -> ModelSpec:
    if not model_id:
        raise ValidationError("enter a Hugging Face model id (org/name)")
    spec = _model(model_id)
    st.session_state.setdefault("resolved_models", {})[model_id] = spec
    return spec


def _plan(inputs: Inputs, seen: dict[str, WorkloadStats]) -> state.PlanRun:
    gpus, _ = _catalogs()
    model = _resolve_model(inputs.model_id)
    rows = calibrate.rows_for_plan(inputs.benchmarks, model, gpus)
    workload = inputs.workload()
    stats = seen["stats"] = compute_stats(workload)
    request = state.build_request(
        model=model,
        stats=stats,
        slo=inputs.slo,
        gpus=gpus,
        prices=state.price_rows(inputs.price_table, gpus),
        gpu_ids=inputs.gpu_ids,
        providers=inputs.providers,
        tensor_parallel=inputs.tensor_parallel,
        dtypes=inputs.dtypes,
        max_num_seqs=inputs.max_num_seqs,
        max_model_len=inputs.max_model_len,
        perf_backend=inputs.perf_backend,
        solver=inputs.solver,
        time_limit_s=inputs.time_limit_s,
        classes=classify_spec(workload, inputs.classes),
    )
    options = state.sim_options(stats)
    key = state.cache_key(request, options, workload, rows)
    return _run(key, request, workload, options, gpus, rows)


def _on_plan(inputs: Inputs) -> tuple[str, object]:
    """Run one plan; returns ("run", PlanRun), ("error", message) or ("warning", message)."""
    admitted, times = state.admit_plan(st.session_state.get("plan_times", ()), time.time())
    st.session_state["plan_times"] = times
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
        with st.spinner("Planning the fleet and replaying the traffic..."):
            run = _plan(inputs, seen)
        result, status, outcome = run.result, run.result.solver.status, ("run", run)
    except InfeasiblePlan as exc:
        status, outcome = "infeasible", ("error", str(exc))
    except LLMPlanError as exc:
        status, outcome = "error", ("error", str(exc))
    except Exception:
        log.exception("plan failed request_id=%s", request_id)
        message = f"Something went wrong (request id {request_id}); it has been logged."
        status, outcome = "error", ("error", message)
    if "stats" in seen:  # the planner was reached: log the input shapes (section 7)
        usage_log.append(
            usage_log.usage_record(
                request_id=request_id,
                model_id=inputs.model_id,
                gpu_ids=inputs.gpu_ids,
                stats=seen["stats"],
                slo=inputs.slo,
                result=result,
                solver_status=status,
                duration_s=time.perf_counter() - started,
            )
        )
    return outcome


def main() -> None:
    """The page: sidebar inputs, one Plan button, and the last outcome."""
    st.set_page_config(
        page_title=views.PAGE_TITLE,
        page_icon=views.FAVICON,
        layout="wide",
        initial_sidebar_state="auto",
    )
    inputs = _sidebar()
    views.header()
    if st.button("Plan", type="primary", key="plan"):
        st.session_state["outcome"] = _on_plan(inputs)
    kind, value = st.session_state.get("outcome", ("none", None))
    if kind == "run" and isinstance(value, state.PlanRun):
        views.results(value)
        calibrate.report()
    elif kind == "error":
        st.error(str(value))
    elif kind == "warning":
        st.warning(str(value))
    else:
        st.info("Choose the inputs in the sidebar, then click Plan.")
    views.footer()


main()
