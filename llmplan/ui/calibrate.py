"""Calibrate with your own benchmarks: the web UI side of M8_DESIGN.md section 5. An llmplan
CSV or `vllm bench serve --save-result` JSON is validated on Plan against the chosen model
(`llmplan.perf.uploads`) and reaches the planner through `backends=`; it stays in this
session's memory, never written or logged. After a plan, a report lists the rows used and
rejected and offers the prefilled GitHub issue (or the CSV when the rows are too many).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import streamlit as st

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.perf.contribute import contribute_url
from llmplan.perf.uploads import BenchmarkUpload, VllmRun, load_upload, rows_csv
from llmplan.types import DType

ANCHOR = "calibrate"
REPORT_KEY = "benchmark_upload"  # session state: the last plan's validated upload
RUN_KEYS = ("bench_gpu", "bench_tp", "bench_dtype", "bench_version")  # vLLM JSON details
Rows = tuple[BenchmarkRow, ...]


@dataclass(frozen=True)
class BenchmarkFile:
    """An upload kept in memory only; `run`: what a vLLM JSON does not record (CSV: None)."""

    data: bytes = field(repr=False)
    run: VllmRun | None


def sidebar(gpus: Mapping[str, GPUSpec], dtypes: Sequence[DType]) -> BenchmarkFile | None:
    """The calibration section (under Advanced, M9): the uploaded file and, for a vLLM JSON,
    the GPU, tensor parallelism, dtype and version that it does not record."""
    st.subheader("Calibrate with your own benchmarks", anchor=ANCHOR)
    st.caption(
        "An llmplan CSV or a `vllm bench serve --save-result` JSON (5 MB, 500 rows), for this "
        "session only: never stored or logged."
    )
    uploaded = st.file_uploader("Benchmark file", type=["csv", "json"], key="bench_upload")
    if uploaded is None:
        return None
    if not uploaded.name.lower().endswith(".json"):
        return BenchmarkFile(data=uploaded.getvalue(), run=None)
    st.caption("vLLM JSON details (not in the file)")
    run = VllmRun(
        gpu_id=str(st.selectbox("GPU", list(gpus), key="bench_gpu")),
        tensor_parallel=int(st.number_input("Tensor parallel", 1, 64, 1, key="bench_tp")),
        dtype=st.selectbox("dtype", list(dtypes), key="bench_dtype"),
        engine_version=st.text_input("vLLM version", "unknown", key="bench_version").strip()
        or "unknown",
    )
    return BenchmarkFile(data=uploaded.getvalue(), run=run)


def rows_for_plan(
    file: BenchmarkFile | None, model: ModelSpec, gpus: Mapping[str, GPUSpec]
) -> Rows:
    """The upload's rows valid for `model` (kept with the rejections for `report`); raises
    `ValidationError` for a file-level problem (size, rows, format)."""
    upload = None if file is None else load_upload(file.data, model, gpus, run=file.run)
    st.session_state[REPORT_KEY] = upload
    return () if upload is None else upload.rows


def report() -> None:
    """After a plan: rows used and rejected (with reasons), notes, and how to contribute."""
    upload: BenchmarkUpload | None = st.session_state.get(REPORT_KEY)
    if upload is None:
        return
    st.subheader("Your benchmarks")
    total = len(upload.rows) + len(upload.rejected)
    st.markdown(f"{len(upload.rows)} of {total} uploaded rows used for this plan.")
    for rejection in upload.rejected:
        st.caption(f"Row {rejection.index} rejected: {rejection.reason}")
    for note in upload.notes:
        st.caption(note)
    if not upload.rows:
        return
    url, with_rows = contribute_url(upload.rows)
    if with_rows:
        st.link_button("Contribute these rows (opens a GitHub issue)", url)
        return
    st.caption("Too many rows for a link: download them and attach the CSV to the issue.")
    st.download_button(
        "Download the rows as CSV", rows_csv(upload.rows), "llmplan-benchmarks.csv", "text/csv"
    )
    st.link_button("Open a GitHub issue", url)
