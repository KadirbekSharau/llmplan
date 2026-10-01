"""Calibrate with your own benchmarks: the web UI side of M8_DESIGN.md section 5.

A sidebar section takes an llmplan CSV or a `vllm bench serve --save-result` JSON (plus
the GPU, tensor parallelism, dtype and vLLM version that file does not record). On Plan
the rows are validated against the chosen model (`llmplan.perf.uploads`) and passed to
the planner through the `backends=` override; they stay in this session's memory, are
never written to disk and never logged. After a plan, the report lists the rows used and
rejected, and offers the prefilled GitHub issue (or the CSV when the rows are too many).
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
Rows = tuple[BenchmarkRow, ...]


@dataclass(frozen=True)
class BenchmarkFile:
    """An uploaded benchmark file, kept in memory only (`data` is never shown or logged);
    `run` holds what a vLLM JSON does not record (None for an llmplan CSV)."""

    data: bytes = field(repr=False)
    run: VllmRun | None


def sidebar(gpus: Mapping[str, GPUSpec], dtypes: Sequence[DType]) -> BenchmarkFile | None:
    """The calibration section (under Advanced since M9); returns the uploaded file with,
    for a vLLM JSON, the GPU, tensor parallelism, dtype and version it does not record."""
    st.subheader("Calibrate with your own benchmarks", anchor=ANCHOR)
    st.caption(
        "An llmplan CSV (the benchmark row columns) or the JSON written by `vllm bench serve "
        "--save-result`, up to 5 MB and 500 rows. Used for this session only: never stored "
        "or logged."
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
    """The valid rows of the upload, checked against the model being planned (none without
    an upload); the full report is kept in this session's state for `report`. Raises
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
