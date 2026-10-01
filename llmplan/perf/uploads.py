"""Your own benchmark rows, for one session (M8_DESIGN.md section 5).

Two inputs are accepted: an llmplan CSV (the `BenchmarkRow` columns without `source_url`
and `as_of`) and the JSON that `vllm bench serve --save-result` writes (field names checked
against vLLM v0.30.0, docs/milestones/M8_NOTES.md). Every row is validated like a shipped
row (schema, GPU id, the model being planned, the physical floor); failures are reported
per row and the valid rows are used. Rows are tagged `source_url = "user-upload"`, live
only in memory, and reach the planner through the `backends=` override of
`llmplan.perf.estimate` (`upload_backends`); nothing here writes a file or logs content.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.errors import ValidationError
from llmplan.perf.benchmarks import (
    USER_UPLOAD,
    BenchmarkRow,
    BenchmarkTable,
    default_table,
    row_problem,
)
from llmplan.perf.estimate import PerfBackend
from llmplan.perf.table import TableBackend
from llmplan.types import DType

MAX_UPLOAD_BYTES = 5_000_000
MAX_UPLOAD_ROWS = 500
CSV_COLUMNS = tuple(n for n in BenchmarkRow.model_fields if n not in ("source_url", "as_of"))
OPTIONAL_CSV_COLUMNS = frozenset({"ttft_ms_p50", "ttft_ms_p95", "tpot_ms_p50", "tpot_ms_p95"})


class VllmRun(BaseModel):
    """What a vLLM benchmark JSON does not record, supplied by the user (UI form, CLI
    flags): the GPU, tensor parallelism, weight dtype and vLLM version of the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_id: str = Field(min_length=1)
    tensor_parallel: int = Field(default=1, ge=1)
    dtype: DType = "bf16"
    engine_version: str = Field(default="unknown", min_length=1)


class RowRejection(BaseModel):
    """One upload row that was not used: its 0-based index in the file and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    reason: str


class BenchmarkUpload(BaseModel):
    """A validated upload: the rows to use, the rows rejected (with reasons), and notes
    on values the converter had to choose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[BenchmarkRow, ...]
    rejected: tuple[RowRejection, ...]
    notes: tuple[str, ...]


def _rows_from_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    header = [name.strip() for name in reader.fieldnames or ()]
    missing = [c for c in CSV_COLUMNS if c not in header and c not in OPTIONAL_CSV_COLUMNS]
    if missing:
        raise ValidationError(f"benchmark CSV is missing columns: {', '.join(missing)}")
    unknown = [c for c in header if c not in CSV_COLUMNS]
    if unknown:
        raise ValidationError(f"benchmark CSV has unknown columns: {', '.join(unknown)}")
    rows = []
    for record in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in record.items()}
        cells = {k: (None if v in ("", None) else v) for k, v in row.items()}
        rows.append(dict.fromkeys(OPTIONAL_CSV_COLUMNS) | cells)
    return rows


def _vllm_row(doc: Any, run: VllmRun, notes: list[str], index: int) -> dict[str, Any] | str:
    """One `BenchmarkRow` input from one vLLM result object (fields as in vLLM v0.30.0's
    `vllm/benchmarks/serve.py`), or why it cannot be converted."""
    if not isinstance(doc, dict):
        return "not a JSON object"
    completed = doc.get("completed")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 1:
        return f"'completed' must be a positive integer, got {completed!r}"
    concurrency = doc.get("max_concurrency")
    if concurrency is None:
        concurrency = doc.get("num_prompts")
        notes.append(
            f"row {index}: max_concurrency is null (no client-side limit); concurrency taken "
            f"as num_prompts {concurrency!r}, an upper bound on the requests in flight"
        )

    def per_request(key: str) -> Any:
        total = doc.get(key)
        if not isinstance(total, int | float) or isinstance(total, bool):
            return None
        return round(total / completed)

    return {
        "model_id": doc.get("model_id"),
        "gpu_id": run.gpu_id,
        "engine": "vllm",
        "engine_version": run.engine_version,
        "tensor_parallel": run.tensor_parallel,
        "dtype": run.dtype,
        "concurrency": concurrency,
        "input_len": per_request("total_input_tokens"),
        "output_len": per_request("total_output_tokens"),
        "output_tokens_per_s": doc.get("output_throughput"),
        "ttft_ms_p50": doc.get("median_ttft_ms"),
        "ttft_ms_p95": doc.get("p95_ttft_ms"),
        "tpot_ms_p50": doc.get("median_tpot_ms"),
        "tpot_ms_p95": doc.get("p95_tpot_ms"),
    }


def _rows_from_json(text: str, run: VllmRun | None, notes: list[str]) -> list[dict[str, Any] | str]:
    try:
        docs: list[Any] = [json.loads(text)]
    except json.JSONDecodeError:
        try:  # `vllm bench serve --append-result`: one JSON object per line
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValidationError(f"benchmark JSON is not valid JSON: {exc.msg}") from None
    if run is None:
        raise ValidationError(
            "a vLLM benchmark JSON does not record the GPU; give its gpu id (and tensor "
            "parallel, dtype, vLLM version)"
        )
    return [_vllm_row(doc, run, notes, i) for i, doc in enumerate(docs)]


def _schema_problem(raw: dict[str, Any] | str, today: date) -> BenchmarkRow | str:
    if isinstance(raw, str):
        return raw
    try:
        return BenchmarkRow.model_validate({**raw, "source_url": USER_UPLOAD, "as_of": today})
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "row"
        return f"field {field!r}: {err['msg']}"


def load_upload(
    data: bytes,
    model: ModelSpec,
    gpus: Mapping[str, GPUSpec],
    *,
    run: VllmRun | None = None,
    today: date | None = None,
) -> BenchmarkUpload:
    """Parse and validate an uploaded benchmark file for a plan of `model`.

    `data` is an llmplan CSV or a vLLM `--save-result` JSON (detected from the first
    character); a JSON upload needs `run` for the fields vLLM does not record. Each row is
    checked like a shipped row: schema, `gpu_id` in `gpus`, `model_id` naming `model`
    (directly or through the shipped aliases), tensor parallelism dividing the heads, and
    the physical floor. Rejected rows are returned with their 0-based index and reason.
    Raises `ValidationError` for a file over 5 MB or 500 rows, undecodable text, a CSV
    without the required columns, invalid JSON, or JSON without `run`.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"benchmark upload is {len(data):,} bytes; the limit is {MAX_UPLOAD_BYTES:,}"
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError("benchmark upload is not UTF-8 text") from None
    notes: list[str] = []
    is_json = text.lstrip().startswith(("{", "["))
    raw_rows: list[dict[str, Any] | str] = []
    raw_rows += _rows_from_json(text, run, notes) if is_json else _rows_from_csv(text)
    if len(raw_rows) > MAX_UPLOAD_ROWS:
        raise ValidationError(
            f"benchmark upload has {len(raw_rows):,} rows; the limit is {MAX_UPLOAD_ROWS}"
        )
    if not raw_rows:
        raise ValidationError("benchmark upload has no rows")
    aliases = default_table().aliases
    wanted = aliases.get(model.id, model.id)
    rows: list[BenchmarkRow] = []
    rejected: list[RowRejection] = []
    for index, raw in enumerate(raw_rows):
        row = _schema_problem(raw, today or date.today())
        if isinstance(row, BenchmarkRow):
            row = _row_against(row, model, wanted, aliases, gpus)
        if isinstance(row, str):
            rejected.append(RowRejection(index=index, reason=row))
        else:
            rows.append(row)
    return BenchmarkUpload(rows=tuple(rows), rejected=tuple(rejected), notes=tuple(notes))


def _row_against(
    row: BenchmarkRow,
    model: ModelSpec,
    wanted: str,
    aliases: Mapping[str, str],
    gpus: Mapping[str, GPUSpec],
) -> BenchmarkRow | str:
    if row.gpu_id not in gpus:
        return f"unknown gpu_id {row.gpu_id!r}"
    if aliases.get(row.model_id, row.model_id) != wanted:
        return f"model_id {row.model_id!r} is not the model being planned ({model.id})"
    problem = row_problem(row, model, gpus[row.gpu_id])
    return row if problem is None else problem


def upload_backends(rows: Sequence[BenchmarkRow]) -> dict[str, PerfBackend]:
    """The `backends=` override that adds `rows` to the shipped table for one call (empty
    when there are no rows). Uploaded rows take precedence where they match (table
    backend), so an estimate they answer cites only `"user-upload"`."""
    if not rows:
        return {}
    shipped = default_table()
    table = BenchmarkTable(rows=(*shipped.rows, *rows), aliases=shipped.aliases)
    return {"table": TableBackend(table)}


def rows_csv(rows: Sequence[BenchmarkRow]) -> str:
    """Rows as an llmplan CSV (the upload format), header first."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        values = row.model_dump()
        writer.writerow(["" if values[c] is None else values[c] for c in CSV_COLUMNS])
    return out.getvalue()
