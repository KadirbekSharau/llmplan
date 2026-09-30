"""Chunked CSV reading and row validation shared by every trace format (M2 section 4.4).

A format maps its source columns to roles (`time`, `input_tokens`, `output_tokens`,
optionally `model`, `tenant`) and says whether `time` is seconds or an ISO-8601 datetime.
This module does the rest: size cap, header check, chunked parsing, dropping invalid rows,
normalizing arrivals to start at 0.0, and a stable sort.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from llmplan.errors import ValidationError, WorkloadFormatError
from llmplan.workload.schema import Workload

log = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 2 * 2**30
CHUNK_ROWS = 1_000_000
MAX_TOKENS = 2**31 - 1  # larger counts are treated as corrupt rows
NOTE_DROPPED_FRACTION = 0.05
MAX_DROPPED_FRACTION = 0.5

TimeKind = Literal["seconds", "datetime"]


def read_header(path: Path) -> tuple[list[str], list[str] | None]:
    """Return the header row and the first data row (None if the file has no data rows)."""
    if not path.is_file():
        raise ValidationError(f"trace {path} is not a readable file")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.reader(handle)
            header = next(rows, None)
            first = next(rows, None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError(f"cannot read trace {path}: {exc}") from None
    if not header:
        raise WorkloadFormatError(f"{path.name}: file is empty (no header row)")
    return header, first


def _check_size(path: Path, max_bytes: int) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValidationError(f"trace {path.name} is {size} bytes; the limit is {max_bytes} bytes")


def _times(
    series: pd.Series, kind: TimeKind
) -> tuple[npt.NDArray[np.int64 | np.float64], npt.NDArray[np.bool_]]:
    if kind == "seconds":
        seconds = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        return seconds, np.isfinite(seconds)
    stamps = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    valid = stamps.notna().to_numpy()
    nanos = stamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").view(np.int64)
    return nanos, valid


def _tokens(series: pd.Series, lo: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(values) & (values == np.floor(values))
        valid &= (values >= lo) & (values <= MAX_TOKENS)
    return np.where(valid, values, 0).astype(np.int64), valid


def _chunks(path: Path, columns: Mapping[str, str], kind: TimeKind) -> Iterator[pd.DataFrame]:
    dtype = {columns[r]: "string" for r in ("model", "tenant") if r in columns}
    if kind == "datetime":
        dtype[columns["time"]] = "string"
    try:
        yield from pd.read_csv(
            path,
            usecols=list(columns.values()),
            dtype=dtype,
            chunksize=CHUNK_ROWS,
            encoding="utf-8",
            float_precision="round_trip",
        )
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise WorkloadFormatError(f"{path.name}: cannot parse CSV: {first_line}") from None


def read_trace(
    path: Path, *, fmt: str, columns: Mapping[str, str], time_kind: TimeKind, max_bytes: int
) -> Workload:
    """Parse the CSV at `path` into a `Workload` using `columns` (role -> source column).

    Drops (and counts in `dropped_rows`) rows whose time is unparsable, whose token counts
    are missing, non-integral, out of range (`input_tokens < 1`, `output_tokens < 0`, or
    above 2**31 - 1), and adds a note when more than 5% of rows are dropped. Raises
    `ValidationError` when the file is missing or larger than `max_bytes`, and
    `WorkloadFormatError` when a column is missing, the CSV is malformed, it has no data
    rows, or more than 50% of rows are dropped.
    """
    header, _ = read_header(path)
    _check_size(path, max_bytes)
    missing = [c for c in columns.values() if c not in header]
    if missing:
        raise WorkloadFormatError(f"{path.name}: format {fmt!r} requires column(s) {missing}")

    times: list[npt.NDArray[np.int64 | np.float64]] = []
    inputs: list[npt.NDArray[np.int64]] = []
    outputs: list[npt.NDArray[np.int64]] = []
    labels: dict[str, list[pd.Series]] = {r: [] for r in ("model", "tenant") if r in columns}
    total = 0
    for chunk in _chunks(path, columns, time_kind):
        time, time_ok = _times(chunk[columns["time"]], time_kind)
        n_in, in_ok = _tokens(chunk[columns["input_tokens"]], lo=1)
        n_out, out_ok = _tokens(chunk[columns["output_tokens"]], lo=0)
        keep = time_ok & in_ok & out_ok
        total += len(chunk)
        times.append(time[keep])
        inputs.append(n_in[keep])
        outputs.append(n_out[keep])
        for role, parts in labels.items():
            parts.append(chunk[columns[role]].astype("string")[keep])

    kept = sum(len(t) for t in times)
    dropped = total - kept
    if total == 0:
        raise WorkloadFormatError(f"{path.name}: no data rows")
    notes: list[str] = []
    if dropped > MAX_DROPPED_FRACTION * total:
        raise WorkloadFormatError(
            f"{path.name}: {dropped} of {total} rows are invalid for format {fmt!r}; "
            "the file is probably another format"
        )
    if dropped > NOTE_DROPPED_FRACTION * total:
        notes.append(f"dropped {dropped} of {total} rows ({100 * dropped / total:.1f}%) as invalid")

    arrival = _relative_seconds(np.concatenate(times))
    order = np.argsort(arrival, kind="stable")
    frame = pd.DataFrame(
        {
            "arrival_s": arrival[order],
            "input_tokens": np.concatenate(inputs)[order],
            "output_tokens": np.concatenate(outputs)[order],
            "model": _labels(labels.get("model"), order),
            "tenant": _labels(labels.get("tenant"), order),
        }
    )
    log.info("parsed trace format=%s rows=%d dropped=%d", fmt, kept, dropped)
    return Workload(
        source=str(path), format=fmt, frame=frame, dropped_rows=dropped, notes=tuple(notes)
    )


def _relative_seconds(raw: npt.NDArray[np.int64 | np.float64]) -> npt.NDArray[np.float64]:
    seconds = np.asarray(raw - raw.min(), dtype=np.float64)  # exact integer subtraction first
    if raw.dtype == np.int64:  # nanoseconds
        seconds /= 1e9
    return seconds


def _labels(parts: list[pd.Series] | None, order: npt.NDArray[np.intp]) -> pd.Series:
    if parts is None:
        return pd.Series(pd.NA, index=pd.RangeIndex(len(order)), dtype="string")
    joined = pd.concat(parts, ignore_index=True).astype("string")
    return joined.iloc[order].reset_index(drop=True)
