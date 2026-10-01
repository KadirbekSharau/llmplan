"""Anonymous usage log of the web UI (M6_DESIGN.md section 7): append-only JSON lines at
`LLMPLAN_USAGE_LOG` (off when unset), exactly `FIELDS` per plan run. Never the uploaded
rows, the price edits, or anything identifying the visitor (no IPs, no session ids).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmplan.planner import SLO, PlanResult
from llmplan.workload import WorkloadStats

log = logging.getLogger(__name__)

ENV_VAR = "LLMPLAN_USAGE_LOG"
FIELDS = (
    "ts",
    "request_id",
    "model_id",
    "gpu_ids",
    "n_requests",
    "peak_rps",
    "slo",
    "cost_usd_per_day",
    "baseline_usd_per_day",
    "solver_status",
    "duration_s",
)
FOOTER = (
    f"Usage log: when the operator enables it, each plan run records only {', '.join(FIELDS[:-1])} "
    f"and {FIELDS[-1]}; uploaded traces, price edits and IP addresses are never logged or kept."
)
_WRITE_LOCK = threading.Lock()


def log_path() -> Path | None:
    """The usage log path from `LLMPLAN_USAGE_LOG`, or None (logging disabled)."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def usage_record(
    *,
    request_id: str,
    model_id: str,
    gpu_ids: Sequence[str],
    stats: WorkloadStats,
    slo: SLO,
    result: PlanResult | None,
    solver_status: str,
    duration_s: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One log line's fields, in `FIELDS` order. `result` is None when the plan failed;
    `solver_status` is then `"infeasible"` or `"error"`, and both costs are null."""
    baseline = None if result is None or result.baseline is None else result.baseline
    return {
        "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "request_id": request_id,
        "model_id": model_id,
        "gpu_ids": list(gpu_ids),
        "n_requests": stats.n_requests,
        "peak_rps": stats.peak_window_rps,
        "slo": slo.model_dump(mode="json"),
        "cost_usd_per_day": None if result is None else result.cost_usd_per_day,
        "baseline_usd_per_day": None if baseline is None else baseline.cost_usd_per_day,
        "solver_status": solver_status,
        "duration_s": round(duration_s, 3),
    }


def append(record: dict[str, Any], path: Path | None = None) -> bool:
    """Append `record` as one JSON line to `path` (default: `log_path()`); False when logging
    is off or the file cannot be written (logged as a warning; the UI keeps working)."""
    target = path or log_path()
    if target is None:
        return False
    line = json.dumps(record, separators=(",", ":")) + "\n"
    try:
        with _WRITE_LOCK, target.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        log.warning("usage log not written: %s", exc.strerror)
        return False
    return True
