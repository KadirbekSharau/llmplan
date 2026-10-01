"""scripts/usage_summary.py: counts by model id and GPU set over the log window."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_make_samples import _script

usage_summary = _script("usage_summary")
NOW = datetime(2026, 10, 8, tzinfo=UTC)


def _line(ts: str, model: str, gpus: list[str], status: str = "optimal") -> str:
    record = {"ts": ts, "model_id": model, "gpu_ids": gpus, "solver_status": status}
    return json.dumps(record)


def test_summary_counts_recent_runs_by_model_and_gpu_set() -> None:
    lines = [
        _line("2026-10-07T10:00:00+00:00", "fixture:llama3-8b", ["l4-24gb", "h100-sxm-80gb"]),
        _line("2026-10-06T10:00:00+00:00", "fixture:llama3-8b", ["h100-sxm-80gb", "l4-24gb"]),
        _line("2026-10-05T10:00:00+00:00", "Qwen/Qwen3-8B", [], "infeasible"),
        _line("2026-09-01T10:00:00+00:00", "old/model", ["a10g-24gb"]),  # outside the week
        "not json",
    ]
    text = usage_summary.summarize(lines, days=7, now=NOW)
    assert text.startswith("plan runs since 2026-10-01: 3 (1 malformed lines skipped)\n")
    assert "       2  fixture:llama3-8b" in text
    assert "       2  h100-sxm-80gb,l4-24gb" in text
    assert "       1  (none)" in text
    assert "       1  infeasible" in text
    assert "old/model" not in text


def test_main_reads_the_log_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text(_line(datetime.now(UTC).isoformat(), "m", ["g"]) + "\n", encoding="utf-8")
    usage_summary.main([str(path), "--days", "1"])
    assert ": 1 (0 malformed lines skipped)" in capsys.readouterr().out
