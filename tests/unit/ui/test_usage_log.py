from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmplan.planner import SLO
from llmplan.ui import usage_log
from tests.fake_planner import sim_plan, stats

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _record(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "request_id": "abc123",
        "model_id": "fixture:llama3-8b",
        "gpu_ids": ["h100-sxm-80gb", "l4-24gb"],
        "stats": stats(2.5),
        "slo": SLO(ttft_ms_p95=500.0, tpot_ms_p95=50.0),
        "result": sim_plan(),
        "solver_status": "optimal",
        "duration_s": 1.23456,
        "now": NOW,
    }
    return usage_log.usage_record(**{**args, **overrides})


def test_record_has_exactly_the_section_7_fields() -> None:
    record = _record()
    assert tuple(record) == usage_log.FIELDS
    assert record["ts"] == "2026-10-01T12:00:00+00:00"
    assert record["n_requests"] == 1000
    assert record["peak_rps"] == 2.5
    assert record["slo"] == {"ttft_ms_p95": 500.0, "tpot_ms_p95": 50.0, "utilization_target": 0.8}
    assert record["cost_usd_per_day"] == sim_plan().cost_usd_per_day
    assert record["duration_s"] == 1.235
    failed = _record(result=None, solver_status="infeasible")
    assert failed["cost_usd_per_day"] is None
    assert failed["baseline_usd_per_day"] is None


def test_append_writes_one_json_line_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(usage_log.ENV_VAR, raising=False)
    assert usage_log.log_path() is None
    assert not usage_log.append(_record())
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv(usage_log.ENV_VAR, str(path))
    assert usage_log.log_path() == path
    assert usage_log.append(_record())
    assert usage_log.append(_record(request_id="def456"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["abc123", "def456"]


def test_an_unwritable_log_never_breaks_a_plan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert not usage_log.append(_record(), tmp_path / "missing" / "usage.jsonl")
    assert "usage log not written" in caplog.text


def test_footer_names_every_field() -> None:
    assert all(field in usage_log.FOOTER for field in usage_log.FIELDS)
