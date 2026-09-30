from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmplan import render
from llmplan.errors import ValidationError
from llmplan.planner import SLO, plan
from llmplan.planner.result import load_plan_json
from tests.conftest import UseFakePerf
from tests.fake_planner import FakePerf, request

CAPACITY = {("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)}


def test_plan_json_round_trips(fake_perf: UseFakePerf, tmp_path: Path) -> None:
    fake_perf(CAPACITY)
    req = request(34, slo=SLO(ttft_ms_p95=250.0, utilization_target=1.0))
    result = plan(req)
    assert result.baseline is not None
    path = tmp_path / "plan.json"
    path.write_text(render.get("json").plan(req, result), encoding="utf-8")
    loaded, slo = load_plan_json(path)
    assert loaded.model_dump_json() == result.model_dump_json()
    assert loaded.solver.solve_time_s == 0.0
    assert loaded.baseline is not None
    assert loaded.baseline.solver.solve_time_s == 0.0
    assert slo == req.slo


def test_bare_plan_result_has_no_slo(fake_perf: UseFakePerf, tmp_path: Path) -> None:
    fake_perf(CAPACITY)
    result = plan(request(34))
    path = tmp_path / "bare.json"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    loaded, slo = load_plan_json(path)
    assert loaded.model_dump_json() == result.model_dump_json()
    assert slo is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "is not valid JSON"),
        (b"\xff\xfe", "is not valid JSON"),
        (b"[1, 2]", "does not hold a JSON object"),
        (b'{"fleet": []}', "is not a plan result: replicas: Field required"),
    ],
)
def test_bad_plan_files(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "plan.json"
    path.write_bytes(content)
    with pytest.raises(ValidationError, match=message):
        load_plan_json(path)


def test_bad_recorded_slo(fake_perf: UseFakePerf, tmp_path: Path) -> None:
    fake_perf(CAPACITY)
    doc = json.loads(plan(request(34)).model_dump_json())
    doc["request"] = {"slo": {"ttft_ms_p95": -1}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValidationError, match="ttft_ms_p95"):
        load_plan_json(path)


def test_missing_and_oversized_plan_files(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="is not readable"):
        load_plan_json(tmp_path / "missing.json")
    path = tmp_path / "big.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="larger than 1 bytes"):
        load_plan_json(path, max_bytes=1)
