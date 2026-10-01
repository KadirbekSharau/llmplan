"""M7 simulation: incremental KV accounting, class-weighted routing, per-class summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from llmplan import render
from llmplan.cli import app
from llmplan.errors import ValidationError
from llmplan.planner import SLO, plan
from llmplan.simulate import SimOptions, replay, replay_requests, routing
from llmplan.simulate.stepfn import linear_windows
from tests.acceptance.test_m7 import two_class_request, two_class_workload
from tests.fake_planner import sim_plan, workload

runner = CliRunner()


def test_linear_windows_integral_and_max() -> None:
    # 10 tokens at t = 0 growing at 2/s, back to 0 at t = 1.
    integral, maximum = linear_windows(
        np.array([0.0, 1.0]), np.array([10.0, 0.0]), np.array([2.0, 0.0]), np.array([0, 0.5, 2])
    )
    assert integral.tolist() == pytest.approx(
        [5.25, 5.75]
    )  # (10 + 11) / 2 x 0.5, (11 + 12) / 2 x 0.5
    assert maximum.tolist() == pytest.approx([11.0, 12.0])


def test_incremental_kv_grows_linearly() -> None:
    # One request: 200 input, 100 output; service 0.2 + 100 x 0.01 = 1.2 s. KV in use goes
    # from 200 to 300 over 1.2 s: mean over a 60 s window (200 + 300) / 2 x 1.2 / 60 = 5.
    fleet = sim_plan(kv_token_capacity=1000)
    trace = workload([0.0], 200, 100)
    record = replay(fleet, trace).windows[0].replicas[0]
    assert (record.kv_tokens_in_use_mean, record.kv_tokens_in_use_max) == (5, 300)
    full = replay(fleet, trace, options=SimOptions(kv_accounting="full")).windows[0].replicas[0]
    assert (full.kv_tokens_in_use_mean, full.kv_tokens_in_use_max) == (6, 300)  # 300 x 1.2 / 60
    assert replay_requests(fleet, trace).frame["kv_tokens"].tolist() == [300]


def test_incremental_admission_uses_mean_occupancy() -> None:
    # 1,000 KV tokens, requests of 200 + 100: incremental reserves 250 each, so 4 run at once
    # (M5's full reservation of 300 lets 3 run; acceptance test 9.4 pins that).
    fleet = sim_plan(effective_batch=8, kv_token_capacity=1000)
    starts = replay_requests(fleet, workload([0.0] * 10, 200, 100)).frame["start_s"].tolist()
    assert sum(start == 0.0 for start in starts) == 4


def test_deficit_rule_tracks_the_weights() -> None:
    route = routing.class_weighted([0, 1], [[(0, 0.25), (1, 0.75)]], [0] * 8)
    assert [route([0, 0], i) for i in range(8)] == [1, 0, 1, 1, 1, 0, 1, 1]


def test_a_class_without_rules_goes_to_the_least_outstanding_replica() -> None:
    route = routing.class_weighted([0, 1], [[(0, 1.0)], []], [0, 1, 1])
    assert [route([5, 2], i) for i in range(3)] == [0, 1, 1]


def test_class_weighted_on_a_plan_without_classes() -> None:
    fleet = sim_plan(replicas=2)
    trace = workload([0.25 * i for i in range(20)], 100, 40)
    weighted = replay(fleet, trace, options=SimOptions(routing="class_weighted"))
    blind = replay(fleet, trace, options=SimOptions(routing="least_outstanding"))
    assert weighted.windows == blind.windows
    assert weighted.classes == ()
    assert any("plan without classes routes least-outstanding" in a for a in weighted.assumptions)


def test_class_weighted_rejects_unknown_replica_types() -> None:
    trace = two_class_workload()
    result = plan(two_class_request(trace))
    stranger = result.routing[0].candidate.model_copy(update={"reason": "elsewhere"})
    broken = result.model_copy(
        update={"routing": (result.routing[0].model_copy(update={"candidate": stranger}),)}
    )
    with pytest.raises(ValidationError, match="names a replica type the plan does not have"):
        replay(broken, trace, options=SimOptions(routing="class_weighted"))


def test_class_summaries_and_text() -> None:
    trace = two_class_workload()
    req = two_class_request(trace)
    result = plan(req)
    timeline = replay(result, trace, slo=req.slo, options=SimOptions(routing="class_weighted"))
    text = render.get("text").timeline(timeline)
    assert "Class 0   300 requests, TTFT p95 50.0 ms, 0.00% TTFT and 0.00% TPOT violations" in text
    shorts = workload([0.1 * i for i in range(10)], 100, 10)
    only_short = replay(result, shorts, options=SimOptions(routing="class_weighted"))
    assert only_short.classes[1].n_requests == 0
    assert only_short.classes[1].ttft_ms_p95 is None
    assert "Class 1   0 requests" in render.get("text").timeline(only_short)


def test_simulate_cli_routes_class_plans_by_weight(tmp_path: Path) -> None:
    trace = two_class_workload()
    req = two_class_request(trace).model_copy(update={"slo": SLO(utilization_target=1.0)})
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(render.get("json").plan(req, plan(req)), encoding="utf-8")
    trace_file = tmp_path / "trace.csv"
    trace.frame[["arrival_s", "input_tokens", "output_tokens"]].to_csv(trace_file, index=False)
    args = ["simulate", "--plan", str(plan_file), "--trace", str(trace_file), "--format", "json"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["options"]["routing"] == "class_weighted"
    assert doc["options"]["kv_accounting"] == "incremental"
    assert [c["n_requests"] for c in doc["classes"]] == [300, 300]
    blocked = runner.invoke(app, [*args, "--requests-csv", str(tmp_path / "missing" / "r.csv")])
    assert blocked.exit_code == 2
    assert "is not writable" in blocked.stderr
