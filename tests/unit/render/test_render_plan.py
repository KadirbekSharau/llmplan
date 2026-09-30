from __future__ import annotations

import json

import pytest

from llmplan import render
from llmplan.perf import ReplicaConfig
from llmplan.planner import plan
from llmplan.render.vllm_cmd import plan_commands, serve_command
from tests.conftest import UseFakePerf
from tests.fake_planner import MODEL, FakePerf, request

CAPACITY = {("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)}


def test_serve_command_dtypes() -> None:
    hf_model = MODEL.model_copy(update={"id": "meta-llama/Llama-3.1-8B-Instruct"})
    bf16 = serve_command(hf_model, ReplicaConfig(max_model_len=8192, max_num_seqs=64))
    assert bf16 == (
        "vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 "
        "--max-num-seqs 64 --max-model-len 8192 --gpu-memory-utilization 0.9 --dtype bfloat16"
    )
    fp8 = serve_command(
        MODEL, ReplicaConfig(max_model_len=4096, dtype="fp8", kv_dtype="fp8", tensor_parallel=2)
    )
    assert fp8.startswith("vllm serve <MODEL_ID> --tensor-parallel-size 2 ")
    assert fp8.endswith("--dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8")
    fp16 = serve_command(MODEL, ReplicaConfig(max_model_len=4096, dtype="fp16"))
    assert fp16.endswith("--dtype float16")
    comment, line = serve_command(MODEL, ReplicaConfig(max_model_len=4096, dtype="int4")).split(
        "\n"
    )
    assert comment.startswith("# int4 weights need a pre-quantized int4 checkpoint")
    assert line.endswith("--dtype auto")


@pytest.fixture
def planned(fake_perf: UseFakePerf):  # type: ignore[no-untyped-def]
    fake_perf(CAPACITY)
    req = request(34)
    return req, plan(req)


def test_plan_commands(planned) -> None:  # type: ignore[no-untyped-def]
    req, result = planned
    lines = plan_commands(req, result).splitlines()
    assert lines[0] == "# 1 replica(s) on 1 x test a-1x (1 x fake-a per instance)"
    assert lines[1].startswith("vllm serve <MODEL_ID> --tensor-parallel-size 1")
    assert lines[2] == "# 8 replica(s) on 1 x test b-8x (8 x fake-b per instance)"
    assert len(lines) == 4


def test_plan_text(planned) -> None:  # type: ignore[no-untyped-def]
    req, result = planned
    out = render.get("text").plan(req, result)
    assert "Cost      $504.00/day  (baseline $576.00/day, saving 12.5%)" in out
    assert "Binding   requests" in out
    assert "Demand    34 req/s, 0 output tokens/s (peak 60 s window)" in out
    assert "  test      b-8x                8 x fake-b                 1       $456.00" in out
    assert "  8 x bf16 tp1 max_num_seqs 64 on test b-8x  (perf table/measured)" in out
    assert "    vllm serve <MODEL_ID> --tensor-parallel-size 1" in out
    assert "Candidates (top 2 of 2 by $/hour per req/s)" in out
    assert "  eligible       0.5938  test b-8x tp1 bf16 seqs64" in out
    assert "  - dominance pruning removed 0 of 2 eligible candidates" in out


def test_plan_text_without_baseline(fake_perf: UseFakePerf) -> None:
    fake_perf(CAPACITY)
    homogeneous = request(34, homogeneous=True)
    out = render.get("text").plan(homogeneous, plan(homogeneous))
    assert "(homogeneous requested)" in out
    capped = request(34, max_instances_per_row=1)
    assert "(no homogeneous fleet)" in render.get("text").plan(capped, plan(capped))


def test_plan_json(planned) -> None:  # type: ignore[no-untyped-def]
    req, result = planned
    doc = json.loads(render.get("json").plan(req, result))
    assert doc["cost_usd_per_day"] == 504.0
    assert doc["baseline"]["cost_usd_per_day"] == 576.0
    assert doc["request"]["model"] == "fixture:llama3-8b"
    assert doc["request"]["options"]["perf_backend"] == "fake"
    assert doc["solver"]["backend"] == "highs"
    assert "solve_time_s" not in doc["solver"]  # wall-clock; kept out of JSON
    assert len(doc["candidates"]) == 2
