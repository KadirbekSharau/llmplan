"""M8 acceptance tests: M8_DESIGN.md section 9. These define done; do not relax them.

9.11 (live Hugging Face checks) is manual: tests/live/test_hf_configs.py, marked `network`,
with the recorded run in docs/milestones/M8_NOTES.md.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from llmplan import render
from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import load_model
from llmplan.cli import app
from llmplan.errors import UnsupportedArchitecture
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, fit
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import expert_weight_bytes, model_info, weight_bytes
from llmplan.perf import ReplicaConfig, estimate
from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.perf.contribute import MAX_URL_CHARS, REPOSITORY_URL, contribute_url
from llmplan.perf.roofline import BANDWIDTH_EFFICIENCY, decode_weight_bytes
from llmplan.perf.uploads import CSV_COLUMNS, VllmRun, load_upload, upload_backends
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.simulate.compare import compare_single_class
from llmplan.ui import presets
from llmplan.workload import compute_stats, load_workload
from tests.acceptance.test_m6 import APP, PLAN_TIMEOUT_S, texts
from tests.acceptance.test_m6 import plan as plan_ui
from tests.acceptance.test_m7 import two_class_request, two_class_workload
from tests.conftest import UseFakePerf
from tests.fake_planner import SHAPE_GPUS, FakePerf, request
from tests.unit.perf.stats import FakeStats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ROOFLINE_SENTENCE = "roofline (uncalibrated first-principles model; expect ±30% on throughput)"


def confidence_line(text: str) -> str:
    (line,) = [line for line in text.splitlines() if line.startswith("Confidence")]
    return line


# 9.1 Confidence everywhere.
def test_9_1_fake_measured_plan_reports_measured(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-a", 1): FakePerf(rps=3), ("fake-b", 1): FakePerf(rps=4)})
    req = request(34)
    result = plan(req)
    assert result.perf_confidence == "measured"
    assert result.perf_sources == ()
    line = confidence_line(render.get("text").plan(req, result))
    assert line.startswith("Confidence  measured (")


def test_9_1_real_catalog_plan_is_roofline() -> None:
    req = PlanRequest(  # M4 test 10.9
        model=load_model("fixture:llama3-8b"),
        stats=compute_stats(load_workload(FIXTURES / "workload_10.csv")),
        slo=SLO(),
        engine=EngineProfile(),
        options=PlanOptions(max_model_len=8192, perf_backend="roofline"),
        gpus=load_gpus(),
        prices=load_prices(),
    )
    result = plan(req)
    assert result.perf_confidence == "roofline"
    line = confidence_line(render.get("text").plan(req, result))
    assert line == f"Confidence  {ROOFLINE_SENTENCE}"


def test_9_1_cli_plan_text_has_the_confidence_line() -> None:
    out = CliRunner().invoke(
        app,
        [
            "plan",
            "--model",
            "fixture:llama3-8b",
            "--trace",
            str(FIXTURES / "workload_10.csv"),
            "--max-model-len",
            "8192",
            "--perf-backend",
            "roofline",
        ],
    )
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Cost"))
    assert lines[index + 1] == f"Confidence  {ROOFLINE_SENTENCE}"  # directly under Cost


def test_9_1_two_confidences_are_mixed(fake_perf: UseFakePerf) -> None:
    fake_perf(
        {
            ("fake-a", 1): FakePerf(rps=3, confidence="measured"),
            ("fake-b", 1): FakePerf(rps=4, confidence="roofline"),
        }
    )
    req = request(34)
    result = plan(req)  # 1 x B (8 GPUs) + 1 x A, as in M4 test 10.1
    assert {r.candidate.perf.confidence for r in result.replicas if r.candidate.perf} == {
        "measured",
        "roofline",
    }
    assert result.perf_confidence == "mixed"
    assert confidence_line(render.get("text").plan(req, result)).startswith("Confidence  mixed (")


# 9.2 Wording: no negative percentage; the two-cost sentence instead.
NEGATIVE_PCT = re.compile(r"-\d[\d,.]*%")
AZURE_SAMPLE = presets.SAMPLES_DIR / "azure2024_conv.csv"  # the M7 Azure sample (M7_NOTES.md 6.3)


def test_9_2_cli_wording_on_the_azure_sample() -> None:
    slo = ["--ttft-p95-ms", str(presets.DEFAULT_SLO.ttft_ms_p95)]
    slo += ["--tpot-p95-ms", str(presets.DEFAULT_SLO.tpot_ms_p95)]
    out = CliRunner().invoke(
        app,
        [
            "plan",
            "--model",
            "fixture:llama3-8b",
            "--trace",
            str(AZURE_SAMPLE),
            "--max-model-len",
            "8192",
            *slo,
            "--classes",
            "2x2",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "Sized for the mean request" in out.output
    assert "-87" not in out.output
    assert not NEGATIVE_PCT.search(out.output)


@pytest.mark.ui
def test_9_2_ui_wording_on_the_azure_sample() -> None:
    at = AppTest.from_file(str(APP), default_timeout=PLAN_TIMEOUT_S).run()
    assert at.selectbox(key="preset").value == "azure2024-conv"  # UI defaults: 2x2, default SLO
    assert at.selectbox(key="classes").value == "2x2"
    plan_ui(at)
    assert not at.error
    shown = [*texts(at), *(f"{m.label} {m.value}" for m in at.metric)]
    assert any("Sized for the mean request" in text for text in shown)
    assert not any("-87" in text for text in shown)
    assert not any(NEGATIVE_PCT.search(text) for text in shown)


def test_9_2_saving_scenario_says_saves() -> None:
    trace = two_class_workload()  # M7 test 8.2: $96/day with classes, $144/day without
    req = two_class_request(trace)
    result = plan(req)
    comparison = compare_single_class(req, result, trace, gpus=SHAPE_GPUS)
    text = render.get("text").plan(req, result, comparison)
    assert "saves $" in text
    assert "Request-size routing saves $48.00/day (33.3%)" in text
    assert not NEGATIVE_PCT.search(text)


# 9.3 Benchmark CSV import.
M8_FIXTURES = FIXTURES / "m8"
LLAMA8B = load_model("fixture:llama3-8b")
H100 = load_gpus()["h100-sxm-80gb"]


def test_9_3_benchmark_csv_import() -> None:
    upload = load_upload((M8_FIXTURES / "benchmarks_3_rows.csv").read_bytes(), LLAMA8B, load_gpus())
    assert len(upload.rows) == 2
    (rejected,) = upload.rejected
    assert rejected.index == 1  # the 50,000 tokens/s row
    assert "physical floor" in rejected.reason
    assert all(row.source_url == "user-upload" for row in upload.rows)
    stats = FakeStats(
        input_tokens_mean=1000,
        input_tokens_p50=1000,
        input_tokens_p95=1000,
        output_tokens_mean=200,
        output_tokens_p50=200,
        output_tokens_p95=200,
    )
    config = ReplicaConfig(dtype="fp8", max_num_seqs=64, max_model_len=8192)
    result = estimate(LLAMA8B, H100, config, stats, backends=upload_backends(upload.rows))
    assert result.confidence == "measured"  # effective batch 64 hits the concurrency-64 row
    assert result.source_urls == ("user-upload",)
    assert result.decode_tokens_per_s == 4200.0


# 9.4 vLLM benchmark JSON import.
def test_9_4_vllm_json_import() -> None:
    data = (M8_FIXTURES / "vllm_bench_serve.json").read_bytes()
    doc = json.loads(data)
    run = VllmRun(gpu_id="h100-sxm-80gb", tensor_parallel=1, dtype="bf16", engine_version="0.30.0")
    upload = load_upload(data, LLAMA8B, load_gpus(), run=run)
    assert upload.rejected == ()
    (row,) = upload.rows
    assert row.input_len == round(doc["total_input_tokens"] / doc["completed"]) == 1023
    assert row.output_len == round(doc["total_output_tokens"] / doc["completed"]) == 128
    assert row.output_tokens_per_s == doc["output_throughput"]
    assert row.ttft_ms_p50 == doc["median_ttft_ms"]
    assert row.tpot_ms_p50 == doc["median_tpot_ms"]
    assert row.concurrency == doc["max_concurrency"]
    assert row.ttft_ms_p95 is None  # p99_* ignored; p95_* absent at vLLM's default percentiles
    assert (row.engine, row.gpu_id, row.source_url) == ("vllm", "h100-sxm-80gb", "user-upload")


# 9.5 Contribute link.
def _rows(n: int) -> list[BenchmarkRow]:
    base = load_upload(
        (M8_FIXTURES / "benchmarks_3_rows.csv").read_bytes(), LLAMA8B, load_gpus()
    ).rows[0]
    return [base.model_copy(update={"concurrency": 1 + i}) for i in range(n)]


def test_9_5_contribute_link() -> None:
    url, with_rows = contribute_url(_rows(20))
    assert url.startswith(f"{REPOSITORY_URL}/issues/new?")
    assert len(url) < MAX_URL_CHARS == 6_000
    assert with_rows
    (body,) = parse_qs(urlsplit(url).query)["body"]
    assert ",".join(CSV_COLUMNS) in body.splitlines()  # the CSV header line
    url_500, with_rows_500 = contribute_url(_rows(500))
    assert not with_rows_500
    assert url_500.startswith(f"{REPOSITORY_URL}/issues/new?")
    assert len(url_500) < MAX_URL_CHARS
    (body_500,) = parse_qs(urlsplit(url_500).query)["body"]
    assert ",".join(CSV_COLUMNS) not in body_500


# 9.6 MoE parameter counts (M8_DESIGN.md section 6 table; official 46.7B/12.9B, 30.5B/3.3B).
@pytest.mark.parametrize(
    ("fixture", "params", "active", "kv_bytes"),
    [
        ("mixtral-8x7b", 46_702_792_704, 12_879_925_248, 131_072),
        ("qwen3-30b-a3b", 30_532_122_624, 3_353_032_704, 98_304),
    ],
)
def test_9_6_moe_parameter_counts(fixture: str, params: int, active: int, kv_bytes: int) -> None:
    spec = load_model(f"fixture:{fixture}")
    info = model_info(spec)
    assert info.param_count == params
    assert info.active_param_count == active
    assert info.attention == spec.attention == "gqa"
    assert kv_bytes_per_token_total(spec, "bf16") == kv_bytes


# 9.7 MoE fit and roofline.
QWEN3_MOE = load_model("fixture:qwen3-30b-a3b")


def test_9_7_moe_fit() -> None:
    h100 = fit(
        FitRequest(model=QWEN3_MOE, gpu=H100, tensor_parallel=1, dtype="bf16", context_len=8192)
    )
    assert h100.fits
    assert h100.per_gpu_weight_bytes == 61_064_245_248  # all 128 experts resident
    l4 = fit(
        FitRequest(
            model=QWEN3_MOE,
            gpu=load_gpus()["l4-24gb"],
            tensor_parallel=1,
            dtype="bf16",
            context_len=8192,
        )
    )
    assert not l4.fits
    assert l4.binding == "weights"


@pytest.mark.parametrize(("batch", "share"), [(1, 8 / 128), (64, 1.0)])
def test_9_7_moe_roofline_decode_bytes(batch: int, share: float) -> None:
    expert = expert_weight_bytes(QWEN3_MOE, "bf16")
    non_expert = weight_bytes(QWEN3_MOE, "bf16") - expert
    assert non_expert + expert == 61_064_245_248
    expected = non_expert + expert * min(1, batch * 8 / 128)
    assert share == min(1, batch * 8 / 128)
    read = decode_weight_bytes(QWEN3_MOE, "bf16", 1, batch, 61_064_245_248)
    assert read == pytest.approx(expected, rel=1e-9)
    # ...and the roofline estimate's decode step reads exactly that (memory-bound here).
    stats = FakeStats(
        input_tokens_mean=1000,
        input_tokens_p50=1000,
        input_tokens_p95=1000,
        output_tokens_mean=200,
        output_tokens_p50=200,
        output_tokens_p95=200,
    )
    config = ReplicaConfig(max_num_seqs=batch, max_model_len=8192)
    result = estimate(QWEN3_MOE, H100, config, stats, backend="roofline")
    assert result.effective_batch == batch
    assert H100.memory_bandwidth_gbps is not None
    kv = batch * 1100 * 98_304  # batch x (input + output / 2) tokens x KV bytes per token
    step_s = (expected + kv) / (H100.memory_bandwidth_gbps * 1e9 * BANDWIDTH_EFFICIENCY)
    assert result.tpot_ms_p50 == pytest.approx(step_s * 1e3, rel=1e-9)


# 9.8 Unsupported MLA.
def test_9_8_unsupported_mla() -> None:
    with pytest.raises(UnsupportedArchitecture, match="MLA") as info:
        load_model("fixture:deepseek-v3")  # architectures: ["DeepseekV3ForCausalLM"]
    assert info.value.field == "architectures"


# 9.9 Packaging: the wheel carries the data and runs from outside the checkout.
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_wheel_smoke(tmp_path: Path) -> None:
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    build = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / "dist"), str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    (wheel,) = (tmp_path / "dist").glob("llmplan-*.whl")
    spec = importlib.util.spec_from_file_location(
        "wheel_smoke", ROOT / "scripts" / "wheel_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / "work"  # pytest's tmp_path is outside the checkout
    assert not work.resolve().is_relative_to(ROOT)
    fit_json, gpus_text = module.smoke(wheel, work, installer="uv")
    assert json.loads(fit_json)["fits"] is True
    assert "h100-sxm-80gb" in gpus_text
