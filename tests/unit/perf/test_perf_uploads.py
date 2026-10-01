from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.cli import app
from llmplan.errors import BenchmarkError, ValidationError
from llmplan.perf import ReplicaConfig, estimate
from llmplan.perf.benchmarks import load_benchmarks
from llmplan.perf.confidence import confidence_sentence
from llmplan.perf.contribute import contribute_url
from llmplan.perf.uploads import (
    MAX_UPLOAD_ROWS,
    VllmRun,
    load_upload,
    rows_csv,
    upload_backends,
)
from tests.unit.perf.stats import FakeStats

M8 = Path(__file__).resolve().parents[2] / "fixtures" / "m8"
CSV = (M8 / "benchmarks_3_rows.csv").read_bytes()
VLLM = json.loads((M8 / "vllm_bench_serve.json").read_text(encoding="utf-8"))
MODEL = load_model("fixture:llama3-8b")
GPUS = load_gpus()
RUN = VllmRun(gpu_id="h100-sxm-80gb")
HEADER, FIRST = CSV.decode().splitlines()[:2]


def csv_with(*lines: str) -> bytes:
    return "\n".join([HEADER, *lines]).encode()


def vllm(**changes: object) -> bytes:
    return json.dumps({**VLLM, **changes}).encode()


def test_file_level_errors() -> None:
    with pytest.raises(ValidationError, match="limit is 5,000,000"):
        load_upload(b"x" * 5_000_001, MODEL, GPUS)
    with pytest.raises(ValidationError, match="not UTF-8"):
        load_upload(b"\xff\xfe\x00", MODEL, GPUS)
    with pytest.raises(ValidationError, match="missing columns: model_id, gpu_id"):
        load_upload(b"a,b\n1,2\n", MODEL, GPUS)
    with pytest.raises(ValidationError, match="unknown columns: extra"):
        load_upload(f"{HEADER},extra\n".encode(), MODEL, GPUS)
    with pytest.raises(ValidationError, match="no rows"):
        load_upload(HEADER.encode(), MODEL, GPUS)
    too_many = csv_with(*[FIRST] * (MAX_UPLOAD_ROWS + 1))
    with pytest.raises(ValidationError, match="501 rows; the limit is 500"):
        load_upload(too_many, MODEL, GPUS)
    with pytest.raises(ValidationError, match="not valid JSON"):
        load_upload(b"{not json\n{", MODEL, GPUS, run=RUN)
    with pytest.raises(ValidationError, match="does not record the GPU"):
        load_upload(vllm(), MODEL, GPUS)


def test_optional_latency_columns_may_be_left_out() -> None:
    columns = HEADER.split(",")[:10]
    text = ",".join(columns) + "\n" + ",".join(FIRST.split(",")[:10]) + "\n"
    (row,) = load_upload(text.encode(), MODEL, GPUS, today=date(2026, 10, 1)).rows
    assert row.ttft_ms_p50 is None
    assert row.as_of == date(2026, 10, 1)


def test_rows_are_checked_like_shipped_rows() -> None:
    a10g = FIRST.replace("h100-sxm-80gb", "a10g-24gb").replace(",fp8,", ",bf16,")
    upload = load_upload(
        csv_with(
            FIRST.replace("h100-sxm-80gb", "no-such-gpu"),
            FIRST.replace("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
            FIRST.replace(",1,fp8,", ",3,fp8,"),
            FIRST.replace(",fp8,", ",fp99,"),
            a10g,
            FIRST.replace("meta-llama/Llama-3.1-8B-Instruct", "fixture:llama3-8b"),
        ),
        MODEL,
        GPUS,
    )
    reasons = [r.reason for r in upload.rejected]
    assert [r.index for r in upload.rejected] == [0, 1, 2, 3, 4]
    assert reasons[0] == "unknown gpu_id 'no-such-gpu'"
    assert "is not the model being planned (fixture:llama3-8b)" in reasons[1]
    assert "does not divide num_attention_heads 32" in reasons[2]
    assert reasons[3].startswith("field 'dtype'")
    assert "physical bound cannot be checked" in reasons[4]
    assert len(upload.rows) == 1  # the fixture id resolves through the shipped aliases


def test_vllm_json_variants() -> None:
    lines = "\n".join(
        [
            json.dumps({**VLLM, "max_concurrency": None, "p95_ttft_ms": 150.0}),
            json.dumps(["not", "an", "object"]),
            json.dumps({**VLLM, "completed": 0}),
            json.dumps({**VLLM, "total_input_tokens": "lots"}),
        ]
    )
    upload = load_upload(lines.encode(), MODEL, GPUS, run=RUN)  # --append-result layout
    (row,) = upload.rows
    assert row.concurrency == VLLM["num_prompts"]
    assert row.ttft_ms_p95 == 150.0
    assert "max_concurrency is null" in upload.notes[0]
    assert [(r.index, r.reason) for r in upload.rejected][:2] == [
        (1, "not a JSON object"),
        (2, "'completed' must be a positive integer, got 0"),
    ]
    assert upload.rejected[2].reason.startswith("field 'input_len'")


def test_uploaded_rows_answer_first_and_round_trip() -> None:
    rows = load_upload(CSV, MODEL, GPUS).rows
    assert upload_backends(()) == {}
    backends = upload_backends(rows)
    config = ReplicaConfig(dtype="fp8", max_num_seqs=16, max_model_len=8192)
    h100 = GPUS["h100-sxm-80gb"]

    def stats(input_tokens: float) -> FakeStats:
        return FakeStats(
            input_tokens_mean=input_tokens,
            input_tokens_p50=input_tokens,
            input_tokens_p95=input_tokens,
            output_tokens_mean=200,
            output_tokens_p50=200,
            output_tokens_p95=200,
        )

    mine = estimate(MODEL, h100, config, stats(1000), backends=backends)
    assert mine.source_urls == ("user-upload",)  # NIM rows match too; the upload answers
    assert confidence_sentence(mine.confidence, mine.source_urls) == "measured (your upload)"
    shipped = estimate(MODEL, h100, config, stats(200), backend="table")
    fallback = estimate(MODEL, h100, config, stats(200), backends=backends)
    assert fallback == shipped  # uploaded shape 1000/200 is 5x off: the shipped rows answer
    tp2 = load_upload(csv_with(FIRST.replace(",1,fp8,", ",2,fp8,")), MODEL, GPUS).rows
    table = upload_backends(tp2)["table"]
    config2 = config.model_copy(update={"tensor_parallel": 2})
    assert table.estimate(MODEL, h100, config2, stats(200)) is None  # no shipped tp2 rows
    assert "more than 2x" in table.explain(MODEL, h100, config2, stats(200))
    again = load_upload(rows_csv(rows).encode(), MODEL, GPUS)
    assert again.rows == rows


def test_shipped_tables_may_not_claim_uploads(tmp_path: Path) -> None:
    row = load_upload(CSV, MODEL, GPUS).rows[0].model_dump(mode="json")
    (tmp_path / "h100-sxm-80gb.yaml").write_text(yaml.safe_dump([row]), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="reserved for uploads"):
        load_benchmarks(tmp_path)


def test_contribute_needs_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        contribute_url(())


def test_cli_benchmarks(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_bytes(CSV)
    stats = ["--in-mean", "1000", "--in-p50", "1000", "--in-p95", "1000"]
    stats += ["--out-mean", "200", "--out-p50", "200", "--out-p95", "200"]
    base = ["perf", "estimate", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb"]
    base += ["--dtype", "fp8", "--max-num-seqs", "64", *stats]
    out = CliRunner().invoke(app, [*base, "--benchmarks", str(csv_path)])
    assert out.exit_code == 0, out.output
    assert "Confidence  measured (your upload)" in out.stdout
    assert "benchmarks: 2 of 3 rows used" in out.stderr
    assert "benchmarks: row 1 rejected: output_tokens_per_s 50000" in out.stderr

    json_path = tmp_path / "run.json"
    json_path.write_bytes(vllm(max_concurrency=None))
    out = CliRunner().invoke(
        app,
        [*base, "--benchmarks", str(json_path), "--benchmarks-engine-version", "0.30.0"],
    )
    assert out.exit_code == 0, out.output
    assert "benchmarks: 1 of 1 rows used" in out.stderr  # the run defaults to --gpu, --dtype
    assert "max_concurrency is null" in out.stderr

    plan = ["plan", "--model", "fixture:llama3-8b", "--max-model-len", "8192", "--gpus"]
    plan += ["h100-sxm-80gb", "--dtypes", "fp8", "--tp", "1"]
    plan += ["--trace", str(Path(__file__).resolve().parents[2] / "fixtures" / "workload_10.csv")]
    for flags in (
        ["--benchmarks", str(json_path), "--benchmarks-gpu", "h100-sxm-80gb"],
        ["--benchmarks", str(json_path), "--benchmarks-dtype", "fp8", "--benchmarks-tp", "1"],
    ):
        out = CliRunner().invoke(app, [*plan, *flags])
        if "--benchmarks-gpu" in flags:
            assert out.exit_code == 0, out.output
            assert "benchmarks: 1 of 1 rows used" in out.stderr
        else:
            assert out.exit_code == 2
            assert "does not record the GPU" in out.stderr

    missing = CliRunner().invoke(app, [*base, "--benchmarks", str(tmp_path / "nope.csv")])
    assert missing.exit_code == 2
    assert "is not readable" in missing.stderr
    big = tmp_path / "big.csv"
    big.write_bytes(b"x" * 5_000_001)
    too_big = CliRunner().invoke(app, [*base, "--benchmarks", str(big)])
    assert too_big.exit_code == 2
    assert "larger than 5,000,000 bytes" in too_big.stderr
