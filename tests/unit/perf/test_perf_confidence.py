from __future__ import annotations

from typer.testing import CliRunner

from llmplan.cli import app
from llmplan.perf.confidence import combined_confidence, confidence_sentence, source_label

NIM = "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html"


def test_combined_confidence() -> None:
    assert combined_confidence(["measured", "measured"]) == "measured"
    assert combined_confidence(["interpolated"]) == "interpolated"
    assert combined_confidence(["measured", "roofline"]) == "mixed"
    assert combined_confidence([]) == "roofline"


def test_sentences_name_the_source_and_its_date() -> None:
    assert source_label(NIM) == "NVIDIA NIM performance page, as of 2026-09-30"
    assert source_label("https://example.com/x") == "https://example.com/x"
    assert confidence_sentence("measured", (NIM,)) == (
        "measured (NVIDIA NIM performance page, as of 2026-09-30)"
    )
    assert confidence_sentence("interpolated", (NIM, NIM)) == (
        "interpolated between measured rows (NVIDIA NIM performance page, as of 2026-09-30)"
    )
    assert confidence_sentence("measured", ()) == "measured (no source recorded)"
    mixed = confidence_sentence("mixed", (NIM,))
    assert mixed.startswith("mixed (some replicas use measured rows")
    assert "expect ±30% on throughput; measured rows: NVIDIA NIM" in mixed
    assert "measured rows:" not in confidence_sentence("mixed", ())


def test_perf_estimate_text_states_the_confidence() -> None:
    base = ["perf", "estimate", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb"]
    stats = ["--in-mean", "200", "--in-p50", "200", "--in-p95", "200"]
    stats += ["--out-mean", "200", "--out-p50", "200", "--out-p95", "200"]
    roofline = CliRunner().invoke(app, [*base, *stats, "--backend", "roofline"])
    assert roofline.exit_code == 0, roofline.output
    assert "Confidence  roofline (uncalibrated first-principles model" in roofline.output
    table = CliRunner().invoke(
        app, [*base, *stats, "--dtype", "fp8", "--max-num-seqs", "100", "--backend", "table"]
    )
    assert table.exit_code == 0, table.output
    assert "Confidence  measured (NVIDIA NIM performance page, as of 2026-09-30)" in table.output
