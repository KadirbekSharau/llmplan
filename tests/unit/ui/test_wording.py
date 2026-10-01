"""Words and numbers of the web UI (M9): formats and the model summary chip."""

from __future__ import annotations

from llmplan.catalog.models import load_model
from llmplan.ui import wording


def test_currency_and_durations() -> None:
    assert wording.usd(1234.5) == "$1,234.50"
    assert wording.usd(None) == "n/a"
    assert wording.duration(12.345) == "12.3 ms"
    assert wording.duration(523.4) == "523 ms"
    assert wording.duration(1530.0) == "1.53 s"


def test_model_summary_dense_and_mixture_of_experts() -> None:
    assert wording.model_summary(load_model("fixture:llama3-8b")) == (
        "8.03B parameters · GQA · 128 KiB KV cache per token (bf16)"
    )
    assert wording.model_summary(load_model("fixture:qwen3-30b-a3b")) == (
        "30.53B parameters (3.35B active) · GQA · 96 KiB KV cache per token (bf16)"
    )
