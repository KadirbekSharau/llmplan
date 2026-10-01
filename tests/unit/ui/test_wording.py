"""Words and numbers of the web UI (M9): formats, the model chip and error hints."""

from __future__ import annotations

from llmplan.catalog.models import load_model
from llmplan.errors import (
    FetchError,
    InfeasiblePlan,
    SolverError,
    ValidationError,
    WorkloadFormatError,
)
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


def test_error_hints_follow_the_exception_type() -> None:
    """M9 section 4: the hint comes from the type, never from the message."""
    infeasible = wording.error_text(
        InfeasiblePlan("0 of 3 candidates meet the target", reason="slo")
    )
    assert infeasible.startswith("0 of 3 candidates meet the target\n\nWhat to change: relax")
    assert "HF_TOKEN" in wording.error_text(FetchError("anything"))
    assert "input_tokens and output_tokens" in wording.error_text(WorkloadFormatError("x"))
    assert "time limit" in wording.error_text(SolverError("x"))
    assert wording.error_text(ValidationError("relax: not a hint")) == "relax: not a hint"
