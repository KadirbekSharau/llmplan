"""Words and numbers of the web UI (M9): formats and error hints."""

from __future__ import annotations

from llmplan.errors import (
    FetchError,
    InfeasiblePlan,
    SolverError,
    ValidationError,
    WorkloadFormatError,
)
from llmplan.ui import state


def test_currency_and_durations() -> None:
    assert state.usd(1234.5) == "$1,234.50"
    assert state.usd(None) == "n/a"
    assert state.duration(12.345) == "12.3 ms"
    assert state.duration(523.4) == "523 ms"
    assert state.duration(1530.0) == "1.53 s"


def test_error_hints_follow_the_exception_type() -> None:
    """M9 section 4: the hint comes from the type, never from the message."""
    infeasible = state.error_text(InfeasiblePlan("0 of 3 candidates meet the target", reason="slo"))
    assert infeasible.startswith("0 of 3 candidates meet the target\n\nWhat to change: relax")
    assert "HF_TOKEN" in state.error_text(FetchError("anything"))
    assert "timestamp, input_tokens, output_tokens" in state.error_text(WorkloadFormatError("x"))
    assert "time limit" in state.error_text(SolverError("x"))
    assert state.error_text(ValidationError("relax: not a hint")) == "relax: not a hint"
