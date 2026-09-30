from __future__ import annotations

import pytest

from llmplan import render
from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.errors import UnknownRegistryKey
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, fit


def test_unknown_format() -> None:
    with pytest.raises(UnknownRegistryKey, match="yaml"):
        render.get("yaml")


def test_text_fit_non_default_overhead() -> None:
    request = FitRequest(
        model=load_model("fixture:llama3-8b"),
        gpu=load_gpus()["l40s-48gb"],
        engine=EngineProfile(fixed_overhead_bytes=500_000_000),
        dtype="int4",
        context_len=4096,
    )
    out = render.get("text").fit(request, fit(request))
    assert "(0.50 GB fixed + " in out
    assert "int4   5.59 GB total" in out
    assert "- int4/int8: scales/zeros not counted" in out


def test_text_model_info_marks_override() -> None:
    spec = load_model("fixture:llama3-8b").model_copy(update={"param_count_override": 10**9})
    assert "[override]" in render.get("text").model_info(spec)
