from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmplan import render
from llmplan.errors import ValidationError
from llmplan.render.plots import render_png, save_png, vram_split_gb
from llmplan.simulate import SimOptions, Timeline, replay
from tests.fake_planner import GPUS, sim_plan, workload


def _timeline(replicas: int = 2, **options: object) -> Timeline:
    trace = workload([0.25 * i for i in range(40)], 100, 40)
    return replay(sim_plan(replicas=replicas), trace, options=SimOptions.model_validate(options))


def test_json_is_the_timeline_and_deterministic() -> None:
    timeline = _timeline(ttft_budget_ms=150.0)
    first = render.get("json").timeline(timeline)
    assert first == render.get("json").timeline(_timeline(ttft_budget_ms=150.0))
    assert first.endswith("\n")
    assert Timeline.model_validate(json.loads(first)) == timeline


def test_text_has_summary_windows_and_assumptions() -> None:
    text = render.get("text").timeline(_timeline(window_s=5.0, ttft_budget_ms=150.0))
    assert "40 simulated (0 truncated)" in text
    assert "budget 150 ms: 0.00% violations" in text
    assert "TPOT      p95 10.00 ms  (no budget)" in text
    assert "2 replicas" in text
    assert "demand_rps" in text
    assert "Assumptions" in text
    rows = [line for line in text.splitlines() if line.strip().startswith(("0 ", "1 ", "2 "))]
    assert len(rows) == 3  # 10 s of arrivals plus the last completion, in 5 s windows


def test_vram_split_with_and_without_the_gpu_spec() -> None:
    trace = workload([0.25 * i for i in range(40)], 100, 40)
    known = replay(sim_plan(replicas=2), trace, gpus=GPUS)
    window = known.windows[0]
    weights, kv, free = vram_split_gb(window)
    replica = window.replicas[0]
    assert replica.vram_bytes_total == GPUS["fake-a"].vram_bytes
    assert weights == 2 * replica.weight_bytes / 1e9
    assert kv == sum(r.kv_bytes_in_use_max for r in window.replicas) / 1e9
    assert free == pytest.approx(2 * GPUS["fake-a"].vram_bytes / 1e9 - weights - kv)
    unknown = replay(sim_plan(replicas=2), trace)
    assert vram_split_gb(unknown.windows[0])[2] is None
    assert any("total VRAM unknown for fake-a" in note for note in unknown.assumptions)
    assert not any("VRAM unknown" in note for note in known.assumptions)
    for timeline in (known, unknown):
        assert render_png(timeline)[:8] == b"\x89PNG\r\n\x1a\n"
    assert render_png(known) == render_png(known)  # deterministic bytes


def test_png_many_replicas_and_unwritable_path(tmp_path: Path) -> None:
    path = tmp_path / "many.png"
    save_png(_timeline(replicas=9, window_s=2.0), path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValidationError, match="cannot write PNG"):
        save_png(_timeline(), tmp_path / "missing" / "x.png")
