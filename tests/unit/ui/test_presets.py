from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llmplan.ui import presets
from llmplan.workload import compute_stats

PACKAGE = Path(__file__).resolve().parents[3] / "llmplan"


def test_model_choices() -> None:
    assert presets.DEFAULT_MODEL in presets.FIXTURE_MODELS
    assert "fixture:gpt2" not in presets.FIXTURE_MODELS
    assert all("/" in model_id for model_id in presets.POPULAR_MODELS)


def test_preset_lookup_and_synthetic_preset_is_deterministic() -> None:
    assert presets.preset(presets.SYNTHETIC_PRESET.key) is presets.SYNTHETIC_PRESET
    with pytest.raises(KeyError, match="unknown preset 'nope'"):
        presets.preset("nope")
    first = presets.load_preset(presets.SYNTHETIC_PRESET)
    assert first.frame.equals(presets.load_preset(presets.SYNTHETIC_PRESET).frame)
    assert 6_000 < compute_stats(first).n_requests < 8_500  # 2 req/s for an hour


@pytest.mark.parametrize("sample", presets.SAMPLE_PRESETS, ids=lambda p: p.key)
def test_bundled_samples_parse_with_their_recorded_row_counts(
    sample: presets.SamplePreset,
) -> None:
    workload = presets.load_preset(sample)
    assert len(workload.frame) == sample.rows <= 20_000
    assert workload.dropped_rows == 0


def test_limits_match_the_design() -> None:
    assert presets.MAX_UPLOAD_BYTES == 50_000_000
    assert presets.MAX_TIME_LIMIT_S == 30.0
    assert presets.MAX_SIM_REQUESTS == 200_000
    assert presets.MAX_PLANS_PER_HOUR == 30
    assert presets.DEFAULT_SLO.ttft_ms_p95 == 500.0
    assert presets.DEFAULT_SLO.tpot_ms_p95 == 50.0
    assert presets.DEFAULT_SLO.utilization_target == 0.8


def _imports(path: Path) -> list[tuple[str, bool]]:
    """(module, imported at module level) for every import in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, id(node) in top_level) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, id(node) in top_level))
    return found


def test_only_the_ui_imports_streamlit() -> None:
    """ARCHITECTURE.md section 1: nothing in the core imports Streamlit. The `llmplan ui`
    command imports it inside the command function only."""
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in ("ui/app.py", "ui/views.py", "ui/calibrate.py"):  # M8: calibrate
            continue
        for module, top_level in _imports(path):
            if module.split(".")[0] == "streamlit":
                assert relative == "cli_ui.py" and not top_level, f"{relative} imports {module}"
