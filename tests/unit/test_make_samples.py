"""scripts/make_samples.py: hour selection, proportional thinning, and the end-to-end run on
the 50-row format fixtures (the real traces are never needed in tests)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from llmplan.workload import load_workload
from llmplan.workload.fetch import load_manifest
from tests.fake_planner import workload

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


make_samples = _script("make_samples")


def _hours(counts: list[int], tail: int = 0) -> list[float]:
    """Arrivals spread inside consecutive hours with the given counts, plus a partial hour."""
    arrivals: list[float] = []
    for hour, n in enumerate([*counts, tail]):
        arrivals += [hour * 3600.0 + 3000.0 * i / n for i in range(n)]
    return arrivals


def test_busiest_and_median_hours_are_kept_back_to_back_and_thinned_alike() -> None:
    trace = workload(_hours([10, 30, 20, 5], tail=50), 100, 10)  # hour 4 is partial
    sample, info = make_samples.cut(trace, max_rows=20)
    assert info.hours == (1, 0)  # busiest 30, then the lower median of (5, 10, 20, 30)
    assert info.hour_rows == (30, 10)
    assert info.kept_rows == (15, 5)  # floor(n * 20 / 40) each
    assert 4 * 3600 <= info.trace_duration_s < 5 * 3600
    arrival = sample.frame["arrival_s"].to_numpy()
    assert len(arrival) == 20
    assert arrival[0] == 0.0
    assert np.all(np.diff(arrival) >= 0)
    assert arrival[5] - arrival[4] > 600  # the median hour (5 rows) comes first, then a gap
    again, _ = make_samples.cut(trace, max_rows=20)
    assert again.frame.equals(sample.frame)  # seeded


def test_a_single_full_hour_and_an_untouched_small_trace() -> None:
    sample, info = make_samples.cut(workload(_hours([7], tail=3), 100, 10))
    assert info.hours == (0,)
    assert info.kept_rows == (7,)
    assert len(sample.frame) == 7


def test_end_to_end_on_the_format_fixtures(tmp_path: Path) -> None:
    traces, out = tmp_path / "traces", tmp_path / "samples"
    traces.mkdir()
    out.mkdir()
    fixtures = {
        "azure2023-code": "workload_azure2023_50.csv",
        "azure2023-conv": "workload_azure2023_50.csv",
        "azure2024-code": "workload_azure2024_50.csv",
        "azure2024-conv": "workload_azure2024_50.csv",
        "burstgpt-1": "workload_burstgpt_50.csv",
    }
    for name, source in load_manifest().items():
        assert source.url is not None
        shutil.copy(FIXTURES / fixtures[name], traces / source.url.rsplit("/", 1)[-1])
    make_samples.main([str(traces), "--out", str(out)])
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Bundled trace samples")
    for filename, label, _ in make_samples.SAMPLES.values():
        sample = load_workload(out / filename)
        assert sample.format == "csv"
        assert 0 < len(sample.frame) <= make_samples.MAX_ROWS
        assert f"## `{filename}`: {label}" in readme
    assert "CC-BY-4.0" in readme
