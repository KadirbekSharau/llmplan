"""scripts/melange_crosscheck.py: llmplan's side of the Mélange cross-check (M7 section 6.2).

Mélange itself is not a dependency; its solver is replaced by a stub package here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.unit.test_make_samples import _script

crosscheck = _script("melange_crosscheck")


def test_toy_example_in_the_continuous_limit() -> None:
    # 1 A10G + 1 A100: the A100 carries 1.125 load units if alone; moving 5 of the 15 req/s
    # of bucket (1, 0) to the A10G (load 0.2 each) frees 0.125 and fills the A10G exactly.
    cost, fleet = crosscheck.llmplan_solve(crosscheck.TOY)
    assert cost == pytest.approx(4.68)
    assert fleet == {"A10G": 1, "A100-80GB": 1}


def test_synthetic_scenarios_share_one_capacity_table() -> None:
    scenarios = crosscheck.synthetic(20.0)
    assert list(scenarios) == list(crosscheck.MIXES)
    gpu_info = scenarios["mixed"]["gpu_info"]
    assert list(gpu_info) == ["l4-24gb (aws)", "l40s-48gb (aws)", "h100-sxm-80gb (runpod)"]
    l4 = gpu_info["l4-24gb (aws)"]["tputs"]
    assert l4[0][0] > 0  # chat meets the SLO on an L4
    assert l4[1][0] == 0.0  # 5,000-token documents miss the 500 ms TTFT
    cost, fleet = crosscheck.llmplan_solve(scenarios["short-chat heavy"])
    assert cost > 0
    assert "l4-24gb (aws)" in fleet


def test_main_and_the_melange_hook(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "melange"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "solver.py").write_text(
        "class MelangeSolver:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
        "    def run(self):\n"
        "        return {'slice_factor': self.kwargs['slice_factor']}\n"
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "melange", raising=False)
    monkeypatch.delitem(sys.modules, "melange.solver", raising=False)
    crosscheck.main(["--melange-dir", str(tmp_path), "--rate", "10"])
    out = capsys.readouterr().out
    assert "## toy (example.json)" in out
    assert "llmplan: $4.6800/h" in out
    assert "melange slice 16: {'slice_factor': 16}" in out
    assert out.count("## ") == 4
