from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_secrets", ROOT / "scripts" / "check_secrets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_each_marker_is_detected_and_masked() -> None:
    scan = _module().scan_patch
    # Built at run time so that no secret-shaped string is ever committed.
    fakes = {
        "hf_": "hf" + "_" + "a1" * 17,
        "ghp_": "ghp" + "_" + "B2" * 18,
        "AKIA": "AKIA" + "Z3" * 8,
        "sk-": "sk" + "-" + "c4" * 12,
        "token": "token = '" + "d5" * 10 + "'",
        "bearer": "Authorization: Bearer " + "e6" * 12,
    }
    lines = ["commit 0123456789abcdef0123", "+++ b/config.py"]
    lines += [f"+x = {value}" for value in fakes.values()]
    lines += [" context " + fakes["hf_"], "-y = 1  # input_tokens, HF_TOKEN, sk-learn"]
    hits = list(scan(lines))
    assert sorted(h.marker for h in hits) == sorted(fakes)
    assert {h.commit for h in hits} == {"0123456789ab"}
    assert {h.path for h in hits} == {"config.py"}
    assert all(h.masked.endswith("...") and len(h.masked) <= 9 for h in hits)


def test_main_reports_and_exits(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()  # the real scan runs in tests/acceptance/test_m8.py (9.10)
    monkeypatch.setattr(module, "scan_history", lambda repo, revisions: [])
    assert module.main([]) == 0
    assert "secrets scan: 0 hits in the history of HEAD" in capsys.readouterr().out
    hit = module.Hit("0123456789ab", "a.py", "hf_", "hf_abc...")
    monkeypatch.setattr(module, "scan_history", lambda repo, revisions: [hit])
    assert module.main(["main"]) == 1
    assert "0123456789ab a.py: hf_ hf_abc..." in capsys.readouterr().out
