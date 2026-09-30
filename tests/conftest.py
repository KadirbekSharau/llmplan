from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from tests.fake_planner import FAKE_PERF, FakePerf

UseFakePerf = Callable[[Mapping[tuple[str, int], FakePerf]], None]


@pytest.fixture
def fake_perf(monkeypatch: pytest.MonkeyPatch) -> UseFakePerf:
    """Set the fake perf backend's capacities for one test; restored afterwards."""

    def use(capacities: Mapping[tuple[str, int], FakePerf]) -> None:
        for key, value in capacities.items():
            monkeypatch.setitem(FAKE_PERF, key, value)

    return use
