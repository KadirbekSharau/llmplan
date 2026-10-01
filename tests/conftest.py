from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import pytest

from tests.fake_planner import FAKE_CLASS_PERF, FAKE_PERF, ClassCapacity, FakePerf

UseFakePerf = Callable[[Mapping[tuple[str, int], FakePerf]], None]
UseFakeClassPerf = Callable[[Mapping[tuple[str, int], tuple[ClassCapacity, ...]]], None]


@pytest.fixture
def fake_perf(monkeypatch: pytest.MonkeyPatch) -> UseFakePerf:
    """Set the fake perf backend's capacities for one test; restored afterwards."""

    def use(capacities: Mapping[tuple[str, int], FakePerf]) -> None:
        for key, value in capacities.items():
            monkeypatch.setitem(FAKE_PERF, key, value)

    return use


@pytest.fixture
def fake_class_perf() -> Iterator[UseFakeClassPerf]:
    """Replace the class-aware fake backend's capacities (M7); emptied afterwards."""

    def use(capacities: Mapping[tuple[str, int], tuple[ClassCapacity, ...]]) -> None:
        FAKE_CLASS_PERF.clear()
        FAKE_CLASS_PERF.update(capacities)

    yield use
    FAKE_CLASS_PERF.clear()
