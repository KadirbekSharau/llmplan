from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest

from llmplan.workload.schema import Distribution, Workload


def frame(**overrides: Any) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "arrival_s": np.array([0.0, 1.0, 1.0], dtype=np.float64),
        "input_tokens": np.array([1, 5, 9], dtype=np.int64),
        "output_tokens": np.array([0, 2, 3], dtype=np.int64),
        "model": pd.array(["a", None, "b"], dtype="string"),
        "tenant": pd.array([None, None, None], dtype="string"),
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def make(df: pd.DataFrame) -> Workload:
    return Workload(source="test", format="csv", frame=df, dropped_rows=0)


def test_valid_workload() -> None:
    workload = make(frame())
    assert workload.notes == ()
    assert len(workload.frame) == 3


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (frame().drop(columns="tenant"), "columns"),
        (frame()[["arrival_s", "output_tokens", "input_tokens", "model", "tenant"]], "columns"),
        (frame().set_axis([1, 2, 3]), "RangeIndex"),
        (frame().iloc[:0], "at least one"),
        (frame(arrival_s=np.array([0, 1, 2], dtype=np.int64)), "arrival_s"),
        (frame(input_tokens=np.array([1.0, 2.0, 3.0])), "input_tokens"),
        (frame(model=np.array(["a", "b", "c"], dtype=object)), "model"),
        (frame(tenant=pd.array(["a", "b", "c"], dtype=pd.StringDtype(na_value=np.nan))), "tenant"),
        (frame(arrival_s=np.array([0.5, 1.0, 2.0])), "start at 0.0"),
        (frame(arrival_s=np.array([0.0, np.inf, np.inf])), "finite"),
        (frame(arrival_s=np.array([0.0, 2.0, 1.0])), "non-decreasing"),
        (frame(input_tokens=np.array([0, 1, 1], dtype=np.int64)), "input_tokens"),
        (frame(output_tokens=np.array([-1, 1, 1], dtype=np.int64)), "output_tokens"),
    ],
)
def test_invalid_frames(df: pd.DataFrame, message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        make(df)


def test_workload_is_frozen() -> None:
    workload = make(frame())
    with pytest.raises(pydantic.ValidationError):
        workload.dropped_rows = 3  # type: ignore[misc]  # asserting frozen at runtime


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "fixed", "value": 7},
        {"kind": "lognormal", "mean": 6.2, "sigma": 0.8, "lo": 1, "hi": 4096},
        {"kind": "uniform", "lo": 3, "hi": 3},
    ],
)
def test_valid_distributions(kwargs: dict[str, Any]) -> None:
    Distribution(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "fixed"}, "requires 'value'"),
        ({"kind": "fixed", "value": 5, "mean": 1.0}, "does not take 'mean'"),
        ({"kind": "lognormal", "mean": 1.0}, "requires 'sigma'"),
        ({"kind": "uniform", "value": 3}, "does not take 'value'"),
        ({"kind": "uniform", "lo": 5, "hi": 4}, "lo"),
        ({"kind": "fixed", "value": 0}, r"\[lo, hi\]"),
        ({"kind": "lognormal", "mean": float("nan"), "sigma": 1.0}, "finite"),
        ({"kind": "lognormal", "mean": 1.0, "sigma": -1.0}, "greater than or equal"),
        ({"kind": "normal"}, "kind"),
    ],
)
def test_invalid_distributions(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        Distribution(**kwargs)
