from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from llmplan.errors import ValidationError
from llmplan.workload import Distribution, Workload, generate, load_workload
from llmplan.workload.formats.generic_csv import write_csv
from llmplan.workload.synth import parse_distribution

FIXED = Distribution(kind="fixed", value=100)


def synth(**overrides: Any) -> Workload:
    args: dict[str, Any] = {
        "rate_rps": 2.0,
        "duration_s": 600.0,
        "input_tokens": FIXED,
        "output_tokens": Distribution(kind="uniform", lo=0, hi=50),
        "seed": 7,
    }
    args.update(overrides)
    return generate(**args)


def test_source_format_and_notes() -> None:
    workload = synth()
    assert workload.source == "synthetic:7"
    assert workload.format == "synthetic"
    assert workload.dropped_rows == 0
    assert workload.notes == ("synthetic Poisson arrivals: rate_rps=2.0, duration_s=600.0",)
    assert synth(diurnal=(1.0,) * 24).notes[0].endswith(", diurnal thinning")


def test_draws_respect_distributions() -> None:
    frame = synth().frame
    assert (frame["input_tokens"] == 100).all()
    assert frame["output_tokens"].between(0, 50).all()
    assert frame["arrival_s"].max() <= 600.0
    assert frame["model"].isna().all()


def test_lognormal_is_clipped() -> None:
    dist = Distribution(kind="lognormal", mean=20.0, sigma=1.0, lo=5, hi=9)
    assert (synth(input_tokens=dist).frame["input_tokens"] == 9).all()


def test_seed_changes_output() -> None:
    assert not synth(seed=1).frame.equals(synth(seed=2).frame)
    pd.testing.assert_frame_equal(synth(seed=3).frame, synth(seed=3).frame)


def test_diurnal_is_a_relative_shape() -> None:
    pd.testing.assert_frame_equal(
        synth(diurnal=(1.0,) * 24).frame, synth(diurnal=(2.5,) * 24).frame
    )


def test_first_arrival_kept_even_when_hour_zero_is_silent() -> None:
    diurnal = (0.0,) + (1.0,) * 23
    frame = synth(duration_s=7200.0, diurnal=diurnal).frame
    assert frame["arrival_s"].iloc[0] == 0.0
    assert frame["arrival_s"].iloc[1] >= 3600.0


def test_round_trip_through_generic_csv(tmp_path: Path) -> None:
    original = synth()
    path = tmp_path / "synth.csv"
    write_csv(original, path)
    pd.testing.assert_frame_equal(load_workload(path).frame, original.frame)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rate_rps": 0.0}, "rate_rps"),
        ({"rate_rps": float("inf")}, "rate_rps"),
        ({"duration_s": -1.0}, "duration_s"),
        ({"rate_rps": 1e6, "duration_s": 1e3}, "exceeds"),
        ({"input_tokens": Distribution(kind="uniform", lo=0, hi=5)}, "lo >= 1"),
        ({"seed": -1}, "seed"),
        ({"diurnal": (1.0,) * 23}, "24 finite"),
        ({"diurnal": (-1.0,) + (1.0,) * 23}, "24 finite"),
        ({"diurnal": (0.0,) * 24}, "positive"),
    ],
)
def test_invalid_arguments(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        synth(**overrides)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("fixed:12", Distribution(kind="fixed", value=12)),
        ("lognormal:6.2:0.8", Distribution(kind="lognormal", mean=6.2, sigma=0.8)),
        (
            "lognormal:5.5:0.9:2:4096",
            Distribution(kind="lognormal", mean=5.5, sigma=0.9, lo=2, hi=4096),
        ),
        ("uniform:10:20", Distribution(kind="uniform", lo=10, hi=20)),
    ],
)
def test_parse_distribution(spec: str, expected: Distribution) -> None:
    assert parse_distribution(spec) == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("normal:1:2", "expected fixed:N"),
        ("fixed", "expected fixed:N"),
        ("lognormal:1:2:3", "expected fixed:N"),
        ("fixed:abc", "invalid literal"),
        ("uniform:9:3", "lo"),
        ("lognormal:1:-2", "greater than or equal"),
    ],
)
def test_parse_distribution_errors(spec: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_distribution(spec)


def test_unvalidated_lognormal_without_parameters_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires mean and sigma"):
        synth(input_tokens=Distribution.model_construct(kind="lognormal", lo=1, hi=10))
