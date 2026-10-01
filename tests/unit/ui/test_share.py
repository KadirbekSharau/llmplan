"""Share links (M9 section 5): encode and decode round trips, and what is ignored."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest

from llmplan.catalog.hardware import load_gpus
from llmplan.ui import presets, share

GPUS = list(load_gpus())
PROVIDERS = ["aws", "lambda", "runpod"]
CHOICES: dict[str, Any] = {
    **presets.CHOICES,
    "model_choice": presets.MODEL_CHOICES,
    "preset": [p.key for p in presets.PRESETS],
    "gpu_ids": GPUS,
    "providers": PROVIDERS,
}


def inputs(**changes: Any) -> dict[str, Any]:
    return {**presets.DEFAULTS, "gpu_ids": GPUS[:2], "providers": PROVIDERS, **changes}


def round_trip(values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    params, _ = share.encode(values)
    url = share.link("https://llmplan.dev/?old=1", params)
    assert url is not None
    assert url.startswith("https://llmplan.dev/?model=")
    return share.decode(dict(parse_qsl(urlsplit(url).query)), CHOICES)


def test_a_preset_plan_round_trips() -> None:
    values = inputs(ttft=None, tpot=30.0, utilization=0.65, tp=[1, 2], classes="3x3")
    decoded, ignored = round_trip(values)
    assert ignored == []
    assert {key: values[key] for key in decoded} == decoded
    assert "syn_rate" not in decoded


def test_synthetic_traffic_and_a_custom_model_round_trip() -> None:
    values = inputs(
        model_choice=presets.CUSTOM_MODEL,
        model_custom=" acme/My-Model_1.5 ",
        traffic_mode=presets.TRAFFIC_MODES[2],
        syn_rate=0.5,
        syn_in="fixed:100",
        syn_seed=7,
    )
    decoded, ignored = round_trip(values)
    assert ignored == []
    assert decoded["model_custom"] == "acme/My-Model_1.5"
    assert (decoded["syn_rate"], decoded["syn_in"], decoded["syn_seed"]) == (0.5, "fixed:100", 7)


def test_uploads_are_not_shareable() -> None:
    params, shareable = share.encode(inputs(traffic_mode=presets.TRAFFIC_MODES[1]))
    assert not shareable
    assert "traffic" not in params


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("ttft", "-5"),
        ("ttft", "nan"),
        ("utilization", "2"),
        ("max_model_len", "8192.5"),
        ("gpu_ids", "h100-sxm-80gb,b999"),
        ("tp", "1,1"),
        ("tp", ""),
        ("classes", "9x9"),
        ("model", "not a repo id"),
        ("model", "a/../b"),
        ("traffic", "nope"),
        ("syn_in", "gaussian:1"),
        ("unknown", "1"),
    ],
)
def test_invalid_or_unknown_parameters_are_ignored(name: str, raw: str) -> None:
    decoded, ignored = share.decode({name: raw, "tpot": "40"}, CHOICES)
    assert ignored == [name]
    assert decoded == {"tpot": 40.0}


def test_long_links_are_refused() -> None:
    assert share.link("", {"model": "x" * share.MAX_URL_CHARS}) is None
