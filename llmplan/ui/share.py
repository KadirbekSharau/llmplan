"""Share links (M9_DESIGN.md section 5): a plan's inputs as URL query parameters (`model`,
`traffic`, and the other inputs under their widget keys), checked on the way back against
the request models (`SLO`, `PlanOptions`, `Distribution`) and the page's choices and
bounds. Uploads and price edits are not shareable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from llmplan.catalog.models import is_repo_id
from llmplan.errors import LLMPlanError
from llmplan.planner import SLO, PlanOptions
from llmplan.ui.presets import BOUNDS, CUSTOM_MODEL, DEFAULTS, OPTION_KEYS, TRAFFIC_MODES
from llmplan.workload.synth import parse_distribution

MAX_URL_CHARS = 2000
Choices = Mapping[str, Sequence[Any]]
SYNTHETIC = ("syn_rate", "syn_duration", "syn_in", "syn_out", "syn_seed")
FIELDS = ("ttft", "tpot", "utilization", "classes", *OPTION_KEYS)
_SLO = {"ttft": "ttft_ms_p95", "tpot": "tpot_ms_p95", "utilization": "utilization_target"}
_CHOICE_FIELDS = {"tensor_parallel": "tensor_parallel_choices", "dtypes": "dtype_choices"}
_CHOICE_FIELDS["max_num_seqs"] = "max_num_seqs_choices"  # PlanOptions names the rest alike


def _text(value: Any) -> str:
    if value is None:
        return "none"
    return ",".join(map(str, value)) if isinstance(value, list | tuple) else str(value)


def encode(values: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
    """Query parameters that reproduce the inputs in `values`, and whether the traffic is
    shareable (False for an upload, which the link leaves out)."""
    custom = values["model_choice"] == CUSTOM_MODEL
    params = {"model": str(values["model_custom"]).strip() if custom else values["model_choice"]}
    mode = values["traffic_mode"]
    if mode == TRAFFIC_MODES[0]:
        params["traffic"] = values["preset"]
    elif mode == TRAFFIC_MODES[2]:
        params |= {"traffic": "synthetic", **{key: _text(values[key]) for key in SYNTHETIC}}
    params |= {key: _text(values[key]) for key in FIELDS}
    return params, mode != TRAFFIC_MODES[1]


def link(base: str, params: Mapping[str, str]) -> str | None:
    """`base` (the page URL, its own query dropped) with `params`; None over 2,000 chars."""
    url = f"{base.split('?')[0]}?{urlencode(params, safe=',:/')}"
    return url if len(url) <= MAX_URL_CHARS else None


def decode(params: Mapping[str, str], choices: Choices) -> tuple[dict[str, Any], list[str]]:
    """The input values in URL `params`, and the names of the parameters ignored because
    they are unknown or invalid. `choices` holds the options of each choice input, plus
    `model_choice` (the listed models) and `preset` (the preset keys)."""
    values: dict[str, Any] = {}
    ignored = []
    for name, raw in params.items():
        try:
            values |= _decode(name, raw, choices)
        except (KeyError, ValueError, LLMPlanError):  # pydantic's ValidationError included
            ignored.append(name)
    return values, ignored


def _pick(raw: str, options: Sequence[Any]) -> Any:
    return {str(option): option for option in options}[raw]


def _decode(name: str, raw: str, choices: Choices) -> dict[str, Any]:
    if name == "model":
        if raw in choices["model_choice"]:
            return {"model_choice": raw}
        if is_repo_id(raw):  # the Hugging Face fetcher's own rule
            return {"model_choice": CUSTOM_MODEL, "model_custom": raw}
        raise ValueError(name)
    if name == "traffic":
        if raw == "synthetic":
            return {"traffic_mode": TRAFFIC_MODES[2]}
        return {"traffic_mode": TRAFFIC_MODES[0], "preset": _pick(raw, choices["preset"])}
    if name in ("syn_in", "syn_out"):
        parse_distribution(raw)  # raises ValidationError for a bad distribution
        return {name: raw}
    if name not in (*FIELDS, *SYNTHETIC):
        raise KeyError(name)
    default = DEFAULTS[name]
    value: Any
    if raw == "none" and name in ("ttft", "tpot"):
        value = None  # no target
    elif name in BOUNDS:
        value = int(raw) if isinstance(default, int) else float(raw)
        if not BOUNDS[name][0] <= value <= BOUNDS[name][1]:  # NaN fails too
            raise ValueError(name)
    elif isinstance(default, list):
        value = [_pick(item, choices[name]) for item in raw.split(",")]
    else:
        value = _pick(raw, choices[name])
    if name in _SLO:
        SLO.model_validate({_SLO[name]: value})
    if name in OPTION_KEYS:
        PlanOptions.model_validate({"max_model_len": 1, _CHOICE_FIELDS.get(name, name): value})
    return {name: value}
