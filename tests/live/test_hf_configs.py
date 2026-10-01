"""Live Hugging Face checks (M8_DESIGN.md section 7): run by hand before a release.

Marked `network` and deselected by default (never in CI): `uv run pytest -m network
tests/live --no-cov -v -s`. Each supported family's real config.json is fetched from
huggingface.co, its parameter count must be within 1.5% of the official figure recorded
here, and `fit()` must run. The gated Llama repo is skipped without `HF_TOKEN`.
"""

from __future__ import annotations

import os

import pytest

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.memory.fit import FitRequest, fit
from llmplan.memory.weights import model_info

pytestmark = pytest.mark.network

TOLERANCE = 0.015
# repo id -> (official parameter count, where the figure is published)
OFFICIAL = {
    "Qwen/Qwen2.5-7B-Instruct": (7.61e9, "model card: Number of Parameters 7.61B"),
    "mistralai/Mistral-7B-v0.3": (7.25e9, "Hugging Face model page: 7.25B params"),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (46.7e9, "Mistral AI: 46.7B total parameters"),
    "Qwen/Qwen3-30B-A3B": (30.5e9, "model card: 30.5B total, 3.3B activated"),
    "meta-llama/Llama-3.1-8B-Instruct": (8.03e9, "Hugging Face model page: 8.03B params"),
}
GATED = {"meta-llama/Llama-3.1-8B-Instruct"}


@pytest.mark.parametrize("repo_id", list(OFFICIAL))
def test_live_config_param_count_and_fit(repo_id: str) -> None:
    if repo_id in GATED and not os.environ.get("HF_TOKEN"):
        pytest.skip(f"{repo_id} is gated; set HF_TOKEN to check it")
    official, where = OFFICIAL[repo_id]
    spec = load_model(repo_id)
    info = model_info(spec)
    error = abs(info.param_count - official) / official
    result = fit(
        FitRequest(model=spec, gpu=load_gpus()["h100-sxm-80gb"], dtype="bf16", context_len=4096)
    )
    print(
        f"{repo_id}: {spec.architecture}, params {info.param_count:,} (active "
        f"{info.active_param_count:,}), official {official / 1e9:g}B ({where}), off by "
        f"{error * 100:.2f}%; fit bf16 on 1 x H100: fits={result.fits} binding={result.binding}"
    )
    assert error <= TOLERANCE
