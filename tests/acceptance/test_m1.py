"""M1 acceptance tests: M1_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

import httpx
import pytest

from llmplan.catalog import architectures
from llmplan.catalog.models import HttpConfigFetcher, ModelSpec, load_model
from llmplan.errors import FetchError, UnsupportedArchitecture


def _spec(name: str) -> ModelSpec:
    return load_model(f"fixture:{name}")


def count_params(spec: ModelSpec) -> int:
    return architectures.get(spec.architecture).count_params(spec)


# 9.1 Parameter counts (exact)
@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("llama3-70b", 70_553_706_496),
        ("llama3-8b", 8_030_261_248),
        ("mistral-7b-v0.1", 7_241_732_096),
        ("qwen2.5-7b", 7_615_616_512),
    ],
)
def test_9_1_param_counts(fixture: str, expected: int) -> None:
    assert count_params(_spec(fixture)) == expected


# 9.6 Validation and errors (catalog cases)
def test_9_6_unsupported_architecture() -> None:
    with pytest.raises(UnsupportedArchitecture) as info:
        load_model("fixture:gpt2")
    assert info.value.field == "architectures"


def test_9_6_fetch_rejects_traversal_without_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    fetcher = HttpConfigFetcher(transport=httpx.MockTransport(handler))
    with pytest.raises(FetchError):
        fetcher.fetch("evil/../x")
    assert requests == []
