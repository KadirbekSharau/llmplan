"""M1 acceptance tests: M1_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llmplan.catalog import architectures
from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.catalog.models import HttpConfigFetcher, ModelSpec, load_model
from llmplan.errors import CatalogError, FetchError, UnsupportedArchitecture


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


def test_9_6_price_row_unknown_gpu(tmp_path: Path) -> None:
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        """
- provider: aws
  instance: g6.xlarge
  gpu_id: l4-24gb
  gpu_count: 1
  price_usd_per_hour: 0.8
  commitment: on_demand
  region: us-east-1
  source_url: https://aws.amazon.com/ec2/pricing/on-demand/
  as_of: 2026-09-30
- provider: aws
  instance: x9.huge
  gpu_id: b999-sxm
  gpu_count: 8
  price_usd_per_hour: 10.0
  commitment: on_demand
  region: us-east-1
  source_url: https://aws.amazon.com/ec2/pricing/on-demand/
  as_of: 2026-09-30
"""
    )
    with pytest.raises(CatalogError) as info:
        load_prices(prices)
    assert "row 1" in str(info.value)
    assert "b999-sxm" in str(info.value)


# 7.1 pinned catalog values that the section 9 numbers depend on
def test_7_1_pinned_vram() -> None:
    gpus = load_gpus()
    assert gpus["h100-sxm-80gb"].vram_bytes == 81559 * 2**20
    assert gpus["a10g-24gb"].vram_bytes == 23028 * 2**20
