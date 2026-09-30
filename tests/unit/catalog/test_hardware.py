from __future__ import annotations

from pathlib import Path

import pytest

from llmplan.catalog.hardware import load_gpus, load_prices
from llmplan.errors import CatalogError

GPU_ROW = """
- id: test-gpu
  vendor: nvidia
  name: Test GPU
  vram_bytes: 1000
  memory_bandwidth_gbps: null
  fp16_dense_tflops: null
  fp8_dense_tflops: null
  nvlink: false
  source_url: https://example.com/gpu
  as_of: 2026-09-30
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_shipped_gpus_have_sources() -> None:
    gpus = load_gpus()
    assert set(gpus) == {
        "h100-sxm-80gb",
        "h200-sxm-141gb",
        "a100-sxm-80gb",
        "a100-sxm-40gb",
        "l40s-48gb",
        "l4-24gb",
        "a10g-24gb",
    }
    for gpu in gpus.values():
        assert gpu.source_url.startswith("https://")
        assert gpu.vram_bytes % 2**20 == 0


def test_shipped_prices_reference_gpus() -> None:
    gpus = load_gpus()
    rows = load_prices()
    assert len(rows) == 8
    assert {r.provider for r in rows} == {"aws", "lambda", "runpod"}
    for row in rows:
        assert row.gpu_id in gpus
        assert row.price_usd_per_hour > 0


def test_catalog_is_read_only() -> None:
    gpus = load_gpus()
    with pytest.raises(TypeError):
        gpus["x"] = gpus["l4-24gb"]  # type: ignore[index]  # asserting immutability


def test_custom_gpu_file(tmp_path: Path) -> None:
    gpus = load_gpus(_write(tmp_path, "g.yaml", GPU_ROW))
    assert gpus["test-gpu"].vram_bytes == 1000


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{a: 1}", "list of rows"),
        ("- 1", "row 0: expected a mapping"),
        ("- [unclosed", "invalid YAML"),
        (GPU_ROW.replace("vram_bytes: 1000", "vram_bytes: 0"), "row 0: field 'vram_bytes'"),
        (GPU_ROW.replace("vendor: nvidia", "vendor: amd"), "field 'vendor'"),
        (GPU_ROW.replace("https://example.com/gpu", "ftp://x"), "field 'source_url'"),
        (GPU_ROW + "  extra: 1\n", "field 'extra'"),
        (GPU_ROW + GPU_ROW, "row 1: duplicate gpu id 'test-gpu'"),
    ],
)
def test_bad_gpu_rows(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(CatalogError, match=match):
        load_gpus(_write(tmp_path, "g.yaml", text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="cannot read catalog"):
        load_gpus(tmp_path / "nope.yaml")


def test_bad_price_field(tmp_path: Path) -> None:
    text = """
- provider: aws
  instance: x
  gpu_id: l4-24gb
  gpu_count: 1
  price_usd_per_hour: 1.0
  commitment: forever
  region: null
  source_url: https://example.com
  as_of: 2026-09-30
"""
    with pytest.raises(CatalogError, match="row 0: field 'commitment'"):
        load_prices(_write(tmp_path, "p.yaml", text))


def test_prices_against_custom_gpus(tmp_path: Path) -> None:
    gpus = load_gpus(_write(tmp_path, "g.yaml", GPU_ROW))
    text = """
- provider: onprem
  instance: rack-1
  gpu_id: test-gpu
  gpu_count: 4
  price_usd_per_hour: 2.5
  commitment: reserved_3y
  region: null
  source_url: https://example.com
  as_of: 2026-09-30
"""
    (row,) = load_prices(_write(tmp_path, "p.yaml", text), gpus=gpus)
    assert row.gpu_count == 4
