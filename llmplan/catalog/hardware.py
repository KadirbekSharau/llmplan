"""Hardware catalog: `GPUSpec` and `PriceRow` loaded from YAML (M1_DESIGN.md section 7).

Both files are lists of rows validated on load; one bad row fails the whole load with the
row index and field. Every row carries `source_url` and `as_of`; unknown numbers are null.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmplan.errors import CatalogError

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_GPUS_PATH = DATA_DIR / "gpus.yaml"
DEFAULT_PRICES_PATH = DATA_DIR / "prices.yaml"

_ID = r"^[a-z0-9][a-z0-9._-]*$"
_URL = r"^https://\S+$"

_Row = TypeVar("_Row", bound=BaseModel)


class GPUSpec(BaseModel):
    """One GPU model. `vram_bytes` is the nvidia-smi total; TFLOPS are dense, not sparse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_ID)
    vendor: Literal["nvidia"]
    name: str = Field(min_length=1)
    vram_bytes: int = Field(gt=0)
    memory_bandwidth_gbps: float | None = Field(gt=0)
    fp16_dense_tflops: float | None = Field(gt=0)
    fp8_dense_tflops: float | None = Field(gt=0)
    nvlink: bool
    source_url: str = Field(pattern=_URL)
    as_of: date


class PriceRow(BaseModel):
    """Price of one cloud instance (all its GPUs) under one commitment type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=_ID)
    instance: str = Field(min_length=1)
    gpu_id: str = Field(pattern=_ID)
    gpu_count: int = Field(gt=0)
    price_usd_per_hour: float = Field(gt=0)
    commitment: Literal["on_demand", "reserved_1y", "reserved_3y", "spot"]
    region: str | None
    source_url: str = Field(pattern=_URL)
    as_of: date


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read catalog {path}: {exc.strerror}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path.name}: invalid YAML: {exc}") from None
    if not isinstance(data, list):
        raise CatalogError(f"{path.name}: expected a list of rows at top level")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise CatalogError(f"{path.name} row {index}: expected a mapping")
    return data


def _validate(model: type[_Row], row: dict[str, Any], where: str) -> _Row:
    try:
        return model.model_validate(row)
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "row"
        raise CatalogError(f"{where}: field {field!r}: {err['msg']}") from None


def load_gpus(path: Path | None = None) -> Mapping[str, GPUSpec]:
    """Load the GPU catalog (default `data/gpus.yaml`) as a read-only mapping keyed by id.

    Raises `CatalogError` naming the file, row index, and field for any invalid row, and for
    duplicate ids.
    """
    path = path or DEFAULT_GPUS_PATH
    gpus: dict[str, GPUSpec] = {}
    for index, row in enumerate(_read_rows(path)):
        gpu = _validate(GPUSpec, row, f"{path.name} row {index}")
        if gpu.id in gpus:
            raise CatalogError(f"{path.name} row {index}: duplicate gpu id {gpu.id!r}")
        gpus[gpu.id] = gpu
    return MappingProxyType(gpus)


def load_prices(
    path: Path | None = None, *, gpus: Mapping[str, GPUSpec] | None = None
) -> tuple[PriceRow, ...]:
    """Load the price catalog (default `data/prices.yaml`).

    Every row's `gpu_id` must exist in `gpus` (default: the shipped GPU catalog); otherwise
    `CatalogError` names the row index and the unknown id.
    """
    path = path or DEFAULT_PRICES_PATH
    known = load_gpus() if gpus is None else gpus
    rows: list[PriceRow] = []
    for index, raw in enumerate(_read_rows(path)):
        row = _validate(PriceRow, raw, f"{path.name} row {index}")
        if row.gpu_id not in known:
            raise CatalogError(f"{path.name} row {index}: unknown gpu_id {row.gpu_id!r}")
        rows.append(row)
    return tuple(rows)
