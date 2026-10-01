"""Benchmark table: row schema, YAML loader, and load-time validation (M3_DESIGN.md 5.1-5.2).

`data/benchmarks/<gpu-id>.yaml` files hold lists of published measurements;
`data/benchmarks/aliases.yaml` maps model ids (fixtures, renamed repos) to the canonical id
rows are recorded under. Every row is checked against the GPU catalog, a model fixture, and
the physical floor of the roofline model before it can be used.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.hardware import DATA_DIR, GPUSpec, load_gpus
from llmplan.catalog.models import FIXTURE_PREFIX, ModelSpec, load_model
from llmplan.errors import BenchmarkError, CatalogError
from llmplan.memory.kv_cache import kv_bytes_per_token_per_gpu
from llmplan.memory.weights import per_gpu_weight_bytes
from llmplan.perf.roofline import decode_compute_s, decode_memory_s, dense_tflops, param_count
from llmplan.types import DType

DEFAULT_BENCHMARKS_DIR = DATA_DIR / "benchmarks"
ALIASES_FILE = "aliases.yaml"
USER_UPLOAD = "user-upload"  # M8: source_url of a row uploaded for one session

_ID = r"^[a-z0-9][a-z0-9._-]*$"
_URL = r"^(https://\S+|user-upload)$"


class BenchmarkRow(BaseModel):
    """One published measurement: a model on a GPU at a concurrency and request shape.

    `output_tokens_per_s` is aggregate output throughput across `concurrency` concurrent
    requests of `input_len` prompt and `output_len` generated tokens. Latencies are `None`
    when the source does not print them (or not their statistic). `engine` records the
    serving stack measured; `"nim"` is an NVIDIA NIM container whose inner engine the source
    does not name, with the container version as `engine_version`. `source_url` is an https
    URL, or (M8) `"user-upload"` for a row uploaded for one session, which a shipped table
    may not contain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    gpu_id: str = Field(pattern=_ID)
    engine: Literal["vllm", "trtllm", "sglang", "nim"]
    engine_version: str = Field(min_length=1)
    tensor_parallel: int = Field(ge=1)
    dtype: DType
    concurrency: int = Field(ge=1)
    input_len: int = Field(ge=1)
    output_len: int = Field(ge=1)
    output_tokens_per_s: float = Field(gt=0)
    ttft_ms_p50: float | None = Field(gt=0)
    ttft_ms_p95: float | None = Field(gt=0)
    tpot_ms_p50: float | None = Field(gt=0)
    tpot_ms_p95: float | None = Field(gt=0)
    source_url: str = Field(pattern=_URL)
    as_of: date


class BenchmarkTable(BaseModel):
    """Validated benchmark rows plus the model-id alias map used to match them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[BenchmarkRow, ...]
    aliases: dict[str, str]

    def canonical(self, model_id: str) -> str:
        """The id rows are recorded under for `model_id` (itself when not aliased)."""
        return self.aliases.get(model_id, model_id)


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read {path}: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise BenchmarkError(f"{path.name}: invalid YAML: {exc}") from None


def _read_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = _read_yaml(path)
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise BenchmarkError(f"{path.name}: expected a mapping of model id -> canonical id")
    for key, value in data.items():
        if value in data:
            raise BenchmarkError(f"{path.name}: alias {key!r} -> {value!r} chains to another alias")
    return data


def _fixture_for(model_id: str, aliases: Mapping[str, str]) -> str | None:
    if model_id.startswith(FIXTURE_PREFIX):
        return model_id
    canonical = aliases.get(model_id, model_id)
    fixtures = sorted(
        key
        for key, value in aliases.items()
        if key.startswith(FIXTURE_PREFIX) and value == canonical
    )
    return fixtures[0] if fixtures else None


def physical_floor_s(row: BenchmarkRow, model: ModelSpec, gpu: GPUSpec) -> float | None:
    """Absolute lower bound on seconds per decode step at the row's concurrency and shape.

    Roofline step time at 100% bandwidth efficiency and 100% MFU, with the smallest storage
    the row could have used (fp8 KV cache, embeddings at the weight dtype), at context
    `input_len + output_len / 2`. `None` when the GPU has neither bandwidth nor TFLOPS.
    """
    tp = row.tensor_parallel
    terms: list[float] = []
    if gpu.memory_bandwidth_gbps is not None:
        weights = per_gpu_weight_bytes(model, row.dtype, tp, quantize_embeddings=True)
        kv = kv_bytes_per_token_per_gpu(model, "fp8", tp)
        ctx = row.input_len + row.output_len / 2
        terms.append(
            decode_memory_s(
                weights, kv, row.concurrency, ctx, gpu.memory_bandwidth_gbps, efficiency=1.0
            )
        )
    tflops = dense_tflops(gpu, row.dtype)
    if tflops is not None:
        terms.append(decode_compute_s(param_count(model), row.concurrency, tp, tflops, mfu=1.0))
    return max(terms) if terms else None


def row_problem(row: BenchmarkRow, model: ModelSpec, gpu: GPUSpec) -> str | None:
    """Why `row` cannot be a measurement of `model` on `gpu`, or None: tensor parallelism
    must divide the attention heads, and the implied time per token per sequence must not
    beat the physical floor (`physical_floor_s`, which needs bandwidth or TFLOPS)."""
    if model.num_attention_heads % row.tensor_parallel != 0:
        return (
            f"tensor_parallel {row.tensor_parallel} does not divide "
            f"num_attention_heads {model.num_attention_heads}"
        )
    floor_s = physical_floor_s(row, model, gpu)
    if floor_s is None:
        return (
            f"gpu {row.gpu_id} has neither memory_bandwidth_gbps nor dense TFLOPS, "
            "so the physical bound cannot be checked"
        )
    implied_s = row.concurrency / row.output_tokens_per_s
    if implied_s < floor_s:
        return (
            f"output_tokens_per_s {row.output_tokens_per_s:g} implies "
            f"{implied_s * 1e3:.3f} ms per token per sequence, below the physical floor "
            f"{floor_s * 1e3:.3f} ms"
        )
    return None


def _check_row(
    row: BenchmarkRow,
    where: str,
    path: Path,
    gpus: Mapping[str, GPUSpec],
    aliases: Mapping[str, str],
) -> None:
    if row.source_url == USER_UPLOAD:
        raise BenchmarkError(f"{where}: source_url {USER_UPLOAD!r} is reserved for uploads")
    if row.gpu_id not in gpus:
        raise BenchmarkError(f"{where}: unknown gpu_id {row.gpu_id!r}")
    if row.gpu_id != path.stem:
        raise BenchmarkError(f"{where}: gpu_id {row.gpu_id!r} does not match file {path.name}")
    fixture = _fixture_for(row.model_id, aliases)
    if fixture is None:
        raise BenchmarkError(
            f"{where}: no fixture for this model, so the physical bound cannot be checked; "
            f"map a fixture to {row.model_id!r} in {ALIASES_FILE}"
        )
    try:
        model = load_model(fixture)
    except CatalogError as exc:
        raise BenchmarkError(f"{where}: {exc}") from None
    problem = row_problem(row, model, gpus[row.gpu_id])
    if problem is not None:
        raise BenchmarkError(f"{where}: {problem}")


def load_benchmarks(
    directory: Path | None = None, *, gpus: Mapping[str, GPUSpec] | None = None
) -> BenchmarkTable:
    """Load and validate every `<gpu-id>.yaml` in `directory` (default `data/benchmarks`).

    `gpus` is the FK target (default: the shipped GPU catalog). Raises `BenchmarkError`
    naming the file, row index, and `model_id` for a malformed row, an unknown GPU or
    fixture, a model with no fixture alias, or a row faster than the physical floor.
    """
    directory = directory or DEFAULT_BENCHMARKS_DIR
    known = load_gpus() if gpus is None else gpus
    aliases = _read_aliases(directory / ALIASES_FILE)
    rows: list[BenchmarkRow] = []
    for path in sorted(p for p in directory.glob("*.yaml") if p.name != ALIASES_FILE):
        data = _read_yaml(path)
        if not isinstance(data, list):
            raise BenchmarkError(f"{path.name}: expected a list of rows at top level")
        for index, raw in enumerate(data):
            model_id = raw.get("model_id") if isinstance(raw, dict) else None
            where = f"{path.name} row index {index} (model_id {model_id!r})"
            try:
                row = BenchmarkRow.model_validate(raw)
            except pydantic.ValidationError as exc:
                err = exc.errors()[0]
                field = ".".join(str(p) for p in err["loc"]) or "row"
                raise BenchmarkError(f"{where}: field {field!r}: {err['msg']}") from None
            _check_row(row, where, path, known, aliases)
            rows.append(row)
    return BenchmarkTable(rows=tuple(rows), aliases=aliases)


@functools.cache
def default_table() -> BenchmarkTable:
    """The shipped benchmark table, loaded and validated once per process."""
    return load_benchmarks()
