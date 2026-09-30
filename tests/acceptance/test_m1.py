"""M1 acceptance tests: M1_DESIGN.md section 9. These define done; do not relax them."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from llmplan.catalog import architectures
from llmplan.catalog.hardware import GPUSpec, load_gpus, load_prices
from llmplan.catalog.models import HttpConfigFetcher, ModelSpec, load_model
from llmplan.cli import app
from llmplan.errors import CatalogError, FetchError, UnsupportedArchitecture, ValidationError
from llmplan.memory.engine import EngineProfile
from llmplan.memory.fit import FitRequest, FitResult, fit
from llmplan.memory.kv_cache import kv_bytes_per_token_per_gpu, kv_bytes_per_token_total
from llmplan.memory.weights import per_gpu_weight_bytes, weight_bytes
from llmplan.types import DType


def _spec(name: str) -> ModelSpec:
    return load_model(f"fixture:{name}")


def count_params(spec: ModelSpec) -> int:
    return architectures.get(spec.architecture).count_params(spec)


def _gpu(gpu_id: str) -> GPUSpec:
    return load_gpus()[gpu_id]


def _fit(model: str, gpu: str, *, tp: int = 1, dtype: DType = "bf16", ctx: int = 8192) -> FitResult:
    return fit(
        FitRequest(
            model=_spec(model),
            gpu=_gpu(gpu),
            engine=EngineProfile(),
            tensor_parallel=tp,
            dtype=dtype,
            context_len=ctx,
        )
    )


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


# 9.2 KV bytes per token (exact)
@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("llama3-70b", 327_680),
        ("llama3-8b", 131_072),
        ("mistral-7b-v0.1", 131_072),
        ("qwen2.5-7b", 57_344),
    ],
)
def test_9_2_kv_bytes_per_token_bf16(fixture: str, expected: int) -> None:
    assert kv_bytes_per_token_total(_spec(fixture), "bf16") == expected


def test_9_2_kv_bytes_per_token_fp8() -> None:
    assert kv_bytes_per_token_total(_spec("llama3-70b"), "fp8") == 163_840


# 9.3 Weight bytes
def test_9_3_weight_bytes_llama3_70b_bf16() -> None:
    assert weight_bytes(_spec("llama3-70b"), "bf16") == 141_107_412_992


def test_9_3_weight_bytes_llama3_8b_int4_16bit_embeddings() -> None:
    expected = (8_030_261_248 - 2 * 525_336_576) * 0.5 + 2 * 525_336_576 * 2
    assert expected == 3_489_794_048 + 2_101_346_304 == 5_591_140_352
    assert weight_bytes(_spec("llama3-8b"), "int4") == 5_591_140_352


# 9.4 Fit outcomes on h100-sxm-80gb (81559 MiB), default EngineProfile, context 8192
@pytest.mark.parametrize(
    ("model", "tp", "fits", "binding", "cap_lo", "cap_hi", "conc_lo", "conc_hi"),
    [
        ("llama3-70b", 1, False, "weights", 0, 0, 0, 0),
        ("llama3-70b", 2, True, "ok", 15_000, 25_000, 2, 2),
        ("llama3-70b", 4, True, "ok", 400_000, 520_000, 50, 60),
        ("llama3-8b", 1, True, "ok", 400_000, 500_000, 48, 60),
    ],
)
def test_9_4_fit_h100(
    model: str,
    tp: int,
    fits: bool,
    binding: str,
    cap_lo: int,
    cap_hi: int,
    conc_lo: int,
    conc_hi: int,
) -> None:
    result = _fit(model, "h100-sxm-80gb", tp=tp)
    assert result.fits is fits
    assert result.binding == binding
    assert cap_lo <= result.kv_token_capacity <= cap_hi
    assert conc_lo <= result.max_concurrent_seqs_at_context <= conc_hi


def test_9_4_reference_arithmetic_tp2() -> None:
    result = _fit("llama3-70b", "h100-sxm-80gb", tp=2)
    assert result.per_gpu_usable_bytes == 76_968_728_985
    assert result.per_gpu_weight_bytes == 70_553_706_496
    assert result.per_gpu_overhead_bytes == 3_221_225_472
    assert result.per_gpu_kv_budget_bytes == 3_193_797_017
    assert result.kv_bytes_per_token_per_gpu == 163_840
    assert result.kv_token_capacity == 19_493
    assert result.max_concurrent_seqs_at_context == 2


# 9.5 Fit on a10g-24gb (23028 MiB)
def test_9_5_a10g_llama3_8b() -> None:
    result = _fit("llama3-8b", "a10g-24gb")
    assert result.fits is True
    assert 20_000 <= result.kv_token_capacity <= 35_000
    assert result.max_concurrent_seqs_at_context == 3


def test_9_5_a10g_llama3_70b() -> None:
    result = _fit("llama3-70b", "a10g-24gb")
    assert result.fits is False
    assert result.binding == "weights"


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


def test_9_6_tensor_parallel_must_divide_heads() -> None:
    with pytest.raises(ValidationError) as info:
        _fit("llama3-8b", "h100-sxm-80gb", tp=3)
    assert "3" in str(info.value)
    assert "32" in str(info.value)


def test_9_6_context_exceeds_max_position_embeddings() -> None:
    with pytest.raises(ValidationError):
        _fit("llama3-8b", "h100-sxm-80gb", ctx=9000)


# 9.7 Property tests (hypothesis)
@st.composite
def valid_specs(draw: st.DrawFn) -> ModelSpec:
    kv_heads = draw(st.integers(1, 16))
    heads = kv_heads * draw(st.integers(1, 8))
    head_dim = draw(st.sampled_from([32, 64, 80, 96, 128, 256]))
    return ModelSpec(
        id="manual:hypothesis",
        architecture="llama_like",
        hidden_size=heads * head_dim,
        num_layers=draw(st.integers(1, 128)),
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=draw(st.integers(1, 65_536)),
        vocab_size=draw(st.integers(1, 262_144)),
        tie_word_embeddings=draw(st.booleans()),
        attention_bias=draw(st.booleans()),
        mlp_bias=draw(st.booleans()),
        qk_norm=draw(st.booleans()),
        max_position_embeddings=131_072,
        source="manual",
    )


def _divisors(n: int) -> list[int]:
    return [tp for tp in range(1, n + 1) if n % tp == 0]


@given(valid_specs())
def test_9_7_weight_bytes_monotone_in_dtype_width(spec: ModelSpec) -> None:
    # Width chain with every parameter stored at the dtype width. With the default 16-bit
    # embeddings, int4 > fp8 whenever embeddings exceed a third of all parameters (see
    # M1_NOTES.md), so the default int4 path is bounded by bf16 instead.
    assert (
        weight_bytes(spec, "fp32", quantize_embeddings=True)
        >= weight_bytes(spec, "bf16", quantize_embeddings=True)
        >= weight_bytes(spec, "fp8", quantize_embeddings=True)
        >= weight_bytes(spec, "int4", quantize_embeddings=True)
    )
    assert weight_bytes(spec, "bf16") >= weight_bytes(spec, "int8") >= weight_bytes(spec, "int4")


@given(valid_specs(), st.sampled_from(["fp32", "bf16", "fp8", "int8", "int4"]))
def test_9_7_per_gpu_weight_bytes_non_increasing_in_tp(spec: ModelSpec, dtype: DType) -> None:
    values = [per_gpu_weight_bytes(spec, dtype, tp) for tp in _divisors(spec.num_attention_heads)]
    assert values == sorted(values, reverse=True)


@given(valid_specs(), st.sampled_from(["bf16", "fp16", "fp8"]))
def test_9_7_kv_replication_never_loses_kv(spec: ModelSpec, kv_dtype: str) -> None:
    total = kv_bytes_per_token_total(spec, kv_dtype)  # type: ignore[arg-type]  # sampled KVDType
    for tp in _divisors(spec.num_attention_heads):
        per_gpu = kv_bytes_per_token_per_gpu(spec, kv_dtype, tp)  # type: ignore[arg-type]  # sampled KVDType
        assert per_gpu * tp >= total


# 9.8 CLI smoke
FIT_ARGS = ["fit", "--model", "fixture:llama3-70b", "--gpu", "h100-sxm-80gb"]


def test_9_8_cli_fit_tp2_json() -> None:
    result = CliRunner().invoke(app, [*FIT_ARGS, "--tp", "2", "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["fits"] is True


def test_9_8_cli_fit_tp1_is_a_valid_false_answer() -> None:
    result = CliRunner().invoke(app, [*FIT_ARGS, "--tp", "1", "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["fits"] is False


def test_9_8_cli_unknown_gpu_exits_3() -> None:
    args = ["fit", "--model", "fixture:llama3-70b", "--gpu", "no-such-gpu"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 3
