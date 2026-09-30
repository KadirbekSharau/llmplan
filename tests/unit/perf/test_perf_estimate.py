from __future__ import annotations

import pytest

from llmplan.catalog.hardware import GPUSpec, load_gpus
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.errors import PerfError, UnknownRegistryKey, ValidationError
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, estimate
from llmplan.perf.config import fit_for
from tests.unit.perf.stats import FakeStats

MODEL = load_model("fixture:llama3-8b")
GPU = load_gpus()["h100-sxm-80gb"]
CONFIG = ReplicaConfig(max_model_len=4096)

RESULT = PerfEstimate(
    backend="table",
    confidence="measured",
    effective_batch=1,
    decode_tokens_per_s=1.0,
    prefill_tokens_per_s=1.0,
    requests_per_s_capacity=1.0,
    ttft_ms_p50=1.0,
    ttft_ms_p95=1.0,
    tpot_ms_p50=1.0,
    tpot_ms_p95=1.0,
    assumptions=(),
    source_urls=(),
)


class Fake:
    def __init__(self, name: str, result: PerfEstimate | None) -> None:
        self.name = name
        self.result = result
        self.calls = 0

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        self.calls += 1
        return self.result

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        return f"{self.name} says no"


def test_fake_stats_satisfies_protocol() -> None:
    assert isinstance(FakeStats(), StatsLike)
    assert not isinstance(object(), StatsLike)


def test_auto_returns_first_answer_in_order() -> None:
    table, roofline = Fake("table", None), Fake("roofline", RESULT)
    got = estimate(MODEL, GPU, CONFIG, FakeStats(), backends={"table": table, "roofline": roofline})
    assert got == RESULT
    assert (table.calls, roofline.calls) == (1, 1)


def test_named_backend_is_used_alone() -> None:
    table, roofline = Fake("table", RESULT), Fake("roofline", RESULT)
    backends = {"table": table, "roofline": roofline}
    estimate(MODEL, GPU, CONFIG, FakeStats(), backend="roofline", backends=backends)
    assert (table.calls, roofline.calls) == (0, 1)


def test_all_backends_failing_raises_with_reasons() -> None:
    backends = {"table": Fake("table", None), "roofline": Fake("roofline", None)}
    with pytest.raises(PerfError, match=r"table: table says no; roofline: roofline says no"):
        estimate(MODEL, GPU, CONFIG, FakeStats(), backends=backends)


def test_unknown_backend() -> None:
    with pytest.raises(UnknownRegistryKey, match="'nope'"):
        estimate(MODEL, GPU, CONFIG, FakeStats(), backend="nope", backends={})


def test_max_model_len_beyond_model_rejected() -> None:
    with pytest.raises(ValidationError, match="max_model_len 9000 exceeds"):
        estimate(MODEL, GPU, ReplicaConfig(max_model_len=9000), FakeStats())


@pytest.mark.parametrize(
    ("field", "value"),
    [("input_tokens_mean", 0.0), ("input_tokens_p95", float("nan")), ("output_tokens_p50", -1.0)],
)
def test_bad_stats_rejected(field: str, value: float) -> None:
    stats = FakeStats.model_validate({field: value})
    with pytest.raises(ValidationError, match=f"stats.{field}"):
        estimate(MODEL, GPU, CONFIG, stats, backends={"table": Fake("t", RESULT)})


def test_stats_without_fields_rejected() -> None:
    bad: StatsLike = object()  # type: ignore[assignment]  # deliberately wrong type
    with pytest.raises(ValidationError, match="stats must provide"):
        estimate(MODEL, GPU, CONFIG, bad, backends={"table": Fake("t", RESULT)})


def test_fit_for_uses_replica_settings() -> None:
    config = ReplicaConfig(max_model_len=4096, kv_dtype="fp8", gpu_memory_utilization=0.5)
    result = fit_for(MODEL, GPU, config)
    assert result.kv_bytes_per_token_per_gpu == 65_536
    assert result.per_gpu_usable_bytes == GPU.vram_bytes // 2
