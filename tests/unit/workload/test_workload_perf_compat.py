"""M2 -> M3 hand-off: `WorkloadStats` satisfies M3's `StatsLike` protocol."""

from __future__ import annotations

from pathlib import Path

from llmplan.catalog.hardware import load_gpus
from llmplan.catalog.models import load_model
from llmplan.perf import ReplicaConfig, StatsLike, estimate
from llmplan.workload import compute_stats, load_workload

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_workload_stats_feed_the_performance_model() -> None:
    stats = compute_stats(load_workload(FIXTURES / "workload_azure2023_50.csv"))
    assert isinstance(stats, StatsLike)
    result = estimate(
        load_model("fixture:llama3-8b"),
        load_gpus()["h100-sxm-80gb"],
        ReplicaConfig(max_model_len=8192),
        stats,
        backend="roofline",
    )
    assert result.backend == "roofline"
    assert result.decode_tokens_per_s > 0
