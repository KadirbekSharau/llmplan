"""Mélange cross-check (M7_DESIGN.md section 6.2): same inputs, two independent solvers.

Usage:

    uv run python scripts/melange_crosscheck.py [--melange-dir DIR] [--rate RPS]
    uv run --with pulp==2.8.0 python scripts/melange_crosscheck.py --melange-dir DIR

Builds four scenarios in Mélange's input format: its toy example (melange-release
`melange/config/example.json`), and three synthetic ones from the shipped catalogs and the
perf model (short-chat heavy, long-document heavy, mixed). Each is solved with llmplan's
class MILP and, when `--melange-dir` points at a checkout of
https://github.com/tyler-griggs/melange-release (its solver needs `pulp`), with Mélange's
solver at slice factors 1, 4 and 16. Prints one row per scenario in USD per hour.

Mapping: a Mélange bucket is an llmplan class (demand = share x rate, no token demand); a
GPU is a one-GPU price row at Mélange's hourly cost; a bucket throughput is the class
capacity (already derated). A throughput of 0 marks a bucket the GPU cannot serve within
the SLO: llmplan makes it ineligible, Mélange gets 1e-9 req/s (an unaffordable load).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from llmplan.catalog.hardware import GPUSpec, PriceRow, load_gpus, load_prices
from llmplan.catalog.models import ModelSpec, load_model
from llmplan.errors import PerfError
from llmplan.memory.engine import EngineProfile
from llmplan.perf import PerfEstimate, ReplicaConfig, StatsLike, estimate, register
from llmplan.planner import SLO, PlanOptions, PlanRequest, plan
from llmplan.workload import WorkloadStats
from llmplan.workload.classes import DemandClass

Scenario = dict[str, Any]  # Mélange's input: gpu_info, workload_distribution, rate, ...
SOURCE = "https://github.com/tyler-griggs/melange-release/blob/main/melange/config/example.json"
TOY: Scenario = {
    "gpu_info": {
        "A10G": {"cost": 1.01, "tputs": [[2, 1], [5, 2]]},
        "A100-80GB": {"cost": 3.67, "tputs": [[20, 20], [40, 20]]},
    },
    "workload_distribution": [[0.2, 0.1], [0.5, 0.2]],
    "total_request_rate": 30.0,
}
# Buckets on a 2 x 2 grid (inputs <= 500 | > 500 tokens, outputs <= 500 | > 500), each at a
# request shape the NIM benchmark rows cover (llmplan/data/benchmarks/): chat 200/200, generation
# 500/2,000, document 5,000/500, balanced 1,000/1,000 (input/output tokens).
SHAPES = ((200, 200), (500, 2000), (5000, 500), (1000, 1000))
MIXES = {
    "short-chat heavy": [[0.70, 0.10], [0.05, 0.15]],
    "long-document heavy": [[0.10, 0.05], [0.70, 0.15]],
    "mixed": [[0.25, 0.25], [0.25, 0.25]],
}
SYNTHETIC_ROWS = ("g6.xlarge", "g6e.xlarge", "h100-sxm")  # one-GPU rows: L4, L40S, H100
SLO_USED = SLO(ttft_ms_p95=500.0, tpot_ms_p95=100.0, utilization_target=0.8)
SEQS = (32, 64, 128, 256)
INELIGIBLE_TPUT = 1e-9
TABLE: dict[str, list[float]] = {}  # per GPU name: capacity per bucket (0: not eligible)


class Shape:
    """A bucket's token statistics as a perf `StatsLike` (every percentile at the shape)."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens_mean = self.input_tokens_p50 = self.input_tokens_p95 = input_tokens
        self.output_tokens_mean = self.output_tokens_p50 = self.output_tokens_p95 = output_tokens


@register("melange_table")
class TableBackend:
    """Returns the scenario's bucket throughput as the class capacity (llmplan side)."""

    name = "melange_table"

    def estimate(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> PerfEstimate | None:
        caps = TABLE[gpu.id]
        index = getattr(stats, "index", None)
        cap = max(caps) if index is None else caps[index]
        return PerfEstimate(
            backend="table",
            confidence="measured",
            effective_batch=1,
            decode_tokens_per_s=1e6,
            prefill_tokens_per_s=1e6,
            requests_per_s_capacity=cap if cap > 0 else INELIGIBLE_TPUT,
            ttft_ms_p50=1.0,
            ttft_ms_p95=1.0 if cap > 0 else 1e9,
            tpot_ms_p50=1.0,
            tpot_ms_p95=1.0,
            assumptions=("cross-check table",),
            source_urls=(),
        )

    def explain(
        self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike
    ) -> str:
        return "always answers"


def capacity(model: ModelSpec, gpu: GPUSpec, shape: tuple[int, int]) -> float:
    """Best derated req/s of one fp8 tp1 replica on `gpu` at `shape` within `SLO_USED`
    over the max_num_seqs choices (0.0 when none meets the SLO)."""
    best = 0.0
    stats = Shape(*shape)
    for seqs in SEQS:
        config = ReplicaConfig(dtype="fp8", max_num_seqs=seqs, max_model_len=8192)
        try:
            perf = estimate(model, gpu, config, stats)
        except PerfError:
            continue
        ttft_ok = SLO_USED.ttft_ms_p95 is None or perf.ttft_ms_p95 <= SLO_USED.ttft_ms_p95
        tpot_ok = SLO_USED.tpot_ms_p95 is None or perf.tpot_ms_p95 <= SLO_USED.tpot_ms_p95
        if ttft_ok and tpot_ok:
            best = max(best, perf.requests_per_s_capacity * SLO_USED.utilization_target)
    return best


def synthetic(rate_rps: float) -> dict[str, Scenario]:
    """The three synthetic scenarios: shipped one-GPU rows, llama3-8b, the perf model."""
    model = load_model("fixture:llama3-8b")
    gpus = load_gpus()
    rows = {r.instance: r for r in load_prices(gpus=gpus) if r.instance in SYNTHETIC_ROWS}
    gpu_info = {}
    for instance in SYNTHETIC_ROWS:
        row = rows[instance]
        caps = [capacity(model, gpus[row.gpu_id], shape) for shape in SHAPES]
        gpu_info[f"{row.gpu_id} ({row.provider})"] = {
            "cost": row.price_usd_per_hour,
            "tputs": [caps[:2], caps[2:]],
        }
    return {
        name: {"gpu_info": gpu_info, "workload_distribution": mix, "total_request_rate": rate_rps}
        for name, mix in MIXES.items()
    }


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")


def _gpu(name: str) -> GPUSpec:
    return GPUSpec(
        id=_slug(name),
        vendor="nvidia",
        name=name,
        vram_bytes=10**13,  # fit never binds: capacities come from the scenario
        memory_bandwidth_gbps=None,
        fp16_dense_tflops=None,
        fp8_dense_tflops=None,
        nvlink=False,
        source_url=SOURCE,
        as_of=date(2026, 10, 1),
    )


def _classes(distribution: Sequence[Sequence[float]], rate: float) -> tuple[DemandClass, ...]:
    shares = [s for row in distribution for s in row]
    return tuple(
        DemandClass(
            index=k,
            input_lo=1000 * k + 1,  # bounds are irrelevant: capacities come from the table
            input_hi=1000 * (k + 1),
            output_lo=0,
            output_hi=1,
            share=share / sum(shares),
            peak_rps=share * rate,
            peak_output_tokens_per_s=0.0,
            input_tokens_mean=1.0,
            output_tokens_mean=1.0,
            input_tokens_p50=1.0,
            output_tokens_p50=1.0,
            input_tokens_p95=1.0,
            output_tokens_p95=1.0,
        )
        for k, share in enumerate(shares)
    )


def llmplan_solve(scenario: Scenario) -> tuple[float, dict[str, int]]:
    """llmplan's cheapest fleet for a Mélange input: USD per hour and GPUs per type."""
    rate = float(scenario["total_request_rate"])
    gpu_info: Mapping[str, Any] = scenario["gpu_info"]
    TABLE.clear()
    TABLE.update(
        {
            _slug(name): [float(t) for row in info["tputs"] for t in row]
            for name, info in gpu_info.items()
        }
    )
    rows = tuple(
        PriceRow(
            provider="melange",
            instance=name,
            gpu_id=_slug(name),
            gpu_count=1,
            price_usd_per_hour=float(info["cost"]),
            commitment="on_demand",
            region=None,
            source_url=SOURCE,
            as_of=date(2026, 10, 1),
        )
        for name, info in gpu_info.items()
    )
    stats = WorkloadStats(
        n_requests=1,
        duration_s=0.0,
        window_s=60.0,
        n_windows=1,
        mean_rps=0.0,
        peak_window_rps=rate,
        peak_window_index=0,
        input_tokens_p50=1.0,
        input_tokens_p95=1.0,
        input_tokens_p99=1.0,
        input_tokens_mean=1.0,
        input_tokens_max=1,
        output_tokens_p50=1.0,
        output_tokens_p95=1.0,
        output_tokens_p99=1.0,
        output_tokens_mean=1.0,
        output_tokens_max=1,
        peak_input_tokens_per_s=0.0,
        peak_output_tokens_per_s=0.0,
    )
    request = PlanRequest(
        model=load_model("fixture:llama3-8b"),
        stats=stats,
        slo=SLO(ttft_ms_p95=1000.0, utilization_target=1.0),
        engine=EngineProfile(),
        options=PlanOptions(
            tensor_parallel_choices=(1,),
            dtype_choices=("bf16",),
            max_num_seqs_choices=(64,),
            max_model_len=8192,
            perf_backend="melange_table",
        ),
        gpus={_slug(name): _gpu(name) for name in gpu_info},
        prices=rows,
        classes=_classes(scenario["workload_distribution"], rate),
    )
    result = plan(request)
    fleet = {item.price_row.instance: item.instances for item in result.fleet}
    return result.cost_usd_per_day / 24, fleet


def melange_solve(scenario: Scenario, melange_dir: Path, slice_factor: int) -> Any:
    """Mélange's answer (its solver's dict: GPUs per type and "cost" per hour, or None)."""
    if str(melange_dir) not in sys.path:
        sys.path.insert(0, str(melange_dir))
    from melange.solver import MelangeSolver  # optional: a checkout outside this repo

    gpu_info = {
        name: {
            "cost": info["cost"],
            "tputs": [[t if t > 0 else INELIGIBLE_TPUT for t in row] for row in info["tputs"]],
        }
        for name, info in scenario["gpu_info"].items()
    }
    solver = MelangeSolver(
        workload_distribution=scenario["workload_distribution"],
        total_request_rate=scenario["total_request_rate"],
        gpu_info=gpu_info,
        slice_factor=slice_factor,
    )
    return solver.run()


def main(argv: Sequence[str] | None = None) -> None:
    """Print each scenario's inputs and both solvers' answers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--melange-dir", type=Path, default=None)
    parser.add_argument("--rate", type=float, default=20.0, help="synthetic request rate")
    args = parser.parse_args(argv)
    scenarios = {"toy (example.json)": TOY, **synthetic(args.rate)}
    for name, scenario in scenarios.items():
        print(f"## {name}")
        for gpu, info in scenario["gpu_info"].items():
            print(f"  {gpu}: ${info['cost']}/h, tputs {info['tputs']}")
        mix, rate = scenario["workload_distribution"], scenario["total_request_rate"]
        print(f"  distribution {mix}, rate {rate}")
        cost, fleet = llmplan_solve(scenario)
        print(f"  llmplan: ${cost:.4f}/h {fleet}")
        if args.melange_dir is not None:
            for factor in (1, 4, 16):
                print(
                    f"  melange slice {factor}: {melange_solve(scenario, args.melange_dir, factor)}"
                )


if __name__ == "__main__":
    main()
