from __future__ import annotations

import dataclasses

import pydantic
import pytest

from llmplan.planner.candidates import (
    columns,
    evaluate_candidates,
    prune_dominated,
    rows_in_scope,
)
from llmplan.planner.request import SLO, PlanOptions, PlanRequest
from llmplan.planner.result import CandidateEval, infeasible_reason, label
from tests.conftest import UseFakePerf
from tests.fake_planner import GPUS, ROW_A, ROW_B, FakePerf, fake_gpu, fake_row, request


def _evaluate(req: PlanRequest) -> tuple[CandidateEval, ...]:
    rows = rows_in_scope(req.prices, req.options)
    return evaluate_candidates(req, rows, req.gpus)


def test_plan_options_reject_duplicates_and_bad_values() -> None:
    with pytest.raises(pydantic.ValidationError, match="tensor_parallel_choices"):
        PlanOptions(max_model_len=8192, tensor_parallel_choices=(1, 1))
    with pytest.raises(pydantic.ValidationError):
        PlanOptions(max_model_len=8192, max_num_seqs_choices=(0,))
    with pytest.raises(pydantic.ValidationError):
        PlanOptions(max_model_len=8192, dtype_choices=())
    with pytest.raises(pydantic.ValidationError):
        SLO(utilization_target=1.5)


def test_rows_in_scope_filters() -> None:
    spot = ROW_A.model_copy(update={"commitment": "spot", "instance": "a-spot"})
    other = ROW_B.model_copy(update={"provider": "other"})
    prices = (ROW_A, spot, other)
    assert rows_in_scope(prices, PlanOptions(max_model_len=1)) == (ROW_A, other)
    assert rows_in_scope(prices, PlanOptions(max_model_len=1, providers=("other",))) == (other,)
    assert rows_in_scope(prices, PlanOptions(max_model_len=1, gpu_ids=("fake-a",))) == (ROW_A,)
    both = PlanOptions(max_model_len=1, commitments=("spot", "on_demand"))
    assert rows_in_scope(prices, both) == prices


def test_statuses_and_reasons(fake_perf: UseFakePerf) -> None:
    fake_perf(
        {
            ("fake-a", 1): FakePerf(rps=3, ttft_ms_p95=800),
            ("fake-b", 1): FakePerf(rps=4, tpot_ms_p95=90),
            ("fake-b", 2): FakePerf(rps=9),
        }
    )
    six = fake_row("c-6x", "fake-b", 6, 10.0)
    three = fake_row("d-3x", "fake-b", 3, 5.0)
    req = request(
        10,
        (ROW_A, ROW_B, six, three),
        slo=SLO(ttft_ms_p95=500, tpot_ms_p95=50, utilization_target=0.5),
        tensor_parallel_choices=(1, 2, 3, 4),
    )
    by_label = {label(c): c for c in _evaluate(req)}
    assert by_label["test a-1x tp1 bf16 seqs64"].status == "slo_ttft"
    assert "800" in by_label["test a-1x tp1 bf16 seqs64"].reason
    assert by_label["test a-1x tp2 bf16 seqs64"].status == "tp_gt_gpus"
    assert by_label["test a-1x tp2 bf16 seqs64"].fit is None
    assert by_label["test b-8x tp1 bf16 seqs64"].status == "slo_tpot"
    assert "TPOT p95 90 ms > SLO 50 ms" in by_label["test b-8x tp1 bf16 seqs64"].reason
    eligible = by_label["test b-8x tp2 bf16 seqs64"]
    assert eligible.status == "eligible"
    assert eligible.replicas_per_instance == 4
    assert eligible.usd_per_hour_per_rps == pytest.approx(19 / (4 * 9 * 0.5))
    assert by_label["test b-8x tp4 bf16 seqs64"].status == "no_perf"
    assert "no fake capacity" in by_label["test b-8x tp4 bf16 seqs64"].reason
    assert by_label["test c-6x tp4 bf16 seqs64"].status == "tp_gt_gpus"
    assert "does not divide gpu_count 6" in by_label["test c-6x tp4 bf16 seqs64"].reason
    heads = by_label["test d-3x tp3 bf16 seqs64"]
    assert (heads.status, heads.fit) == ("no_fit", None)
    assert "num_attention_heads 32" in heads.reason


def test_memory_no_fit_reports_binding() -> None:
    small = fake_gpu("tiny").model_copy(update={"vram_bytes": 8 * 10**9})
    req = request(1, (fake_row("t-1x", "tiny", 1, 1.0),)).model_copy(
        update={"gpus": {**GPUS, "tiny": small}}
    )
    (candidate,) = _evaluate(req)
    assert candidate.status == "no_fit"
    assert candidate.fit is not None
    assert not candidate.fit.fits
    assert "binding: weights" in candidate.reason


def test_infeasible_reason_counts(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-a", 1): FakePerf(rps=3, ttft_ms_p95=800)})
    req = request(1, slo=SLO(ttft_ms_p95=500, utilization_target=1.0))
    assert infeasible_reason(_evaluate(req)) == (
        "0 of 2 candidates eligible: 1 slo_ttft, 1 no_perf"
    )


def test_prune_keeps_undominated_and_first_of_ties(fake_perf: UseFakePerf) -> None:
    fake_perf({("fake-b", 1): FakePerf(rps=4), ("fake-a", 1): FakePerf(rps=3)})
    req = request(1, max_num_seqs_choices=(32, 64, 128))
    cols = columns(_evaluate(req), req.slo)
    # Same perf per (gpu, tp); a larger max_num_seqs reserves more KV, so seqs32 dominates.
    kept = prune_dominated(cols)
    assert [label(c.candidate) for c in kept] == [
        "test a-1x tp1 bf16 seqs32",
        "test b-8x tp1 bf16 seqs32",
    ]
    a32, a64, a128 = cols[:3]
    faster = dataclasses.replace(a128, rps=5.0)
    more_tokens = dataclasses.replace(a64, tps=a64.tps * 2)
    assert prune_dominated([a32, more_tokens, faster]) == (a32, more_tokens, faster)
    tie = dataclasses.replace(a32)
    assert prune_dominated([a32, tie]) == (a32,)
