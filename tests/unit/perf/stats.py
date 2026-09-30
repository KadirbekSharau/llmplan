"""Test-only stand-in for M2's `WorkloadStats` (only the fields M3 reads)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FakeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens_mean: float = 512
    input_tokens_p50: float = 400
    input_tokens_p95: float = 1500
    output_tokens_mean: float = 256
    output_tokens_p50: float = 200
    output_tokens_p95: float = 800
