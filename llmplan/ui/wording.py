"""Words and numbers the web UI shows (M9_DESIGN.md sections 3, 4 and 6).

Currency and durations (milliseconds or seconds, chosen by size), the model summary chip,
the fleet in one sentence, and an error with a "what to change" hint chosen by the
exception's type (never by parsing its message). Everything is read from library results
or models; no Streamlit import, so these are unit-tested without a browser.
"""

from __future__ import annotations

from collections.abc import Mapping

from llmplan.catalog.hardware import GPUSpec
from llmplan.catalog.models import ModelSpec
from llmplan.errors import (
    FetchError,
    InfeasiblePlan,
    LLMPlanError,
    SolverError,
    WorkloadFormatError,
)
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import model_info
from llmplan.planner import PlanResult

HINTS: tuple[tuple[type[LLMPlanError], str], ...] = (  # first match wins
    (InfeasiblePlan, "relax the latency target (TTFT, TPOT, utilization) or add GPUs."),
    (FetchError, "check the Hugging Face id (org/name), or set HF_TOKEN for a gated model."),
    (
        WorkloadFormatError,
        "the trace needs the columns arrival_s (seconds) or timestamp (ISO-8601), input_tokens "
        "and output_tokens (optionally model, tenant); Azure 2023/2024 and BurstGPT exports are "
        "recognised by their own headers.",
    ),
    (SolverError, "raise the solver time limit under Advanced, or select fewer options."),
)


def error_text(error: LLMPlanError) -> str:
    """The library's message, then a "What to change" hint for the error's type (none for
    other types, whose messages already name the field to fix)."""
    hint = next((text for kind, text in HINTS if isinstance(error, kind)), None)
    return str(error) if hint is None else f"{error}\n\nWhat to change: {hint}"


def usd(value: float | None) -> str:
    """Dollars with thousands separators and cents (`$1,234.50`); `n/a` for None."""
    return "n/a" if value is None else f"${value:,.2f}"


def duration(ms: float) -> str:
    """A duration in ms below one second (3 significant digits), else in seconds."""
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1000:,.2f} s"


def model_summary(spec: ModelSpec) -> str:
    """The model chip: parameters (and active parameters for a mixture of experts), the
    attention layout, and KV cache bytes per token at bf16."""
    info = model_info(spec)
    params = f"{info.param_count / 1e9:,.2f}B parameters"
    if info.active_param_count != info.param_count:
        params += f" ({info.active_param_count / 1e9:,.2f}B active)"
    kv_kib = kv_bytes_per_token_total(spec, "bf16") / 1024
    return f"{params} · {info.attention.upper()} · {kv_kib:,.0f} KiB KV cache per token (bf16)"


def fleet_sentence(result: PlanResult, gpus: Mapping[str, GPUSpec]) -> str:
    """The fleet in one sentence, e.g. `2 x H100 SXM 80GB (runpod h100-sxm) serving 2
    replicas`: GPUs per price row with the GPU's catalog name, then the replica count."""
    parts = []
    for item in result.fleet:
        row = item.price_row
        name = gpus[row.gpu_id].name.removeprefix("NVIDIA ")  # price rows name catalog GPUs
        parts.append(f"{item.instances * row.gpu_count} x {name} ({row.provider} {row.instance})")
    replicas = sum(r.count for r in result.replicas)
    return f"{' + '.join(parts)} serving {replicas} replica{'s' if replicas != 1 else ''}"
