"""Words and numbers the web UI shows (M9_DESIGN.md sections 3, 4 and 6).

Currency and durations (milliseconds or seconds, chosen by size), and the model summary
chip. Everything is read from library results or models; no Streamlit import, so these
are unit-tested without a browser.
"""

from __future__ import annotations

from llmplan.catalog.models import ModelSpec
from llmplan.memory.kv_cache import kv_bytes_per_token_total
from llmplan.memory.weights import model_info


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
