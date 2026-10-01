"""How much to trust a performance estimate (M8_DESIGN.md section 3).

One sentence per confidence level, shared by `llmplan plan`, `llmplan perf estimate` and the
web UI banner: what the estimate is based on (the benchmark page and its date, the
visitor's upload, or the uncalibrated roofline model) and how far off it may be.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from llmplan.perf.benchmarks import USER_UPLOAD, default_table
from llmplan.perf.roofline import BANDWIDTH_EFFICIENCY, DECODE_MFU, P95_FACTOR, PREFILL_MFU

PlanConfidence = Literal["measured", "interpolated", "roofline", "mixed"]

ROOFLINE_CAVEAT = "uncalibrated first-principles model; expect ±30% on throughput"
SOURCE_LABELS = {  # URL prefix -> human name of the benchmark source
    "https://docs.nvidia.com/nim/": "NVIDIA NIM performance page",
}
ROOFLINE_CONSTANTS: tuple[tuple[str, float, str], ...] = (  # name, value, meaning
    ("BANDWIDTH_EFFICIENCY", BANDWIDTH_EFFICIENCY, "share of datasheet HBM bandwidth in decode"),
    ("PREFILL_MFU", PREFILL_MFU, "model FLOPs utilization during prefill"),
    ("DECODE_MFU", DECODE_MFU, "model FLOPs utilization during decode"),
    ("P95_FACTOR", P95_FACTOR, "p95 TPOT = p50 x this factor"),
)
_LEVELS: dict[str, PlanConfidence] = {
    "measured": "measured",
    "interpolated": "interpolated",
    "roofline": "roofline",
}


def combined_confidence(confidences: Iterable[str]) -> PlanConfidence:
    """The confidence of several estimates: their common value, or `"mixed"` when they
    differ. No estimate at all is treated as `"roofline"` (nothing measured backs it)."""
    distinct = set(confidences)
    if len(distinct) > 1:
        return "mixed"
    (only,) = distinct or {"roofline"}
    return _LEVELS[only]


def source_label(url: str) -> str:
    """A benchmark source in words: `"your upload"` (M8 section 5), a known page's name
    with its `as_of` date from the shipped table, or the URL itself."""
    if url == USER_UPLOAD:
        return "your upload"
    name = next((label for prefix, label in SOURCE_LABELS.items() if url.startswith(prefix)), url)
    dates = [row.as_of for row in default_table().rows if row.source_url == url]
    return f"{name}, as of {max(dates).isoformat()}" if dates else name


def confidence_sentence(confidence: PlanConfidence, source_urls: Sequence[str]) -> str:
    """The one-line confidence statement, e.g. `"roofline (uncalibrated first-principles
    model; expect ±30% on throughput)"` or `"measured (NVIDIA NIM performance page, as of
    2026-09-30)"`. `source_urls` are the benchmark sources behind the estimate(s)."""
    sources = "; ".join(source_label(url) for url in sorted(set(source_urls)))
    if confidence == "roofline":
        return f"roofline ({ROOFLINE_CAVEAT})"
    if confidence == "mixed":
        tail = f"; measured rows: {sources}" if sources else ""
        return (
            "mixed (some replicas use measured rows, others the roofline model, which is "
            f"{ROOFLINE_CAVEAT}{tail})"
        )
    if confidence == "interpolated":
        return f"interpolated between measured rows ({sources or 'no source recorded'})"
    return f"measured ({sources or 'no source recorded'})"
