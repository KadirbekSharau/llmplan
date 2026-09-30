"""Storage width of each weight and KV-cache dtype (M1_DESIGN.md section 4.2)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from llmplan.types import DType, KVDType

BITS_PER_ELEMENT: Mapping[DType, int] = MappingProxyType(
    {"fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "int4": 4}
)

QUANTIZED_INT: frozenset[DType] = frozenset({"int8", "int4"})


def bytes_per_element(dtype: DType | KVDType) -> float:
    """Bytes per stored element: fp32 4.0, bf16/fp16 2.0, fp8/int8 1.0, int4 0.5."""
    return BITS_PER_ELEMENT[dtype] / 8


def bytes_for(elements: int, dtype: DType | KVDType) -> int:
    """Exact bytes to store `elements` values of `dtype`, rounded up to a whole byte."""
    return -(-elements * BITS_PER_ELEMENT[dtype] // 8)
