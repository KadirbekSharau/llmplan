"""Shared Literal aliases used across packages."""

from __future__ import annotations

from typing import Literal

DType = Literal["fp32", "bf16", "fp16", "fp8", "int8", "int4"]
KVDType = Literal["bf16", "fp16", "fp8"]
Attention = Literal["mha", "gqa", "mqa"]
