"""Architecture registry: per-family parameter and KV-head formulas.

Each member maps a set of Hugging Face `architectures[0]` class names to one formula
implementation. Adding a family is a new module plus an import line at the bottom of this
file. Members: `llama_like` (M1, dense), `mixtral` and `qwen_moe` (M8, mixture of experts).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict

from llmplan.errors import UnknownRegistryKey, UnsupportedArchitecture

if TYPE_CHECKING:
    from llmplan.catalog.models import ModelSpec


class HFClassDefaults(BaseModel):
    """Per-HF-class values used when config.json omits them (M1_DESIGN.md section 3.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attention_bias: bool = False  # default when the config has no `attention_bias` key
    qk_norm: bool = False  # never read from config; fixed by the class


class Architecture(Protocol):
    """Interface every architecture module implements. M8 adds `config_fields` (the
    family's own `ModelSpec` fields read from config.json), `active_params` (parameters
    that compute per token) and `expert_params` (routed-expert parameters)."""

    @property
    def hf_classes(self) -> Mapping[str, HFClassDefaults]: ...

    def config_fields(self, model_id: str, raw: Mapping[str, Any]) -> dict[str, Any]: ...

    def count_params(self, spec: ModelSpec) -> int: ...

    def active_params(self, spec: ModelSpec) -> int: ...

    def expert_params(self, spec: ModelSpec) -> int: ...

    def embedding_params(self, spec: ModelSpec) -> int: ...

    def kv_heads_per_gpu(self, spec: ModelSpec, tensor_parallel: int) -> int: ...


_REGISTRY: dict[str, Architecture] = {}

# HF classes recognised but not modeled, with the reason (M8_DESIGN.md section 6).
_NOT_MODELED = {
    "DeepseekV2ForCausalLM": "multi-head latent attention (MLA) is not modeled",
    "DeepseekV3ForCausalLM": "multi-head latent attention (MLA) is not modeled",
}


def register(key: str) -> Callable[[type[Architecture]], type[Architecture]]:
    """Class decorator: instantiate the class and register it under `key`."""

    def decorator(cls: type[Architecture]) -> type[Architecture]:
        _REGISTRY[key] = cls()
        return cls

    return decorator


def get(key: str) -> Architecture:
    """Return the architecture registered under `key`."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRegistryKey(f"unknown architecture {key!r}; known: {known}") from None


def resolve_hf_class(hf_class: str) -> tuple[str, HFClassDefaults]:
    """Map an HF `architectures[0]` value to (registry key, class defaults). Raises
    `UnsupportedArchitecture` naming the reason for a known unmodeled class (MLA)."""
    if hf_class in _NOT_MODELED:
        raise UnsupportedArchitecture(
            f"unsupported architectures[0] {hf_class!r}: {_NOT_MODELED[hf_class]}",
            field="architectures",
        )
    for key, arch in _REGISTRY.items():
        defaults = arch.hf_classes.get(hf_class)
        if defaults is not None:
            return key, defaults
    raise UnsupportedArchitecture(
        f"unsupported architectures[0] {hf_class!r}; supported: "
        + ", ".join(sorted(c for a in _REGISTRY.values() for c in a.hf_classes)),
        field="architectures",
    )


from llmplan.catalog.architectures import llama_like, moe  # noqa: E402

__all__ = [
    "Architecture",
    "HFClassDefaults",
    "get",
    "llama_like",
    "moe",
    "register",
    "resolve_hf_class",
]
