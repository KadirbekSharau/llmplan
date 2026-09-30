"""Architecture registry: per-family parameter and KV-head formulas.

Each member maps a set of Hugging Face `architectures[0]` class names to one formula
implementation. Adding a family is a new module plus an import line at the bottom of this
file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

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
    """Interface every architecture module implements."""

    @property
    def hf_classes(self) -> Mapping[str, HFClassDefaults]: ...

    def count_params(self, spec: ModelSpec) -> int: ...

    def embedding_params(self, spec: ModelSpec) -> int: ...

    def kv_heads_per_gpu(self, spec: ModelSpec, tensor_parallel: int) -> int: ...


_REGISTRY: dict[str, Architecture] = {}


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
    """Map an HF `architectures[0]` value to (registry key, class defaults)."""
    for key, arch in _REGISTRY.items():
        defaults = arch.hf_classes.get(hf_class)
        if defaults is not None:
            return key, defaults
    raise UnsupportedArchitecture(
        f"unsupported architectures[0] {hf_class!r}; supported: "
        + ", ".join(sorted(c for a in _REGISTRY.values() for c in a.hf_classes)),
        field="architectures",
    )


from llmplan.catalog.architectures import llama_like  # noqa: E402

__all__ = ["Architecture", "HFClassDefaults", "get", "llama_like", "register", "resolve_hf_class"]
