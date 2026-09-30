"""Output renderers, one per `--format` value (ARCHITECTURE.md section 6).

A renderer turns library results into a string; it computes nothing that the library does
not already expose. Adding a format is a new module plus an import line below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from llmplan.errors import UnknownRegistryKey

if TYPE_CHECKING:
    from llmplan.catalog.hardware import GPUSpec
    from llmplan.catalog.models import ModelSpec
    from llmplan.memory.fit import FitRequest, FitResult


class Renderer(Protocol):
    """Formats the results of `llmplan fit`, `llmplan model-info`, and `llmplan gpus`."""

    def fit(self, request: FitRequest, result: FitResult) -> str: ...

    def model_info(self, spec: ModelSpec) -> str: ...

    def gpus(self, gpus: Mapping[str, GPUSpec]) -> str: ...


_REGISTRY: dict[str, Renderer] = {}


def register(key: str) -> Callable[[type[Renderer]], type[Renderer]]:
    """Class decorator: instantiate the class and register it under `key`."""

    def decorator(cls: type[Renderer]) -> type[Renderer]:
        _REGISTRY[key] = cls()
        return cls

    return decorator


def get(key: str) -> Renderer:
    """Return the renderer registered under `key`."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRegistryKey(f"unknown output format {key!r}; known: {known}") from None


from llmplan.render import json_render, text  # noqa: E402

__all__ = ["Renderer", "get", "json_render", "register", "text"]
