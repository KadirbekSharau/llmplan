"""Model catalog: `ModelSpec` built from a Hugging Face config.json or a local fixture.

Fetching sits behind the `ConfigFetcher` protocol. `HttpConfigFetcher` is the only code in
the package that touches the network; tests use `FixtureFetcher` exclusively.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import httpx
import pydantic
from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog import architectures
from llmplan.errors import CatalogError, FetchError, UnsupportedArchitecture
from llmplan.types import Attention, DType

log = logging.getLogger(__name__)

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "model_configs"
FIXTURE_PREFIX = "fixture:"
MAX_CONFIG_BYTES = 1 * 2**20
HF_HOST = "huggingface.co"

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_FIXTURE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ModelSpec(BaseModel):
    """Architectural integers of one dense decoder (ARCHITECTURE.md section 4).

    Strict: config.json values must already have the right JSON type (no `true` -> 1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    qk_norm: bool = False
    max_position_embeddings: int = Field(gt=0)
    sliding_window: int | None = Field(default=None, gt=0)
    param_count_override: int | None = Field(default=None, gt=0)
    source: Literal["huggingface", "fixture", "manual"]

    @pydantic.model_validator(mode="after")
    def _kv_heads_divide_heads(self) -> Self:
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        return self

    @property
    def attention(self) -> Attention:
        if self.num_kv_heads == self.num_attention_heads:
            return "mha"
        return "mqa" if self.num_kv_heads == 1 else "gqa"


class DerivedModelInfo(BaseModel):
    """Values computed from a `ModelSpec` (see `llmplan.memory.weights.model_info`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    param_count: int = Field(gt=0)
    attention: Attention
    weight_bytes_by_dtype: dict[DType, int]


class ConfigFetcher(Protocol):
    def fetch(self, repo_id: str, revision: str = "main") -> dict[str, Any]: ...


def _parse_config(payload: bytes, what: str) -> dict[str, Any]:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise FetchError(f"{what}: config.json is not valid JSON") from None
    if not isinstance(raw, dict):
        raise FetchError(f"{what}: config.json must be a JSON object")
    return raw


class FixtureFetcher:
    """Reads `<dir>/<name>.json` for ids of the form `fixture:<name>`."""

    def __init__(self, dir: Path = DEFAULT_FIXTURE_DIR) -> None:
        self.dir = dir

    def fetch(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        name = repo_id.removeprefix(FIXTURE_PREFIX)
        if not repo_id.startswith(FIXTURE_PREFIX) or not _FIXTURE_NAME_RE.fullmatch(name):
            raise FetchError(f"invalid fixture id {repo_id!r}; expected 'fixture:<name>'")
        path = self.dir / f"{name}.json"
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            raise FetchError(f"fixture {name!r} not found in {self.dir}") from None
        return _parse_config(payload, repo_id)


def _is_hf_host(host: str) -> bool:
    return host == HF_HOST or host.endswith("." + HF_HOST)


def _check_request(request: httpx.Request) -> None:
    """httpx request hook: every request, including redirects, must go to huggingface.co."""
    if request.url.scheme != "https" or not _is_hf_host(request.url.host):
        raise FetchError(f"refusing request to non-huggingface.co host {request.url.host!r}")


class HttpConfigFetcher:
    """Fetches config.json from huggingface.co (the package's only network access)."""

    def __init__(
        self, *, transport: httpx.BaseTransport | None = None, timeout_s: float = 10.0
    ) -> None:
        self._transport = transport
        self._timeout_s = timeout_s

    def fetch(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        if not _REPO_ID_RE.fullmatch(repo_id) or ".." in repo_id:
            raise FetchError(f"invalid Hugging Face repo id {repo_id!r}; expected 'org/name'")
        if not _REVISION_RE.fullmatch(revision) or ".." in revision:
            raise FetchError(f"invalid revision {revision!r}")
        url = f"https://{HF_HOST}/{repo_id}/resolve/{revision}/config.json"
        headers = {"Accept": "application/json"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        log.info("fetching config.json repo=%s revision=%s", repo_id, revision)
        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=self._timeout_s,
                    follow_redirects=True,
                    event_hooks={"request": [_check_request]},
                ) as client,
                client.stream("GET", url, headers=headers) as response,
            ):
                if response.status_code in (401, 403):
                    raise FetchError(
                        f"HTTP {response.status_code} for {repo_id}: the repo may be gated; "
                        "accept its license on huggingface.co and set HF_TOKEN"
                    )
                if response.status_code != 200:
                    raise FetchError(f"HTTP {response.status_code} for {repo_id}")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > MAX_CONFIG_BYTES:
                        raise FetchError(f"config.json for {repo_id} exceeds 1 MiB")
        except httpx.HTTPError as exc:
            raise FetchError(f"fetching {repo_id} failed: {type(exc).__name__}") from None
        return _parse_config(bytes(payload), repo_id)


_REQUIRED = object()

# ModelSpec field -> (HF key, default). `_REQUIRED` raises UnsupportedArchitecture.
_KEY_MAP: dict[str, tuple[str, object]] = {
    "hidden_size": ("hidden_size", _REQUIRED),
    "num_layers": ("num_hidden_layers", _REQUIRED),
    "num_attention_heads": ("num_attention_heads", _REQUIRED),
    "intermediate_size": ("intermediate_size", _REQUIRED),
    "vocab_size": ("vocab_size", _REQUIRED),
    "max_position_embeddings": ("max_position_embeddings", _REQUIRED),
    "tie_word_embeddings": ("tie_word_embeddings", False),
    "mlp_bias": ("mlp_bias", False),
    "sliding_window": ("sliding_window", None),
}


def _spec_from_config(
    model_id: str, raw: dict[str, Any], source: Literal["huggingface", "fixture"]
) -> ModelSpec:
    """Map an HF config.json dict to a validated `ModelSpec` (M1_DESIGN.md section 3.2)."""
    archs = raw.get("architectures")
    if not isinstance(archs, list) or not archs or not isinstance(archs[0], str):
        raise UnsupportedArchitecture(
            f"{model_id}: config.json has no usable 'architectures' list", field="architectures"
        )
    key, defaults = architectures.resolve_hf_class(archs[0])

    fields: dict[str, Any] = {}
    for field, (hf_key, default) in _KEY_MAP.items():
        value = raw.get(hf_key)
        if value is None:
            if default is _REQUIRED:
                raise UnsupportedArchitecture(
                    f"{model_id}: config.json is missing required key {hf_key!r}", field=hf_key
                )
            value = default
        fields[field] = value
    kv_heads = raw.get("num_key_value_heads")
    fields["num_kv_heads"] = fields["num_attention_heads"] if kv_heads is None else kv_heads
    head_dim = raw.get("head_dim")
    if head_dim is None:
        hidden, heads = fields["hidden_size"], fields["num_attention_heads"]
        if isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
            head_dim = hidden // heads  # otherwise ModelSpec validation reports the bad field
    fields["head_dim"] = head_dim
    bias = raw.get("attention_bias")
    fields["attention_bias"] = defaults.attention_bias if bias is None else bias

    try:
        return ModelSpec(
            id=model_id,
            architecture=key,
            qk_norm=defaults.qk_norm,
            source=source,
            **fields,
        )
    except pydantic.ValidationError as exc:
        err = exc.errors()[0]
        loc = ".".join(str(p) for p in err["loc"]) or "config"
        raise CatalogError(f"{model_id}: invalid config field {loc!r}: {err['msg']}") from None


def load_model(id: str, *, fetcher: ConfigFetcher | None = None) -> ModelSpec:
    """Load a `ModelSpec` for an HF repo id (`org/name`) or a fixture id (`fixture:<name>`)."""
    if id.startswith(FIXTURE_PREFIX):
        return _spec_from_config(id, FixtureFetcher().fetch(id), "fixture")
    raw = (fetcher or HttpConfigFetcher()).fetch(id)
    return _spec_from_config(id, raw, "huggingface")
