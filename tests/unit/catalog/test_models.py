from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmplan.catalog.models import (
    FixtureFetcher,
    HttpConfigFetcher,
    ModelSpec,
    load_model,
)
from llmplan.errors import CatalogError, FetchError, UnsupportedArchitecture


class DictFetcher:
    """In-memory ConfigFetcher for mapping tests."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    def fetch(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        return self.raw


QWEN3_LIKE: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "max_position_embeddings": 40960,
}


def test_fixture_mapping_defaults() -> None:
    spec = load_model("fixture:qwen2.5-7b")
    assert spec.source == "fixture"
    assert spec.architecture == "llama_like"
    assert spec.head_dim == 3584 // 28
    assert spec.attention_bias is True  # Qwen2 default
    assert spec.qk_norm is False
    assert spec.sliding_window is None
    assert spec.param_count_override is None
    assert spec.attention == "gqa"


def test_mistral_sliding_window() -> None:
    spec = load_model("fixture:mistral-7b-v0.1")
    assert spec.sliding_window == 4096
    assert spec.max_position_embeddings == 32768
    assert spec.attention_bias is False


def test_max_position_embeddings() -> None:
    assert load_model("fixture:llama3-70b").max_position_embeddings == 8192
    assert load_model("fixture:llama3-8b").max_position_embeddings == 8192
    assert load_model("fixture:qwen2.5-7b").max_position_embeddings == 32768


def test_custom_fetcher_qwen3_explicit_head_dim() -> None:
    spec = load_model("Qwen/Qwen3-8B", fetcher=DictFetcher(QWEN3_LIKE))
    assert spec.source == "huggingface"
    assert spec.head_dim == 128
    assert spec.qk_norm is True
    assert spec.attention_bias is False


def test_kv_heads_default_to_attention_heads() -> None:
    raw = {k: v for k, v in QWEN3_LIKE.items() if k != "num_key_value_heads"}
    spec = load_model("org/mha", fetcher=DictFetcher(raw))
    assert spec.num_kv_heads == 32
    assert spec.attention == "mha"


@pytest.mark.parametrize("key", ["hidden_size", "num_hidden_layers", "vocab_size"])
def test_missing_required_key(key: str) -> None:
    raw = {k: v for k, v in QWEN3_LIKE.items() if k != key}
    with pytest.raises(UnsupportedArchitecture) as info:
        load_model("org/x", fetcher=DictFetcher(raw))
    assert info.value.field == key


def test_missing_architectures() -> None:
    raw = {k: v for k, v in QWEN3_LIKE.items() if k != "architectures"}
    with pytest.raises(UnsupportedArchitecture) as info:
        load_model("org/x", fetcher=DictFetcher(raw))
    assert info.value.field == "architectures"


@pytest.mark.parametrize("value", [0, True, "4096", 4096.5])
def test_invalid_value_names_field(value: object) -> None:
    raw = {**QWEN3_LIKE, "hidden_size": value}
    with pytest.raises(CatalogError, match="hidden_size"):
        load_model("org/x", fetcher=DictFetcher(raw))


def test_kv_heads_must_divide_heads() -> None:
    raw = {**QWEN3_LIKE, "num_key_value_heads": 5}
    with pytest.raises(CatalogError, match="num_kv_heads"):
        load_model("org/x", fetcher=DictFetcher(raw))


def test_spec_is_frozen() -> None:
    spec = load_model("fixture:llama3-8b")
    with pytest.raises(Exception, match="frozen"):
        spec.hidden_size = 1  # type: ignore[misc]  # asserting immutability


def test_fixture_fetcher_errors(tmp_path: Path) -> None:
    fetcher = FixtureFetcher(tmp_path)
    with pytest.raises(FetchError, match="not found"):
        fetcher.fetch("fixture:missing")
    with pytest.raises(FetchError, match="invalid fixture id"):
        fetcher.fetch("fixture:../etc/passwd")
    with pytest.raises(FetchError, match="invalid fixture id"):
        fetcher.fetch("llama3-8b")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(FetchError, match="not valid JSON"):
        fetcher.fetch("fixture:bad")
    (tmp_path / "list.json").write_text(json.dumps([1, 2]))
    with pytest.raises(FetchError, match="JSON object"):
        fetcher.fetch("fixture:list")


def _fetcher(handler: Any) -> HttpConfigFetcher:
    return HttpConfigFetcher(transport=httpx.MockTransport(handler))


def test_http_fetch_ok_and_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=QWEN3_LIKE)

    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    raw = _fetcher(handler).fetch("Qwen/Qwen3-8B", "abc123")
    assert raw == QWEN3_LIKE
    assert str(seen[0].url) == "https://huggingface.co/Qwen/Qwen3-8B/resolve/abc123/config.json"
    assert seen[0].headers["Authorization"] == "Bearer hf_test_value"


def test_http_fetch_no_token_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=QWEN3_LIKE)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    _fetcher(handler).fetch("Qwen/Qwen3-8B")
    assert "Authorization" not in seen[0].headers


@pytest.mark.parametrize("status", [401, 403])
def test_http_gated(status: int) -> None:
    with pytest.raises(FetchError, match="HF_TOKEN") as info:
        _fetcher(lambda r: httpx.Response(status)).fetch("meta-llama/Llama-3.1-8B")
    assert str(status) in str(info.value)
    assert "meta-llama/Llama-3.1-8B" in str(info.value)


def test_http_not_found() -> None:
    with pytest.raises(FetchError, match=r"HTTP 404 for org/x"):
        _fetcher(lambda r: httpx.Response(404)).fetch("org/x")


def test_http_non_json() -> None:
    with pytest.raises(FetchError, match="not valid JSON"):
        _fetcher(lambda r: httpx.Response(200, text="<html>")).fetch("org/x")


def test_http_size_cap() -> None:
    big = b"{" + b" " * (2**20 + 10) + b"}"
    with pytest.raises(FetchError, match="1 MiB"):
        _fetcher(lambda r: httpx.Response(200, content=big)).fetch("org/x")


def test_http_redirect_within_hf_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/old/"):
            return httpx.Response(307, headers={"Location": "/new/x/resolve/main/config.json"})
        return httpx.Response(200, json=QWEN3_LIKE)

    assert _fetcher(handler).fetch("old/x") == QWEN3_LIKE


def test_http_redirect_off_hf_is_refused() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://evil.example/config.json"})

    with pytest.raises(FetchError, match=r"evil\.example"):
        _fetcher(handler).fetch("org/x")
    assert hosts == ["huggingface.co"]


def test_http_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(FetchError, match="ConnectError"):
        _fetcher(handler).fetch("org/x")


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [("../..", "main"), ("org/x", "../main"), ("org/x", "refs/pr/1"), ("x", "main")],
)
def test_http_rejects_bad_ids_without_request(repo_id: str, revision: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    with pytest.raises(FetchError):
        _fetcher(handler).fetch(repo_id, revision)
    assert calls == []


def test_manual_spec() -> None:
    spec = ModelSpec(
        id="manual:x",
        architecture="llama_like",
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=1,
        head_dim=16,
        intermediate_size=128,
        vocab_size=100,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        max_position_embeddings=512,
        source="manual",
    )
    assert spec.attention == "mqa"
