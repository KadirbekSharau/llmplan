"""Mixture-of-experts decoders: Mixtral and Qwen2/Qwen3-MoE (M8_DESIGN.md section 6).

Attention is the dense llama-like attention. Each MoE layer replaces the MLP with `E`
SwiGLU experts of width `I_moe`, a router (`H x E`) and, for Qwen2-MoE, an always-on shared
expert with a one-output gate. Layers in Qwen's `mlp_only_layers` (or off its
`decoder_sparse_step`) keep a dense MLP of width `intermediate_size`. Per token, `k` experts
run (`experts_per_token`): every parameter is resident (weights), only the active ones
compute. DeepSeek-style multi-head latent attention is not modeled (`UnsupportedArchitecture`
names MLA).
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING, Any

from llmplan.catalog.architectures import HFClassDefaults, register
from llmplan.catalog.architectures.llama_like import (
    attention_params,
    dense_mlp_params,
    embedding_params,
    norm_params,
)
from llmplan.errors import CatalogError, UnsupportedArchitecture

if TYPE_CHECKING:
    from llmplan.catalog.models import ModelSpec


def _required(model_id: str, raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    if value is None:
        raise UnsupportedArchitecture(
            f"{model_id}: config.json is missing required key {key!r}", field=key
        )
    return value


def _int(model_id: str, key: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CatalogError(f"{model_id}: invalid config field {key!r}: expected an integer")
    return value


def expert_params(spec: ModelSpec) -> int:
    """One routed expert: gate, up and down projections of width `moe_intermediate_size`."""
    return 3 * spec.hidden_size * (spec.moe_intermediate_size or 0)


def _moe_layer(spec: ModelSpec, experts: int) -> int:
    """One MoE layer with `experts` routed experts counted (all of them, or the active k)."""
    h, i_shared = spec.hidden_size, spec.shared_expert_intermediate_size
    router = h * spec.num_experts
    shared = 3 * h * i_shared + h if i_shared else 0  # shared expert and its 1-output gate
    return (
        attention_params(spec) + norm_params(spec) + experts * expert_params(spec) + router + shared
    )


class _MoE:
    """Formulas shared by the MoE families; subclasses map their config keys."""

    def count_params(self, spec: ModelSpec) -> int:
        if spec.param_count_override is not None:
            return spec.param_count_override
        return self._total(spec, spec.num_experts)

    def active_params(self, spec: ModelSpec) -> int:
        if spec.param_count_override is not None:
            return spec.param_count_override  # no expert split known: treated as dense
        return self._total(spec, spec.experts_per_token)

    def expert_params(self, spec: ModelSpec) -> int:
        if spec.param_count_override is not None:
            return 0
        return len(spec.moe_layer_indices) * spec.num_experts * expert_params(spec)

    def embedding_params(self, spec: ModelSpec) -> int:
        return embedding_params(spec)

    def kv_heads_per_gpu(self, spec: ModelSpec, tensor_parallel: int) -> int:
        return ceil(spec.num_kv_heads / tensor_parallel)  # attention is dense, as llama_like

    def _total(self, spec: ModelSpec, experts: int) -> int:
        moe_layers = len(spec.moe_layer_indices)
        dense_layers = spec.num_layers - moe_layers
        dense = attention_params(spec) + norm_params(spec) + dense_mlp_params(spec)
        layers = moe_layers * _moe_layer(spec, experts) + dense_layers * dense
        return layers + embedding_params(spec) + spec.hidden_size  # + final norm


@register("mixtral")
class Mixtral(_MoE):
    """Mixtral: every layer is a MoE layer; experts have width `intermediate_size`."""

    @property
    def hf_classes(self) -> Mapping[str, HFClassDefaults]:
        return {"MixtralForCausalLM": HFClassDefaults()}

    def config_fields(self, model_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        layers = _int(model_id, "num_hidden_layers", _required(model_id, raw, "num_hidden_layers"))
        return {
            "num_experts": _required(model_id, raw, "num_local_experts"),
            "experts_per_token": _required(model_id, raw, "num_experts_per_tok"),
            "moe_intermediate_size": _required(model_id, raw, "intermediate_size"),
            "moe_layer_indices": tuple(range(layers)),
        }


@register("qwen_moe")
class QwenMoE(_MoE):
    """Qwen2-MoE and Qwen3-MoE: MoE on every `decoder_sparse_step`-th layer outside
    `mlp_only_layers`; Qwen2-MoE adds a shared expert (`shared_expert_intermediate_size`)."""

    @property
    def hf_classes(self) -> Mapping[str, HFClassDefaults]:
        return {
            "Qwen3MoeForCausalLM": HFClassDefaults(qk_norm=True),
            "Qwen2MoeForCausalLM": HFClassDefaults(attention_bias=True),
        }

    def config_fields(self, model_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        layers = _int(model_id, "num_hidden_layers", _required(model_id, raw, "num_hidden_layers"))
        experts = _int(model_id, "num_experts", _required(model_id, raw, "num_experts"))
        step = _int(model_id, "decoder_sparse_step", raw.get("decoder_sparse_step", 1))
        dense_only = raw.get("mlp_only_layers") or []
        if not isinstance(dense_only, list):
            raise CatalogError(
                f"{model_id}: invalid config field 'mlp_only_layers': expected a list"
            )
        dense = {_int(model_id, "mlp_only_layers", i) for i in dense_only}
        if step < 1:
            raise CatalogError(
                f"{model_id}: invalid config field 'decoder_sparse_step': must be >= 1"
            )
        qwen2 = raw.get("architectures", [None])[0] == "Qwen2MoeForCausalLM"
        shared = (raw.get("shared_expert_intermediate_size") or 0) if qwen2 else 0  # Qwen2 only
        return {
            "num_experts": experts,
            "experts_per_token": _required(model_id, raw, "num_experts_per_tok"),
            "moe_intermediate_size": _required(model_id, raw, "moe_intermediate_size"),
            "shared_expert_intermediate_size": shared,
            "moe_layer_indices": tuple(
                i for i in range(layers) if i not in dense and experts > 0 and (i + 1) % step == 0
            ),
        }
