from __future__ import annotations

import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedQKVLinear(nn.Module):
    """Fused QKV projection. Single matmul for q, k, v."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.q_size = q_proj.out_features
        self.k_size = k_proj.out_features
        self.v_size = v_proj.out_features

        fused_weight = torch.cat(
            [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data], dim=0
        )
        self.weight = nn.Parameter(fused_weight)

        if q_proj.bias is not None:
            fused_bias = torch.cat(
                [q_proj.bias.data, k_proj.bias.data, v_proj.bias.data], dim=0
            )
            self.bias = nn.Parameter(fused_bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = F.linear(x, self.weight, self.bias)
        return qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)


class FusedGateUpLinear(nn.Module):
    """Fused gate+up projection. Single matmul for gate and up."""

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear):
        super().__init__()
        self.gate_size = gate_proj.out_features

        fused_weight = torch.cat(
            [gate_proj.weight.data, up_proj.weight.data], dim=0
        )
        self.weight = nn.Parameter(fused_weight)
        self.bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.linear(x, self.weight)
        return out.split([self.gate_size, self.gate_size], dim=-1)


def fuse_model_projections(model: nn.Module) -> nn.Module:
    """Replace separate q/k/v and gate/up projections with fused versions in-place.

    Supports Qwen2, Qwen3, Llama, and Gemma3 architectures. Returns the same model.
    """
    for layer in model.model.layers:
        _fuse_attention(layer.self_attn)
        _fuse_mlp(layer.mlp)
    return model


def unfuse_state_dict(model: nn.Module) -> dict:
    """Convert fused model state_dict back to HF-compatible format.

    Splits qkv_proj → q_proj, k_proj, v_proj and gate_up_proj → gate_proj, up_proj.
    """
    state_dict = {}
    for name, param in model.state_dict().items():
        if ".qkv_proj.weight" in name:
            prefix = name.replace("qkv_proj.weight", "")
            attn = _get_module_by_path(model, prefix.rstrip("."))
            q, k, v = param.split(
                [attn.qkv_proj.q_size, attn.qkv_proj.k_size, attn.qkv_proj.v_size], dim=0
            )
            state_dict[prefix + "q_proj.weight"] = q
            state_dict[prefix + "k_proj.weight"] = k
            state_dict[prefix + "v_proj.weight"] = v
        elif ".qkv_proj.bias" in name:
            prefix = name.replace("qkv_proj.bias", "")
            attn = _get_module_by_path(model, prefix.rstrip("."))
            q, k, v = param.split(
                [attn.qkv_proj.q_size, attn.qkv_proj.k_size, attn.qkv_proj.v_size], dim=0
            )
            state_dict[prefix + "q_proj.bias"] = q
            state_dict[prefix + "k_proj.bias"] = k
            state_dict[prefix + "v_proj.bias"] = v
        elif ".gate_up_proj.weight" in name:
            prefix = name.replace("gate_up_proj.weight", "")
            mlp = _get_module_by_path(model, prefix.rstrip("."))
            gate, up = param.split([mlp.gate_up_proj.gate_size, mlp.gate_up_proj.gate_size], dim=0)
            state_dict[prefix + "gate_proj.weight"] = gate
            state_dict[prefix + "up_proj.weight"] = up
        else:
            state_dict[name] = param
    return state_dict


def _fuse_attention(attn: nn.Module) -> None:
    """Replace q_proj, k_proj, v_proj with fused qkv_proj and patch forward."""
    fused = FusedQKVLinear(attn.q_proj, attn.k_proj, attn.v_proj)
    attn.qkv_proj = fused
    del attn.q_proj, attn.k_proj, attn.v_proj

    model_type = getattr(getattr(attn, "config", None), "model_type", None)
    has_qk_norm = hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
    if model_type == "gemma3_text":
        attn.forward = types.MethodType(_gemma3_attention_forward, attn)
    elif has_qk_norm:
        attn.forward = types.MethodType(_qwen3_attention_forward, attn)
    elif model_type == "llama":
        attn.forward = types.MethodType(_llama_attention_forward, attn)
    else:
        attn.forward = types.MethodType(_qwen2_attention_forward, attn)


def _fuse_mlp(mlp: nn.Module) -> None:
    """Replace gate_proj, up_proj with fused gate_up_proj and patch forward."""
    fused = FusedGateUpLinear(mlp.gate_proj, mlp.up_proj)
    mlp.gate_up_proj = fused
    del mlp.gate_proj, mlp.up_proj

    mlp.forward = types.MethodType(_fused_mlp_forward, mlp)


def _fused_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.gate_up_proj(x)
    return self.down_proj(self.act_fn(gate) * up)


def _qwen2_attention_forward(self, hidden_states, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values") or kwargs.get("past_key_value")
    cache_position = kwargs.get("cache_position")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q, k, v = self.qkv_proj(hidden_states)
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _qwen3_attention_forward(self, hidden_states, **kwargs):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values") or kwargs.get("past_key_value")
    cache_position = kwargs.get("cache_position")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q, k, v = self.qkv_proj(hidden_states)
    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _llama_attention_forward(self, hidden_states, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values") or kwargs.get("past_key_value")
    cache_position = kwargs.get("cache_position")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q, k, v = self.qkv_proj(hidden_states)
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _gemma3_attention_forward(self, hidden_states, **kwargs):
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values") or kwargs.get("past_key_value")
    cache_position = kwargs.get("cache_position")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q, k, v = self.qkv_proj(hidden_states)
    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Navigate to a submodule by dot-separated path."""
    parts = path.split(".")
    module = model
    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module
