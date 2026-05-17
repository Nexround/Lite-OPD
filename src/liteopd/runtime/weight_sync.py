"""Zero-copy weight sharing between HuggingFace nn.Module and inference engine.

After fuse_model_projections(), the HF model has fused qkv_proj and gate_up_proj.
share_weights() makes the inference engine's weight tensors reference the same
GPU memory as the HF model's parameters — optimizer updates are immediately
visible to the engine with zero copy overhead.

For Gemma3 models, norm weights use `(1 + weight)` semantics. The inference
engine handles this via GemmaRMSNorm (flashinfer.gemma_rmsnorm kernel), so
norm weights are shared zero-copy just like other weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def share_weights(hf_model: nn.Module, engine_model) -> None:
    """Make inference engine weights directly reference HF model parameter data.

    Requires the HF model to have fused projections (qkv_proj, gate_up_proj).
    After this call, optimizer updates to HF parameters are immediately visible
    to the inference engine with zero copy overhead.
    """
    raw_hf = hf_model.module if hasattr(hf_model, "module") else hf_model
    hf_params = dict(raw_hf.named_parameters())
    num_layers = sum(1 for name in hf_params if name.endswith(".self_attn.qkv_proj.weight"))

    def _set(engine_key: str, hf_key: str) -> None:
        src = _get_hf_param(raw_hf, hf_key)
        _set_engine_tensor(engine_model, engine_key, src)

    _set("model.embed_tokens.weight", "model.embed_tokens.weight")

    has_separate_lm_head = any(k.startswith("lm_head.") for k in hf_params)
    if has_separate_lm_head:
        _set("lm_head.weight", "lm_head.weight")
    else:
        _set("lm_head.weight", "model.embed_tokens.weight")

    _set("model.norm.weight", "model.norm.weight")

    for i in range(num_layers):
        p = f"model.layers.{i}"
        _set(f"{p}.self_attn.qkv_proj.weight", f"{p}.self_attn.qkv_proj.weight")
        if f"{p}.self_attn.qkv_proj.bias" in hf_params:
            _set(f"{p}.self_attn.qkv_proj.bias", f"{p}.self_attn.qkv_proj.bias")
        if f"{p}.self_attn.q_norm.weight" in hf_params:
            _set(f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.q_norm.weight")
        if f"{p}.self_attn.k_norm.weight" in hf_params:
            _set(f"{p}.self_attn.k_norm.weight", f"{p}.self_attn.k_norm.weight")
        _set(f"{p}.self_attn.o_proj.weight", f"{p}.self_attn.o_proj.weight")
        _set(f"{p}.mlp.gate_up_proj.weight", f"{p}.mlp.gate_up_proj.weight")
        _set(f"{p}.mlp.down_proj.weight", f"{p}.mlp.down_proj.weight")
        _set(f"{p}.input_layernorm.weight", f"{p}.input_layernorm.weight")
        _set(f"{p}.post_attention_layernorm.weight", f"{p}.post_attention_layernorm.weight")
        if f"{p}.pre_feedforward_layernorm.weight" in hf_params:
            _set(f"{p}.pre_feedforward_layernorm.weight", f"{p}.pre_feedforward_layernorm.weight")
        if f"{p}.post_feedforward_layernorm.weight" in hf_params:
            _set(f"{p}.post_feedforward_layernorm.weight", f"{p}.post_feedforward_layernorm.weight")


def _get_hf_param(model: nn.Module, key: str) -> torch.Tensor:
    parts = key.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj.data if isinstance(obj, nn.Parameter) else obj


def _get_engine_tensor(model, key: str) -> torch.Tensor:
    """Get a tensor from the engine model by dot-separated path."""
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        if part == "op_list":
            continue
        if part.isdigit():
            obj = obj.op_list[int(part)]
        else:
            obj = getattr(obj, part)
    return getattr(obj, parts[-1])


def _set_engine_tensor(model, key: str, value: torch.Tensor) -> None:
    """Set a tensor in the engine model to reference the given value's storage."""
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        if part == "op_list":
            continue
        if part.isdigit():
            obj = obj.op_list[int(part)]
        else:
            obj = getattr(obj, part)
    attr_name = parts[-1]
    target = getattr(obj, attr_name)
    if target.shape != value.shape:
        raise ValueError(
            f"Shape mismatch for {key}: engine={target.shape}, hf={value.shape}"
        )
    if target.device.type == "meta":
        setattr(obj, attr_name, value)
    else:
        target.set_(value.untyped_storage(), value.storage_offset(), value.shape, value.stride())
