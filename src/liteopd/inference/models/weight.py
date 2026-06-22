from __future__ import annotations

import glob
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from liteopd.inference.distributed import get_tp_info
from liteopd.inference.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [
    ".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj",
    # DeltaNet col-parallel projections (in_proj_qkv handled separately below)
    ".in_proj_z", ".in_proj_b", ".in_proj_a",
    # DeltaNet output norm weight [nv*dv] sharded by value-head to [local_nv*dv]
    ".attn.norm",
]
_SPLIT_DIM_1 = [
    ".o_proj", ".down_proj",
    # DeltaNet row-parallel output projection
    ".attn.out_proj",
]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}


def _shard_deltanet_qkv(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    nk: int,
    dk: int,
    nv: int,
    dv: int,
) -> torch.Tensor | None:
    """Component-wise TP shard for Qwen3.5 DeltaNet in_proj_qkv and conv1d.

    Both tensors have output/channel dimension structured as [q|k|v]:
      in_proj_qkv.weight : [2*nk*dk + nv*dv, hidden]  — shard dim 0 per component
      conv1d.weight      : [2*nk*dk + nv*dv, 1, ks]   — same
      conv1d.bias        : [2*nk*dk + nv*dv]           — same

    A plain chunk(n, dim=0) would straddle the q/k/v boundary, so each
    component is sharded independently then re-concatenated.

    Returns the sharded tensor, or None if the key doesn't match.
    """
    if n == 1:
        return None   # nothing to shard, fall through to _shard_tensor

    if ".in_proj_qkv" in key or ".conv1d" in key:
        q_size = nk * dk
        k_size = nk * dk
        v_size = nv * dv

        if value.ndim == 1:
            # bias: [2*q+v]
            q = value[:q_size].chunk(n, dim=0)[r]
            k = value[q_size : q_size + k_size].chunk(n, dim=0)[r]
            v = value[q_size + k_size :].chunk(n, dim=0)[r]
        else:
            # weight or 3-D conv kernel: first dim is channels/out-features
            q = value[:q_size].chunk(n, dim=0)[r]
            k = value[q_size : q_size + k_size].chunk(n, dim=0)[r]
            v = value[q_size + k_size :].chunk(n, dim=0)[r]
        return torch.cat([q, k, v], dim=0).clone()

    return None  # not a DeltaNet qkv/conv tensor


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")

                # Qwen3.5 DeltaNet qkv / conv1d need component-wise TP sharding
                if config.is_qwen3_5 and config.linear_num_key_heads is not None:
                    sharded = _shard_deltanet_qkv(
                        name, raw,
                        tp_info.rank, tp_info.size,
                        config.linear_num_key_heads, config.linear_key_head_dim,
                        config.linear_num_value_heads, config.linear_value_head_dim,
                    )
                    tensor = sharded if sharded is not None else _shard_tensor(
                        name, raw, tp_info.rank, tp_info.size, config.num_kv_heads
                    )
                else:
                    tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    yield (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    yield (merged_key, torch.cat(parts, dim=0))

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
