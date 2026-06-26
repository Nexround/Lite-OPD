from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints
import os
import types

import yaml


def _coerce_types(data: dict, cls: type) -> dict:
    """Coerce YAML-loaded values to match dataclass field type annotations."""
    hints = get_type_hints(cls)
    for key, value in list(data.items()):
        if key not in hints or value is None:
            continue
        ann = hints[key]
        origin = get_origin(ann)
        if origin is Union or origin is types.UnionType:
            args = [a for a in get_args(ann) if a is not type(None)]
            if not args:
                continue
            target = args[0]
        else:
            target = ann
        if target in (str, int, float, bool):
            try:
                data[key] = target(value)
            except (ValueError, TypeError):
                pass
    return data


@dataclass
class TrainConfig:
    loss: str
    student_model: str
    teacher_model: str
    train_dataset: str
    output_dir: str
    config_path: str | None = None
    eval_dataset: str | None = None
    learning_rate: float = 2e-6
    global_batch_size: int = 32
    generation_batch_size: int | None = None
    generation_mem_fraction_static: float = 0.9
    generation_top_k: int | None = 20
    enable_gradient_checkpointing: bool = True
    top_p: float = 0.9
    temperature: float = 0.6
    max_total_tokens: int = 16384
    train_subset_size: int = -1
    eval_subset_size: int = -1
    validation_subset_size: int = 0
    eval_every_steps: int = 10
    skip_eval: bool = False
    profile_memory: bool = False
    kl_backward_mode: str = "two_stage"
    log_every: int = 1
    max_steps: int = 20
    num_epochs: int = 1
    save_every_steps: int = 100
    generation_cuda_graph_max_bs: int | None = None
    generation_max_running_req: int = 128
    generation_page_size: int = 1
    generation_attention_backend: str = "auto"
    generation_admission_reserve_tokens: int = 2048
    generation_max_preemptions_per_req: int = 3
    generation_use_vmm: bool = True
    offload_teacher: bool = False
    compile_teacher: bool = True
    attn_implementation: str = "sdpa"
    distributed_strategy: str = "zero2"
    max_pack_tokens: int = 32768
    max_prompt_length: int = 1024
    warmup_ratio: float = 0.0
    cosine_annealing: bool = False
    # ---------------------------------------------------------------------------
    # Hybrid SFT + OPD training
    # ---------------------------------------------------------------------------
    # Dataset field that contains the gold prefix text.  When set, each training
    # sample is split into a gold-prefix region (supervised with cross-entropy
    # against the gold tokens) and a student-generated continuation region
    # (supervised with KL divergence against the teacher, as in standard OPD).
    # Set to None (default) to disable hybrid mode and run pure OPD.
    gold_prefix_field: str | None = None
    # Hard cap on gold-prefix length in tokens.  Prefixes longer than this are
    # truncated at the token boundary.  None means no cap.
    gold_prefix_max_tokens: int | None = None
    # Relative weight of the SFT (CE) loss with respect to the OPD (KL) loss.
    # Both losses are token-count-normalised before weighting, so 1.0 gives
    # equal per-token influence; values > 1 amplify the SFT signal.
    sft_loss_weight: float = 1.0


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["config_path"] = str(path)
    # Strip legacy fields that no longer exist in TrainConfig
    _legacy_fields = {
        "generation_backend", "generation_base_url", "generation_server_timeout",
        "generation_dp_size", "generation_tp_size", "generation_enable_dp_attention",
        "generation_update_weights_abort_all_requests",
        "generation_update_weights_with_memory_release",
        "training_batch_size", "grad_accum_steps",
        "enable_ood_eval", "ood_mmlu_subset_size", "ood_mbpp_subset_size", "ood_mbpp_num_workers",
    }
    for field in _legacy_fields:
        data.pop(field, None)
    data = _coerce_types(data, TrainConfig)
    cfg = TrainConfig(**data)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if not (0.0 < cfg.generation_mem_fraction_static <= 1.0):
        raise ValueError("generation_mem_fraction_static must be in (0, 1]")
    if cfg.generation_top_k is not None and cfg.generation_top_k <= 0:
        raise ValueError("generation_top_k must be positive when set")
    if cfg.global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size ({cfg.global_batch_size}) must be divisible by world_size ({world_size})")
    if cfg.generation_batch_size is None:
        cfg.generation_batch_size = cfg.global_batch_size // world_size
    elif cfg.generation_batch_size != cfg.global_batch_size // world_size:
        raise ValueError(
            f"generation_batch_size ({cfg.generation_batch_size}) must equal global_batch_size // world_size ({cfg.global_batch_size} // {world_size})"
        )
    if cfg.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if cfg.save_every_steps <= 0:
        raise ValueError("save_every_steps must be positive")
    if not cfg.skip_eval and cfg.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if cfg.kl_backward_mode not in {"sample", "chunk", "two_stage"}:
        raise ValueError("kl_backward_mode must be one of {'sample', 'chunk', 'two_stage'}")
    if cfg.attn_implementation not in {"eager", "sdpa", "flash_attention_2", "flex_attention"}:
        raise ValueError(
            f"attn_implementation must be one of {{'eager', 'sdpa', 'flash_attention_2', 'flex_attention'}}, got '{cfg.attn_implementation}'"
        )
    if cfg.distributed_strategy not in {"ddp", "zero2"}:
        raise ValueError("distributed_strategy must be one of {'ddp', 'zero2'}")
    if cfg.distributed_strategy == "ddp" and cfg.kl_backward_mode == "chunk":
        raise ValueError(
            "distributed_strategy='ddp' + kl_backward_mode='chunk' is unsupported: "
            "chunk mode uses retain_graph=True which is incompatible with flex_attention compiled kernels. "
            "Use kl_backward_mode='two_stage' (default) instead."
        )
    if not (0.0 <= cfg.warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must be in [0, 1)")
    if cfg.train_subset_size == 0:
        cfg.train_subset_size = -1
    if cfg.eval_subset_size == 0:
        cfg.eval_subset_size = -1
    if cfg.validation_subset_size < 0:
        raise ValueError("validation_subset_size must be non-negative")
    if cfg.gold_prefix_max_tokens is not None and cfg.gold_prefix_max_tokens <= 0:
        raise ValueError("gold_prefix_max_tokens must be a positive integer when set")
    if cfg.sft_loss_weight <= 0:
        raise ValueError("sft_loss_weight must be positive")
    return cfg
