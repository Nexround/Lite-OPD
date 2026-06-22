from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    model_type: str
    architectures: list[str]
    embed_scale: float = 1.0
    rope_local_base_freq: float | None = None
    layer_types: List[str] | None = None
    sliding_window: int | None = None
    query_pre_attn_scalar: float | None = None
    # Qwen3.5 GatedDeltaNet (linear_attention layers)
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    # Qwen3.5 full_attention output gate
    attn_output_gate: bool = False
    full_attention_interval: int | None = None

    @property
    def attn_scale(self) -> float:
        """Attention softmax scale. Uses query_pre_attn_scalar if set, else head_dim."""
        s = self.query_pre_attn_scalar if self.query_pre_attn_scalar is not None else self.head_dim
        return s ** -0.5

    @property
    def is_gemma3(self) -> bool:
        return self.model_type == "gemma3_text"

    @property
    def is_qwen3_5(self) -> bool:
        return self.model_type == "qwen3_5_text"

    @property
    def sliding_window_sizes(self) -> list[int]:
        """Per-layer sliding window size. -1 = full attention."""
        if self.layer_types is None or self.sliding_window is None:
            return [-1] * self.num_layers
        return [
            self.sliding_window if lt == "sliding_attention" else -1
            for lt in self.layer_types
        ]

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # hidden_act: Gemma3 uses "hidden_activation" instead of "hidden_act"
        hidden_act = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", "silu")
        _ACT_NORMALIZE = {"gelu_pytorch_tanh": "gelu"}
        hidden_act = _ACT_NORMALIZE.get(hidden_act, hidden_act)

        # rope_theta may be a direct attr or inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        # partial_rotary_factor: fraction of head_dim that participates in RoPE
        # (e.g. Qwen3.5 uses 0.25; most models use 1.0 = full head_dim)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)

        # Gemma3-specific fields
        embed_scale = 1.0
        rope_local_base_freq = None
        layer_types = None
        sliding_window = None
        query_pre_attn_scalar = None
        if model_type == "gemma3_text":
            embed_scale = math.sqrt(config.hidden_size)
            rope_local_base_freq = getattr(config, "rope_local_base_freq", 10_000.0)
            layer_types = getattr(config, "layer_types", None)
            sliding_window = getattr(config, "sliding_window", None)
            query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None)

        # Qwen3.5-specific fields
        linear_num_key_heads = None
        linear_num_value_heads = None
        linear_key_head_dim = None
        linear_value_head_dim = None
        linear_conv_kernel_dim = None
        attn_output_gate = False
        full_attention_interval = None
        if model_type == "qwen3_5_text":
            layer_types = getattr(config, "layer_types", None)
            linear_num_key_heads = getattr(config, "linear_num_key_heads", None)
            linear_num_value_heads = getattr(config, "linear_num_value_heads", None)
            linear_key_head_dim = getattr(config, "linear_key_head_dim", None)
            linear_value_head_dim = getattr(config, "linear_value_head_dim", None)
            linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)
            attn_output_gate = getattr(config, "attn_output_gate", False)
            full_attention_interval = getattr(config, "full_attention_interval", 4)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            model_type=model_type,
            architectures=architectures,
            embed_scale=embed_scale,
            rope_local_base_freq=rope_local_base_freq,
            layer_types=layer_types,
            sliding_window=sliding_window,
            query_pre_attn_scalar=query_pre_attn_scalar,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            attn_output_gate=attn_output_gate,
            full_attention_interval=full_attention_interval,
        )