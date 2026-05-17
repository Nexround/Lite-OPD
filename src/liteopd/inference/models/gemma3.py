from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from liteopd.inference.core import get_global_ctx
from liteopd.inference.layers import (
    AttentionLayer,
    BaseOP,
    GemmaRMSNorm,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    gelu_and_mul,
)
from liteopd.inference.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig

if TYPE_CHECKING:
    pass


class Gemma3Attn(BaseOP):
    """Gemma3 attention with per-layer RoPE and Q/K norm."""

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)

        # Per-layer RoPE: sliding layers use local theta, global layers use global theta
        rotary_config = self._get_layer_rotary_config(config, layer_id)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @staticmethod
    def _get_layer_rotary_config(config: ModelConfig, layer_id: int) -> RotaryConfig:
        is_sliding = (
            config.layer_types is not None
            and config.layer_types[layer_id] == "sliding_attention"
        )
        if is_sliding and config.rope_local_base_freq is not None:
            return RotaryConfig(
                head_dim=config.head_dim,
                rotary_dim=config.head_dim,
                max_position=config.rotary_config.max_position,
                base=config.rope_local_base_freq,
                scaling=None,
            )
        return config.rotary_config

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


class Gemma3MLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.act_fn = gelu_and_mul
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class Gemma3DecoderLayer(BaseOP):
    """Gemma3 decoder layer with 4 RMSNorms and explicit residual connections."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Gemma3Attn(config, layer_id)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states)
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.embed_scale = config.embed_scale
        self.layers = OPList(
            [Gemma3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids) * self.embed_scale
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Gemma3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Gemma3ForCausalLM"]