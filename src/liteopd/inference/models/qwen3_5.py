"""Qwen3.5 inference engine model.

Architecture
------------
Hybrid model: ``layer_types`` marks each of the ``num_layers`` layers as
"full_attention" or "linear_attention" (GatedDeltaNet).

Full-attention (every full_attention_interval-th layer)
  Like Qwen3 but the Q projection is doubled: [q | gate].
  Output gating: attn_out *= sigmoid(gate).

GatedDeltaNet (the other 3/4 of layers)
  Linear recurrent attention.  One fixed-size state matrix S per request:
    S ∈ R^{local_nv × dk × dv}  (TP-sharded value heads)
  Update formula (Gated DeltaNet, GQA-aware):
    kh = h // kv_groups    (which key-head serves value-head h)
    decay_h = exp(g_t[kh])
    S[h] ← decay_h * S[h] + outer(v_t[h] - S[h]@k_t[kh], b_t[kh] * k_t[kh])
    o_t[h] = S[h] @ q_t[kh]
  Decode state accesses go through the working buffer (pre-gathered before,
  scattered back after, by Engine.forward_batch).  Both the FLA chunk-T=1
  path and the PyTorch fallback are CUDA-graph-safe — see "CUDA-graph capture".

Tensor parallelism
  Full-attention  → same as Qwen3 (already TP-aware via RopeAttn / LinearQKV)
  GatedDeltaNet   → col-parallel for in-projections, row-parallel for out_proj
                    conv1d channels sharded with value heads

Flash-linear-attention (optional)
  If ``flash-linear-attention`` is installed, prefill uses ``chunk_gated_delta_rule``
  for O(T / chunk) speedup.  Falls back to a PyTorch loop otherwise.

CUDA-graph capture — ENABLED for Qwen3.5
  Decode steps are CUDA-graph-safe under both execution paths:

  • FLA path (``flash-linear-attention`` installed):
      _decode_deltanet calls ``chunk_gated_delta_rule`` with T=1.  All input
      and output tensor shapes are fixed for a given padded batch size, so the
      Triton kernel is captured cleanly.  The final state is copied back to
      pool.working_state (fixed-address buffer) via an in-place ``.copy_()``.

  • PyTorch einsum fallback (FLA not installed):
      All intermediate tensors are fixed-size for a given padded batch size;
      pool.working_state is a pre-allocated fixed-address buffer.  The cached
      self._kh_idx avoids any dynamic allocation inside the captured region.

  Leave ``generation_cuda_graph_max_bs`` unset (or None) in Qwen3.5 configs
  to enable auto-detection (256 on H200, 160 otherwise).  Set to 0 only to
  force eager mode for debugging.

Weight-name compatibility with HuggingFace Qwen3.5
  Full-attention layers   → model.layers.{i}.self_attn.*
  GatedDeltaNet layers    → model.layers.{i}.attn.*
  All layers share        → model.layers.{i}.mlp.*  /  layernorm.*
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from liteopd.inference.core import get_global_ctx
from liteopd.inference.distributed import get_tp_info
from liteopd.inference.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from liteopd.inference.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from liteopd.inference.kvcache import RecurrentStatePool
    from .config import ModelConfig


# ---------------------------------------------------------------------------
# Optional CUDA-accelerated kernel imports
# ---------------------------------------------------------------------------

# flash-linear-attention: chunk_gated_delta_rule for prefill O(T/chunk) path
_FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk  # type: ignore[import]
    _FLA_AVAILABLE = True
except ImportError:
    _fla_chunk = None  # type: ignore[assignment]

# causal-conv1d (Dao-AILab): fused causal depthwise Conv1d
#   causal_conv1d_fn    — prefill: fused padding+conv, returns final state
#   causal_conv1d_update — decode: fused state-roll+conv, in-place state update,
#                          conv_state_indices for direct pool slot access
_CAUSAL_CONV1D_AVAILABLE = False
try:
    from causal_conv1d import causal_conv1d_fn as _cc1d_fn          # type: ignore[import]
    from causal_conv1d import causal_conv1d_update as _cc1d_update   # type: ignore[import]
    _CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _cc1d_fn = _cc1d_update = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Full-attention (every full_attention_interval-th layer)
# ---------------------------------------------------------------------------

class Qwen3_5FullAttn(BaseOP):
    """Multi-head attention for Qwen3.5 full_attention layers.

    Q projection is doubled to carry the per-token output gate.
    After attention: attn_out = attn_out * sigmoid(gate).
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        tp_info = get_tp_info()
        head_dim = config.head_dim
        q_dim    = config.num_qo_heads * head_dim
        k_dim    = config.num_kv_heads * head_dim

        # [q+gate, k, v] — gate is the extra q_dim block
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, [2 * q_dim, k_dim, k_dim], has_bias=False
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn   = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(q_dim, config.hidden_size, has_bias=False)

        local_num_qo = div_even(config.num_qo_heads, tp_info.size)
        self._local_q = local_num_qo * head_dim

    @nvtx_annotate("MHA_gated")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_gate = self.qkv_proj.forward(x)
        del x
        local_kv = qkv_gate.shape[-1] - 2 * self._local_q
        q_and_gate, kv = qkv_gate.split([2 * self._local_q, local_kv], dim=-1)
        del qkv_gate
        q, gate = q_and_gate.split([self._local_q, self._local_q], dim=-1)
        del q_and_gate
        attn_out = self.attn.forward(torch.cat([q, kv], dim=-1))
        del q, kv
        attn_out = attn_out * torch.sigmoid(gate)
        del gate
        return self.o_proj.forward(attn_out)


# ---------------------------------------------------------------------------
# GatedDeltaNet (linear_attention layers)
# ---------------------------------------------------------------------------

class _DeltaNetConv1d(BaseOP):
    """Depthwise causal Conv1d weights container.

    weight: [local_ch, 1, kernel_size]
    bias:   [local_ch]
    The convolution is computed inline in GatedDeltaNetAttn._apply_conv.
    """

    def __init__(self, local_ch: int, kernel_size: int):
        self.weight = torch.empty(local_ch, 1, kernel_size)
        self.bias   = torch.empty(local_ch)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use GatedDeltaNetAttn._apply_conv instead")


class _DeltaNetRMSNorm(BaseOP):
    """Qwen3.5 gated-RMSNorm: output = (1 + weight) * rms_norm(x).

    Checkpoint initialises weight to 0, so effective scale starts at 1.
    Each TP rank holds weight[local_nv*dv] (col-parallel shard).
    """

    def __init__(self, local_size: int, eps: float = 1e-6):
        self.weight = torch.empty(local_size)
        self._eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var    = x.float().pow(2).mean(-1, keepdim=True).add_(self._eps)
        x_norm = x * torch.rsqrt(var).to(x.dtype)
        return x_norm * (1.0 + self.weight)


class GatedDeltaNetAttn(BaseOP):
    """Gated DeltaNet (linear recurrent attention) layer.

    All projections are TP-aware:
      in_proj_qkv  col-parallel  [hidden → 2*nk*dk+nv*dv,  sharded by head]
      in_proj_z    col-parallel  [hidden → nv*dv,           sharded]
      in_proj_b    col-parallel  [hidden → nk*dk,           sharded]
      in_proj_a    col-parallel  [hidden → nk,              sharded]
      conv1d       local weight  [local_conv_ch, 1, ks]
      norm         local weight  [local_nv*dv]
      out_proj     row-parallel  [nv*dv → hidden]           + all-reduce

    Decode uses working buffers (pre-gathered by Engine.forward_batch) instead
    of dynamic pool indexing.  Decode steps are CUDA-graph-safe (see module
    docstring); both the FLA chunk-T=1 path and the PyTorch einsum fallback
    operate exclusively on fixed-address, fixed-shape tensors.

    Prefill uses the flash-linear-attention chunk kernel when available,
    falling back to a PyTorch token loop otherwise.
    """

    def __init__(self, config: ModelConfig, layer_id: int, local_delta_idx: int):
        assert config.linear_num_key_heads is not None, (
            "ModelConfig is missing linear_num_key_heads; "
            "ensure model_type is qwen3_5_text."
        )
        tp_info = get_tp_info()

        nk = config.linear_num_key_heads
        nv = config.linear_num_value_heads
        dk = config.linear_key_head_dim
        dv = config.linear_value_head_dim
        ks = config.linear_conv_kernel_dim or 4
        h  = config.hidden_size

        local_nk      = div_even(nk, tp_info.size)
        local_nv      = div_even(nv, tp_info.size)
        local_conv_ch = 2 * local_nk * dk + local_nv * dv

        # ---- projections ------------------------------------------------
        # Col-parallel: each rank gets [nk/tp*dk, nk/tp*dk, nv/tp*dv]
        self.in_proj_qkv = LinearColParallelMerged(h, [nk*dk, nk*dk, nv*dv], has_bias=False)
        self.in_proj_z   = LinearColParallelMerged(h, [nv*dv],               has_bias=False)
        self.in_proj_b   = LinearColParallelMerged(h, [nk*dk],               has_bias=False)
        self.in_proj_a   = LinearColParallelMerged(h, [nk],                  has_bias=False)
        # Depthwise causal conv on the local projection output
        self.conv1d      = _DeltaNetConv1d(local_ch=local_conv_ch, kernel_size=ks)
        # Row-parallel: takes [local_nv*dv] input, all-reduces to [h]
        self.out_proj    = LinearRowParallel(nv * dv, h, has_bias=False)
        # Gated RMSNorm with (1+w) semantics; weight sharded with value heads
        self.norm        = _DeltaNetRMSNorm(local_size=local_nv * dv)

        # ---- stored for forward ----------------------------------------
        self._layer_id        = layer_id
        self._local_delta_idx = local_delta_idx
        self._local_nk        = local_nk
        self._local_nv        = local_nv
        self._dk              = dk
        self._dv              = dv
        self._ks              = ks
        self._kv_groups       = nv // nk    # GQA factor (invariant under TP)
        self._local_qkv_out   = local_conv_ch
        # Maps each value head to its corresponding key head (GQA broadcast).
        # Lazily initialised on first forward call so we have the right device.
        # Cached as a fixed tensor to avoid a torch.arange allocation on every
        # decode step, which would prevent CUDA-graph capture.
        self._kh_idx: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    @nvtx_annotate("DeltaNet")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx   = get_global_ctx()
        batch = ctx.batch
        pool  = ctx.recurrent_pool
        if pool is None:
            raise RuntimeError(
                "Context.recurrent_pool is None — Qwen3.5 GatedDeltaNet requires "
                "a RecurrentStatePool (created for Qwen3.5 models by Engine.__init__)."
            )

        local_nk, local_nv = self._local_nk, self._local_nv
        dk, dv = self._dk, self._dv

        # ---- projections -----------------------------------------------
        qkv_raw = self.in_proj_qkv.forward(x)              # [T, 2*local_nk*dk+local_nv*dv]
        z       = F.silu(self.in_proj_z.forward(x))        # [T, local_nv*dv]
        b       = torch.sigmoid(self.in_proj_b.forward(x)) # [T, local_nk*dk]
        g       = -F.softplus(self.in_proj_a.forward(x))   # [T, local_nk]

        # ---- causal conv1d --------------------------------------------
        qkv = self._apply_conv(qkv_raw, batch, pool)        # [T, 2*local_nk*dk+local_nv*dv]
        del qkv_raw

        # ---- split q, k, v --------------------------------------------
        q_flat, k_flat, v_flat = qkv.split(
            [local_nk * dk, local_nk * dk, local_nv * dv], dim=-1
        )
        del qkv
        q = q_flat.view(-1, local_nk, dk)   # [T, local_nk, dk]
        k = k_flat.view(-1, local_nk, dk)
        v = v_flat.view(-1, local_nv, dv)   # [T, local_nv, dv]
        b = b.view(-1, local_nk, dk)

        # ---- recurrent computation ------------------------------------
        if batch.is_prefill:
            o = self._prefill_deltanet(q, k, v, b, g, batch, pool)
        else:
            o = self._decode_deltanet(q, k, v, b, g, batch, pool)
        # o: [T, local_nv, dv]

        # ---- output gate + norm + projection --------------------------
        o_flat   = o.reshape(-1, local_nv * dv)
        o_gated  = o_flat * z
        o_normed = self.norm.forward(o_gated)
        return self.out_proj.forward(o_normed)

    # ------------------------------------------------------------------
    # Causal Conv1d
    # ------------------------------------------------------------------

    def _apply_conv(
        self, qkv_raw: torch.Tensor, batch, pool: "RecurrentStatePool"
    ) -> torch.Tensor:
        """Apply depthwise causal Conv1d to qkv_raw.

        Prefill — uses causal_conv1d_fn (library) or F.conv1d (fallback):
          • Reads initial_states from pool.conv_buffer[li, table_idx]
          • Returns final_states into pool.conv_buffer[li, table_idx] in-place
          • One kernel call per sequence; no manual buffer management

        Decode — uses causal_conv1d_update with conv_state_indices (library) or
                  F.conv1d with index_put_ (fallback):
          • conv_state = pool.conv_buffer[li]  [n_slots, conv_ch, ks-1]
          • conv_state_indices = pool.table_indices_buf[:B]  (pre-filled in gather)
          • The kernel reads/writes the correct pool slots directly — zero extra
            gather/scatter for conv state (CUDA-graph-safe)
        """
        li      = self._local_delta_idx
        ks      = self._ks
        conv_ch = self._local_qkv_out

        # weight for library call: [conv_ch, ks]  (no groups dim)
        w_2d = self.conv1d.weight.squeeze(1)          # [conv_ch, ks]
        bias  = self.conv1d.bias                      # [conv_ch]

        if ks <= 1:
            # kernel_size=1: pointwise scale + bias (no state)
            return qkv_raw * w_2d[:, 0].unsqueeze(0) + bias.unsqueeze(0)

        # ------------------------------------------------------------------
        # PREFILL
        # ------------------------------------------------------------------
        if batch.is_prefill:
            outputs: List[torch.Tensor] = []
            offset = 0
            for req in batch.reqs:
                T     = req.extend_len
                seg   = qkv_raw[offset : offset + T]   # [T, conv_ch]

                if _CAUSAL_CONV1D_AVAILABLE and _cc1d_fn is not None:
                    # causal_conv1d >= new API requires initial_states.stride(1) == 1
                    # (channels must be contiguous in memory).  The pool buffer is
                    # [n_layers, n_slots, conv_ch, ks-1] so a [li, idx] slice has
                    # strides (ks-1, 1) → stride(1) == ks-1 after unsqueeze.
                    # Transpose to make channels innermost, copy to get contiguous
                    # memory, then transpose back: strides become (1, conv_ch).
                    s0_2d = pool.conv_buffer[li, req.table_idx]           # [conv_ch, ks-1]
                    s0 = s0_2d.T.contiguous().T.unsqueeze(0)              # [1, conv_ch, ks-1], stride(1)==1
                    sf = torch.empty_like(s0)
                    out_t = _cc1d_fn(
                        x=seg.T.unsqueeze(0),       # [1, conv_ch, T]
                        weight=w_2d,
                        bias=bias,
                        initial_states=s0,
                        return_final_states=True,
                        final_states_out=sf,
                    )  # out_t: [1, conv_ch, T]
                    pool.conv_buffer[li, req.table_idx].copy_(sf.squeeze(0))
                else:
                    # F.conv1d fallback: left-pad with history buffer
                    seg_t   = seg.T                                         # [conv_ch, T]
                    buf     = pool.conv_buffer[li, req.table_idx]           # [conv_ch, ks-1]
                    padded  = torch.cat([buf, seg_t], dim=1).unsqueeze(0)   # [1, conv_ch, T+ks-1]
                    out_t   = F.conv1d(
                        padded, self.conv1d.weight, bias, groups=conv_ch
                    )                                                       # [1, conv_ch, T]
                    pool.conv_buffer[li, req.table_idx].copy_(
                        seg_t[:, -(ks - 1):].detach()
                    )

                outputs.append(out_t.squeeze(0).T)   # [T, conv_ch]
                offset += T
            return torch.cat(outputs, dim=0)

        # ------------------------------------------------------------------
        # DECODE
        # ------------------------------------------------------------------
        B = qkv_raw.shape[0]

        if _CAUSAL_CONV1D_AVAILABLE and _cc1d_update is not None:
            # conv_state = pool.conv_buffer[li]: [n_slots, conv_ch, ks-1]  contiguous
            # conv_state_indices selects the active slots by table_idx — kernel
            # reads + updates those rows in-place, no external gather/scatter.
            out = _cc1d_update(
                x=qkv_raw[:B],                              # [B, conv_ch]
                conv_state=pool.conv_buffer[li],            # [n_slots, conv_ch, ks-1]
                weight=w_2d,
                bias=bias,
                conv_state_indices=pool.table_indices_buf[:B],  # int32 [B]
            )
        else:
            # F.conv1d fallback: gather → manual multiply-sum → scatter
            tbl  = pool.table_indices_buf[:B].long()
            buf  = pool.conv_buffer[li][tbl]                # [B, conv_ch, ks-1]
            tok  = qkv_raw[:B].unsqueeze(-1)                # [B, conv_ch, 1]
            window = torch.cat([buf, tok], dim=2)           # [B, conv_ch, ks]
            w_bc   = w_2d.unsqueeze(0)                      # [1, conv_ch, ks]
            out    = (window * w_bc).sum(dim=-1) + bias.unsqueeze(0)   # [B, conv_ch]
            # Scatter new buffer back into pool (roll + new token)
            pool.conv_buffer[li].index_put_(
                (tbl,), torch.cat([buf[:, :, 1:], tok], dim=2).detach()
            )

        return out

    # ------------------------------------------------------------------
    # DeltaNet recurrent forward — prefill
    # ------------------------------------------------------------------

    def _prefill_deltanet(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        b: torch.Tensor, g: torch.Tensor,
        batch, pool: "RecurrentStatePool",
    ) -> torch.Tensor:
        """Prefill: process each sequence in the batch.

        When flash-linear-attention is available, uses ``chunk_gated_delta_rule``
        for O(T / chunk) throughput; otherwise falls back to a PyTorch loop.
        """
        li          = self._local_delta_idx
        kv_groups   = self._kv_groups
        local_nv    = self._local_nv
        local_nk    = self._local_nk
        dk, dv      = self._dk, self._dv

        if self._kh_idx is None:
            self._kh_idx = torch.arange(local_nv, device=q.device) // kv_groups
        kh_idx = self._kh_idx  # [local_nv]

        outputs: List[torch.Tensor] = []
        offset = 0

        for req in batch.reqs:
            T       = req.extend_len
            q_seq   = q[offset : offset + T]   # [T, local_nk, dk]
            k_seq   = k[offset : offset + T]
            v_seq   = v[offset : offset + T]   # [T, local_nv, dv]
            b_seq   = b[offset : offset + T]   # [T, local_nk, dk]
            g_seq   = g[offset : offset + T]   # [T, local_nk]

            S0 = pool.delta_state[req.table_idx, li].clone().float()  # [nv, dk, dv]

            if _FLA_AVAILABLE and _fla_chunk is not None:
                # ---- FLA path: [1, T, H, D] format ([B, T, H, D] required) ----
                # Expand key/query to value-head count (GQA broadcast)
                q_fla  = q_seq[:, kh_idx, :].unsqueeze(0)    # [1, T, local_nv, dk]
                k_fla  = k_seq[:, kh_idx, :].unsqueeze(0)    # [1, T, local_nv, dk]
                v_fla  = v_seq.unsqueeze(0)                   # [1, T, local_nv, dv]
                # beta per element [1, T, local_nv, dk]; g per head [1, T, local_nv]
                b_fla  = b_seq[:, kh_idx, :].unsqueeze(0)
                g_fla  = g_seq[:, kh_idx].unsqueeze(0)
                s0_fla = S0.unsqueeze(0)                      # [1, local_nv, dk, dv]

                try:
                    o_fla, sf_fla = _fla_chunk(
                        q_fla.to(torch.float32),
                        k_fla.to(torch.float32),
                        v_fla.to(torch.float32),
                        beta=b_fla.to(torch.float32),
                        g=g_fla.to(torch.float32),
                        initial_state=s0_fla,
                        output_final_state=True,
                    )
                    pool.delta_state[req.table_idx, li].copy_(
                        sf_fla.squeeze(0).to(pool.delta_state.dtype)
                    )
                    outputs.append(o_fla.squeeze(0).to(q.dtype))  # [T, local_nv, dv]
                    offset += T
                    continue
                except Exception:
                    pass  # fall through to PyTorch loop if FLA call fails

            # ---- PyTorch loop fallback -----------------------------------
            S   = S0
            seq_out: List[torch.Tensor] = []
            for t in range(T):
                k_t = k_seq[t][kh_idx].float()   # [local_nv, dk]  GQA broadcast
                q_t = q_seq[t][kh_idx].float()
                v_t = v_seq[t].float()            # [local_nv, dv]
                b_t = b_seq[t][kh_idx].float()
                g_t = g_seq[t][kh_idx].float()    # [local_nv]

                decay = torch.exp(g_t)
                S     = S * decay[:, None, None]
                Sk    = torch.einsum("hk,hkv->hv", k_t, S)
                delta = v_t - Sk
                S     = S + torch.einsum("hv,hk->hkv", delta, b_t * k_t)
                o_t   = torch.einsum("hk,hkv->hv", q_t, S)
                seq_out.append(o_t)

            pool.delta_state[req.table_idx, li].copy_(S.to(pool.delta_state.dtype))
            outputs.append(torch.stack(seq_out, dim=0).to(q.dtype))
            offset += T

        return torch.cat(outputs, dim=0)    # [total_tokens, local_nv, dv]

    # ------------------------------------------------------------------
    # DeltaNet recurrent forward — decode
    # ------------------------------------------------------------------

    def _decode_deltanet(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        b: torch.Tensor, g: torch.Tensor,
        batch, pool: "RecurrentStatePool",
    ) -> torch.Tensor:
        """Single-step DeltaNet for the full padded decode batch.

        Uses pool.working_state (pre-gathered by Engine.forward_batch).
        Fully vectorised — no Python loops.

        CUDA-graph status: SAFE.
          Both execution paths operate on fixed-address, fixed-shape tensors
          for a given padded batch size.  See module docstring for details.

        When flash-linear-attention is available, delegates to
        ``chunk_gated_delta_rule`` with T=1 (same kernel as prefill, GQA-
        broadcast inputs in [B, 1, H, D] layout (FLA requires [B, T, H, D]).  The
        updated state is copied back to pool.working_state in-place.

        Falls back to a vectorised PyTorch einsum when FLA is not installed.
        """
        li       = self._local_delta_idx
        local_nv = self._local_nv

        B = q.shape[0]   # padded batch size (fixed per CUDA graph)
        # Use cached head-index tensor: avoids torch.arange inside the graph.
        if self._kh_idx is None:
            self._kh_idx = torch.arange(local_nv, device=q.device) // self._kv_groups
        kh_idx = self._kh_idx  # [local_nv]

        if _FLA_AVAILABLE and _fla_chunk is not None:
            # ---- FLA path: chunk_gated_delta_rule with T=1 ---------------
            # Reshape inputs to [B, T=1, H, D] ([B, T, H, D] required by FLA).
            # GQA-broadcast q/k/b/g from key-head count to value-head count.
            q_fla = q[:, kh_idx, :].unsqueeze(1).float()   # [B, 1, local_nv, dk]
            k_fla = k[:, kh_idx, :].unsqueeze(1).float()   # [B, 1, local_nv, dk]
            v_fla = v.unsqueeze(1).float()                  # [B, 1, local_nv, dv]
            b_fla = b[:, kh_idx, :].unsqueeze(1).float()   # [B, 1, local_nv, dk]
            g_fla = g[:, kh_idx].unsqueeze(1).float()       # [B, 1, local_nv]
            # Read state from fixed-address working buffer.
            s0    = pool.working_state[:B, li].float()      # [B, local_nv, dk, dv]

            o_fla, sf_fla = _fla_chunk(
                q_fla, k_fla, v_fla,
                beta=b_fla,
                g=g_fla,
                initial_state=s0,
                output_final_state=True,
            )
            # Write updated state back to fixed-address working buffer.
            pool.working_state[:B, li].copy_(sf_fla.to(pool.working_state.dtype))
            return o_fla.squeeze(1).to(q.dtype)   # [B, local_nv, dv]

        # ---- PyTorch einsum fallback -------------------------------------
        # Load state from working buffer (fixed address).
        S   = pool.working_state[:B, li].float()  # [B, local_nv, dk, dv]
        # GQA broadcast: map from key-head indexing to value-head indexing.
        k_h = k[:, kh_idx, :].float()   # [B, local_nv, dk]
        q_h = q[:, kh_idx, :].float()   # [B, local_nv, dk]
        b_h = b[:, kh_idx, :].float()   # [B, local_nv, dk]
        g_h = g[:, kh_idx].float()      # [B, local_nv]
        v_f = v.float()                  # [B, local_nv, dv]

        decay = torch.exp(g_h)                                        # [B, local_nv]
        S     = S * decay[:, :, None, None]
        Sk    = torch.einsum("bhk,bhkv->bhv", k_h, S)
        delta = v_f - Sk
        S     = S + torch.einsum("bhv,bhk->bhkv", delta, b_h * k_h)
        o     = torch.einsum("bhk,bhkv->bhv", q_h, S)

        # Write updated state back to working buffer (scatter to pool happens
        # after the full forward pass in Engine.forward_batch).
        pool.working_state[:B, li].copy_(S.to(pool.working_state.dtype))
        return o.to(q.dtype)   # [B, local_nv, dv]


# ---------------------------------------------------------------------------
# Hybrid decoder layer
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        layer_type: str,
        local_delta_idx: Optional[int] = None,
    ):
        self._layer_id   = layer_id
        self._layer_type = layer_type

        if layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttn(config, layer_id)
        else:
            assert local_delta_idx is not None
            self.attn = GatedDeltaNetAttn(config, layer_id, local_delta_idx)

        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if self._layer_type == "full_attention":
            x = self.self_attn.forward(x)
        else:
            x = self.attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


# ---------------------------------------------------------------------------
# Model + top-level wrapper
# ---------------------------------------------------------------------------

class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        layer_types = _resolve_layer_types(config)
        layers: List[Qwen3_5DecoderLayer] = []
        delta_idx = 0
        for lid in range(config.num_layers):
            lt = layer_types[lid]
            if lt == "linear_attention":
                layers.append(Qwen3_5DecoderLayer(config, lid, lt, local_delta_idx=delta_idx))
                delta_idx += 1
            else:
                layers.append(Qwen3_5DecoderLayer(config, lid, lt, local_delta_idx=None))
        self.layers = OPList(layers)
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: Optional[torch.Tensor] = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model   = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_layer_types(config: ModelConfig) -> List[str]:
    if config.layer_types is not None:
        assert len(config.layer_types) == config.num_layers
        return list(config.layer_types)
    interval = config.full_attention_interval or 4
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(config.num_layers)
    ]


__all__ = ["Qwen3_5ForCausalLM"]
