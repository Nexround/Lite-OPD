"""Per-request recurrent state storage for GatedDeltaNet layers (Qwen3.5).

Two memory regions:

Persistent pool  (indexed by table_idx — same slot scheme as page_table)
  delta_state  [n_slots, n_delta_layers, local_nv, dk, dv]   float32
  conv_buffer  [n_delta_layers, n_slots, local_conv_ch, ks-1] model dtype

  conv_buffer is stored with n_delta_layers as the *outer* dimension so that
  pool.conv_buffer[li] is a contiguous [n_slots, local_conv_ch, ks-1] tensor.
  This lets causal_conv1d_update use conv_state_indices to read/write specific
  slots directly — no gather/scatter for the conv state.

Working buffers  (fixed-address, sized to max_padded_bs — CUDA-graph-safe)
  working_state [max_padded_bs, n_delta_layers, local_nv, dk, dv]   float32
  table_indices_buf [n_slots] int32   reusable scratch for index ops

  Only delta_state uses a working buffer (needed for the vectorised einsum
  that cannot do direct scatter inside the CUDA graph).  Conv state is handled
  entirely by causal_conv1d_update (no working buffer needed).

Usage during decode
  1. gather_for_decode(padded_reqs):
       • write table_idx values into table_indices_buf
       • copy delta_state[table_indices] → working_state
     Conv state: NOT gathered — causal_conv1d_update accesses pool.conv_buffer
     directly via conv_state_indices.
  2. CUDA graph replay:
       • GatedDeltaNetAttn reads/writes working_state (fixed address)
       • _apply_conv calls causal_conv1d_update with conv_state=pool.conv_buffer[li]
         and conv_state_indices=pool.table_indices_buf[:B]
  3. scatter_after_decode(padded_reqs):
       • copy working_state[:n] → delta_state[table_indices]  (delta only)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from liteopd.inference.core import Req
    from liteopd.inference.models import ModelConfig


class RecurrentStatePool:
    """Pre-allocated per-request DeltaNet recurrent state storage."""

    def __init__(
        self,
        max_req: int,
        n_delta_layers: int,
        local_nk: int,
        local_nv: int,
        dk: int,
        dv: int,
        local_conv_ch: int,
        kernel_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        n_slots     = max_req + 1       # +1 for dummy request
        ks_minus_1  = max(kernel_size - 1, 0)

        # ---- persistent delta-state [n_slots, n_delta_layers, nv_local, dk, dv] -----
        self.delta_state = torch.zeros(
            n_slots, n_delta_layers, local_nv, dk, dv,
            device=device, dtype=torch.float32,
        )

        # ---- persistent conv buffer [n_delta_layers, n_slots, ch_local, ks-1] -------
        # Layer-first layout so pool.conv_buffer[li] is contiguous; required by
        # causal_conv1d_update's conv_state_indices argument.
        self.conv_buffer = torch.zeros(
            n_delta_layers, n_slots, local_conv_ch, ks_minus_1,
            device=device, dtype=dtype,
        )

        # ---- reusable index scratch (int32 — required by conv_state_indices) --------
        self.table_indices_buf = torch.zeros(n_slots, device=device, dtype=torch.int32)

        # ---- working state (allocated lazily by init_working_buffers) ---------------
        self.working_state: torch.Tensor | None = None

        self._n_delta_layers = n_delta_layers
        self._local_nk       = local_nk
        self._local_nv       = local_nv
        self._dk             = dk
        self._dv             = dv
        self._local_conv_ch  = local_conv_ch
        self._ks             = kernel_size
        self._device         = device
        self._dtype          = dtype

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_delta_layers(self) -> int:
        return self._n_delta_layers

    @property
    def local_nv(self) -> int:
        return self._local_nv

    @property
    def local_nk(self) -> int:
        return self._local_nk

    @property
    def local_conv_ch(self) -> int:
        return self._local_conv_ch

    # ------------------------------------------------------------------
    # Working-buffer lifecycle (delta_state only)
    # ------------------------------------------------------------------

    def init_working_buffers(self, max_padded_bs: int) -> None:
        """Allocate a CUDA-graph-safe decode working buffer for delta_state.

        Conv state is handled by causal_conv1d_update (no working buffer needed).

        Args:
            max_padded_bs: maximum padded decode batch size (e.g. max_running_req+1).
        """
        self.working_state = torch.zeros(
            max_padded_bs, self._n_delta_layers,
            self._local_nv, self._dk, self._dv,
            device=self._device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Gather / scatter (called outside CUDA graph, delta_state only)
    # ------------------------------------------------------------------

    def gather_for_decode(self, padded_reqs: List["Req"]) -> None:
        """Populate table_indices_buf and copy delta_state → working_state.

        Conv state is NOT gathered here; causal_conv1d_update will access
        pool.conv_buffer[li] directly via conv_state_indices.

        Args:
            padded_reqs: batch.padded_reqs (includes dummy requests for padding).
        """
        assert self.working_state is not None, (
            "init_working_buffers() must be called before gather_for_decode()"
        )
        n = len(padded_reqs)

        # CPU int32 → GPU scratch (single H2D copy outside graph)
        idx_cpu = torch.tensor(
            [req.table_idx for req in padded_reqs], dtype=torch.int32
        )
        self.table_indices_buf[:n].copy_(idx_cpu, non_blocking=True)

        # Batched delta-state gather
        self.working_state[:n].copy_(
            self.delta_state[self.table_indices_buf[:n].long()], non_blocking=True
        )

    def scatter_after_decode(self, padded_reqs: List["Req"]) -> None:
        """Write working_state back to delta_state for the current batch slots.

        Conv state is already updated in-place by causal_conv1d_update —
        no scatter needed here.

        Args:
            padded_reqs: same list as passed to gather_for_decode().
        """
        assert self.working_state is not None, (
            "init_working_buffers() must be called before scatter_after_decode()"
        )
        n = len(padded_reqs)
        tbl_long = self.table_indices_buf[:n].long()
        self.delta_state.index_put_(
            (tbl_long,),
            self.working_state[:n].to(self.delta_state.dtype),
        )

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def reset_request(self, table_idx: int) -> None:
        """Zero out all state for a request slot (called when request completes)."""
        self.delta_state[table_idx].zero_()
        if self._ks > 1:
            # conv_buffer layout: [n_delta_layers, n_slots, ch, ks-1]
            self.conv_buffer[:, table_idx].zero_()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_recurrent_pool(
    model_config: ModelConfig,
    max_req: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RecurrentStatePool | None:
    """Return a RecurrentStatePool for Qwen3.5 models, or None for others."""
    if not model_config.is_qwen3_5:
        return None

    from liteopd.inference.distributed import get_tp_info
    from liteopd.inference.utils import div_even

    layer_types = model_config.layer_types
    if layer_types is None:
        interval = model_config.full_attention_interval or 4
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(model_config.num_layers)
        ]
    n_delta_layers = sum(1 for lt in layer_types if lt == "linear_attention")

    nk = model_config.linear_num_key_heads
    nv = model_config.linear_num_value_heads
    dk = model_config.linear_key_head_dim
    dv = model_config.linear_value_head_dim
    ks = model_config.linear_conv_kernel_dim or 4

    assert None not in (nk, nv, dk, dv), (
        "ModelConfig is missing linear_* fields for a qwen3_5_text model."
    )

    tp_info       = get_tp_info()
    local_nk      = div_even(nk, tp_info.size)
    local_nv      = div_even(nv, tp_info.size)
    local_conv_ch = 2 * local_nk * dk + local_nv * dv

    return RecurrentStatePool(
        max_req=max_req,
        n_delta_layers=n_delta_layers,
        local_nk=local_nk,
        local_nv=local_nv,
        dk=dk,
        dv=dv,
        local_conv_ch=local_conv_ch,
        kernel_size=ks,
        device=device,
        dtype=dtype,
    )


__all__ = ["RecurrentStatePool", "create_recurrent_pool"]
