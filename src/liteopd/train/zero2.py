"""Manual ZeRO-2 optimizer: shards optimizer state and gradients across ranks.

Parameters remain complete on every rank (enabling zero-copy weight sharing
with the inference engine). Only optimizer state and gradient buffers are
partitioned — each rank owns a contiguous shard of the flattened parameter
vector and runs AdamW only on that shard.

Communication pattern per step:
  1. reduce-scatter gradients (each rank gets averaged 1/N shard)
  2. local optimizer.step() on the shard
  3. all-gather updated parameter shards (restore full params)

Total communication = 2P, identical to DDP's all-reduce.

Memory layout (vs naive DDP):
  - _flat_params_padded: dual-use as param storage AND grad pack buffer
  - _m, _v: explicit AdamW moment tensors (1/N shard each)
  - _grad_shard_out: reduce-scatter output (1/N shard)
  - No separate _flat_grad or _shard_param buffers (saves 12 GB for 8B model, world_size=2)
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn


class ZeRO2Optimizer:
    """ZeRO-2 optimizer with flat-buffer parameter management.

    After construction, every parameter's .data becomes a view into a single
    contiguous flat buffer. This means all-gather writes directly update the
    parameters (and any inference engine tensors that share storage via
    tensor.set_()).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float,
        rank: int,
        world_size: int,
        process_group=None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        assert len(params) > 0
        self.rank = rank
        self.world_size = world_size
        self._params = params
        self._pg = process_group
        self._param_numels = [p.numel() for p in params]
        self._total_numel = sum(self._param_numels)

        device = params[0].device
        dtype = params[0].dtype

        # --- Flat parameter buffer ---
        # All parameters become views into this buffer.
        # Also reused as the gradient pack buffer during reduce_scatter_grads().
        self._flat_params = torch.empty(self._total_numel, dtype=dtype, device=device)
        offset = 0
        for p in params:
            numel = p.numel()
            self._flat_params[offset : offset + numel].copy_(p.data.view(-1))
            p.data = self._flat_params[offset : offset + numel].view(p.shape)
            offset += numel

        # --- Shard geometry ---
        self._shard_size = (self._total_numel + world_size - 1) // world_size
        padded_numel = self._shard_size * world_size

        # Padded buffer for reduce-scatter / all-gather (must be evenly divisible)
        if padded_numel > self._total_numel:
            self._flat_params_padded = torch.zeros(padded_numel, dtype=dtype, device=device)
            self._flat_params_padded[: self._total_numel].copy_(self._flat_params)
            self._flat_params = self._flat_params_padded[: self._total_numel]
            offset = 0
            for p in params:
                numel = p.numel()
                p.data = self._flat_params[offset : offset + numel].view(p.shape)
                offset += numel
        else:
            self._flat_params_padded = self._flat_params

        self._my_shard = self._flat_params_padded[
            rank * self._shard_size : (rank + 1) * self._shard_size
        ]

        # --- Manual AdamW state (1/N shard each) ---
        self._betas = betas
        self._eps = eps
        self._weight_decay = weight_decay
        self._step_count = 0
        self._m = torch.zeros(self._shard_size, dtype=dtype, device=device)
        self._v = torch.zeros(self._shard_size, dtype=dtype, device=device)

        # --- Gradient shard output buffer ---
        self._grad_shard_out = torch.zeros(self._shard_size, dtype=dtype, device=device)

        # param_groups for LR scheduler compatibility
        self.param_groups = [{"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "params": []}]

    def reduce_scatter_grads(self) -> None:
        """Collect param grads into flat buffer and reduce-scatter across ranks.

        Saves a snapshot of the current parameter shard before overwriting
        _flat_params_padded with gradients. The snapshot is consumed by step().
        """
        # Save pre-update param shard before overwriting _flat_params_padded with grads.
        # step() needs the pre-update values for decoupled weight decay.
        self._param_shard_snapshot = self._my_shard.clone()

        offset = 0
        for p in self._params:
            numel = p.numel()
            if p.grad is not None:
                self._flat_params_padded[offset : offset + numel].copy_(p.grad.view(-1))
                p.grad = None
            else:
                self._flat_params_padded[offset : offset + numel].zero_()
            offset += numel

        if self._flat_params_padded.numel() > self._total_numel:
            self._flat_params_padded[self._total_numel :].zero_()

        dist.reduce_scatter_tensor(
            self._grad_shard_out,
            self._flat_params_padded,
            op=dist.ReduceOp.AVG,
            group=self._pg,
        )

    def release_grad_buffer(self) -> None:
        """Free gradient shard buffer during rollout phase."""
        self._grad_shard_out = None

    def prepare_grad_buffer(self) -> None:
        """Re-allocate gradient shard buffer before training phase."""
        if self._grad_shard_out is None:
            self._grad_shard_out = torch.zeros(
                self._shard_size, dtype=self._m.dtype, device=self._m.device
            )

    def init_optimizer_state(self) -> None:
        """No-op: _m and _v are already allocated as zeros in __init__."""
        pass

    def step(self) -> None:
        """AdamW update on local shard, then all-gather to restore full params."""
        self._step_count += 1
        lr = self.param_groups[0]["lr"]
        beta1, beta2 = self._betas

        grad = self._grad_shard_out
        self._m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        self._v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** self._step_count
        bias_correction2 = 1.0 - beta2 ** self._step_count
        denom = (self._v.sqrt() / math.sqrt(bias_correction2)).add_(self._eps)
        step_size = lr / bias_correction1

        # param_snapshot holds pre-update param values (saved in reduce_scatter_grads)
        param_tmp = self._param_shard_snapshot
        if self._weight_decay != 0.0:
            param_tmp.mul_(1.0 - lr * self._weight_decay)
        param_tmp.addcdiv_(self._m, denom, value=-step_size)

        # All-gather from independent buffer into _flat_params_padded.
        # _my_shard is a view of _flat_params_padded → automatically updated.
        dist.all_gather_into_tensor(self._flat_params_padded, param_tmp, group=self._pg)
        del self._param_shard_snapshot

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients."""
        if set_to_none:
            for p in self._params:
                p.grad = None
        else:
            for p in self._params:
                if p.grad is not None:
                    p.grad.zero_()

    def state_dict(self):
        """Return optimizer state for checkpointing (shard-local)."""
        return {
            "m": self._m,
            "v": self._v,
            "step_count": self._step_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_size": self._shard_size,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state (must match rank and world_size)."""
        assert state["rank"] == self.rank
        assert state["world_size"] == self.world_size
        self._m.copy_(state["m"])
        self._v.copy_(state["v"])
        self._step_count = state["step_count"]
        for saved_pg, pg in zip(state["param_groups"], self.param_groups):
            pg.update({k: v for k, v in saved_pg.items() if k != "params"})
