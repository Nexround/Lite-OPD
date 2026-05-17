from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class GemmaRMSNorm(BaseOP):
    """RMSNorm with Gemma-style (1 + weight) scaling, handled natively by kernel."""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import gemma_rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = gemma_rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if x.ndim == 2:
            self.rmsnorm(x, self.weight, self.eps, out=x)
        else:
            flat = x.contiguous().view(-1, x.shape[-1])
            self.rmsnorm(flat, self.weight, self.eps, out=flat)
            x.copy_(flat.view_as(x))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
