"""RuntimeCoordinator: orchestrates in-process rollout initialization."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .rollout import InProcessRolloutClient


class RuntimeCoordinator:
    """Thin orchestration layer that creates and manages the in-process rollout runtime.

    Future extensions: DDP barrier coordination, TP sub-group management,
    explicit prepare/release state machine.
    """

    def __init__(
        self,
        student_model: nn.Module,
        model_path: str,
        device: torch.device,
        tokenizer,
        *,
        memory_ratio: float = 0.9,
        max_running_req: int = 128,
        max_extend_tokens: int = 65536,
        cuda_graph_max_bs: int | None = None,
        page_size: int = 1,
        max_seq_len_override: int | None = None,
        num_page_override: int | None = None,
        attention_backend: str = "auto",
        admission_reserve_tokens: int = 512,
        max_preemptions_per_req: int = 3,
        use_vmm: bool = True,
        log_dir: str | None = None,
        rank: int = 0,
    ):
        self._student = student_model
        self._rollout = InProcessRolloutClient(
            hf_model=student_model,
            model_path=model_path,
            device=device,
            tokenizer=tokenizer,
            memory_ratio=memory_ratio,
            max_running_req=max_running_req,
            max_extend_tokens=max_extend_tokens,
            cuda_graph_max_bs=cuda_graph_max_bs,
            page_size=page_size,
            max_seq_len_override=max_seq_len_override,
            num_page_override=num_page_override,
            attention_backend=attention_backend,
            admission_reserve_tokens=admission_reserve_tokens,
            max_preemptions_per_req=max_preemptions_per_req,
            use_vmm=use_vmm,
            log_dir=log_dir,
            rank=rank,
        )

    @property
    def rollout_runtime(self) -> InProcessRolloutClient:
        return self._rollout

    def shutdown(self) -> None:
        self._rollout.shutdown()
