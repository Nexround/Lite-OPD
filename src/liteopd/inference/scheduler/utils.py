from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from liteopd.inference.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    decoded_len: int = 0  # tokens already generated before this admission (non-zero after preemption)

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def max_output_len(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def priority_key(self) -> tuple[int, int, int]:
        """Shortest-seq-first priority for rollout batch completion.

        We use the request's current total target length
        (`current_context_len + remaining_decode_budget`) as the primary key so
        preempted requests and fresh requests follow the same ordering rule.
        """
        return (self.input_len + self.max_output_len, self.input_len, self.uid)


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
