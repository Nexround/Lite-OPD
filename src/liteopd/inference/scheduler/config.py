from __future__ import annotations

from dataclasses import dataclass

from liteopd.inference.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 65536
    cache_type: str = "radix"
    admission_reserve_tokens: int = 512
    max_preemptions_per_req: int = 3

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
