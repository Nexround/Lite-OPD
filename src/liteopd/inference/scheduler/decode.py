from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from liteopd.inference.core import Batch, Req


@dataclass
class DecodeManager:
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    def find_longest_preemptable(self) -> Req | None:
        return max(self.running_reqs, key=lambda r: r.device_len, default=None)
