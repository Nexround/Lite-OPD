from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Tuple

import torch
from liteopd.inference.core import Batch, Req
from liteopd.inference.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from liteopd.inference.kvcache import BaseCacheHandle
    from liteopd.inference.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


class AdmitStatus(Enum):
    ADMITTED = auto()
    SKIP = auto()
    STOP = auto()


@dataclass
class AdmitResult:
    status: AdmitStatus
    req: Req | None = None


def _kv_tokens_needed(input_len: int, cached_len: int, decoded_len: int, max_output_len: int, reserve: int) -> int:
    """KV tokens this request needs: uncached prefill + max(already decoded, decode reserve)."""
    return (input_len - cached_len) + max(decoded_len, min(max_output_len, reserve))


@dataclass
class PrefillAdder:
    """
    Greedy admission controller for one prefill round.

    Tracks two budgets:
      token_budget     — max prefill tokens this round (prevents long single-round prefills)
      tokens_committed — KV tokens already promised to admitted requests this round;
                         used to tighten the capacity check for each subsequent candidate
    """
    token_budget: int
    cache_manager: CacheManager
    table_manager: TableManager
    admission_reserve_tokens: int = 512

    def __post_init__(self) -> None:
        self.tokens_committed: int = 0
        # Total KV capacity (fixed), used to detect sequences that can never fit.
        self._total_capacity: int = self.cache_manager.total_capacity

    @property
    def _kv_headroom(self) -> int:
        """KV tokens still available for new admissions this round."""
        return self.cache_manager.available_size - self.tokens_committed

    def try_add_one(self, pending_req: PendingReq) -> AdmitResult:
        """Attempt to admit one request for this prefill round."""
        if self.token_budget <= 0:
            return AdmitResult(AdmitStatus.STOP)

        # --- Resume a chunked (partially-prefilled) request ---
        if chunked_req := pending_req.chunked_req:
            return AdmitResult(
                AdmitStatus.ADMITTED,
                self._build_req(
                    pending_req,
                    chunked_req.cache_handle,
                    chunked_req.table_idx,
                    cached_len=chunked_req.cached_len,
                ),
            )

        # --- New request: check table slot and KV capacity ---
        if self.table_manager.available_size == 0:
            return AdmitResult(AdmitStatus.STOP)

        handle = self.cache_manager.match_req(pending_req).cuda_handle
        cached_len = handle.cached_len
        kv_needed = _kv_tokens_needed(
            pending_req.input_len, cached_len, pending_req.decoded_len, pending_req.max_output_len, self.admission_reserve_tokens
        )

        if kv_needed > self._total_capacity:
            raise RuntimeError(
                f"Sequence requires {kv_needed} KV tokens but total KV capacity is "
                f"{self._total_capacity}. Reduce sequence length or increase KV cache size."
            )
        if kv_needed > self._kv_headroom:
            return AdmitResult(AdmitStatus.SKIP)

        # Lock the matched prefix so it isn't evicted before allocate_paged runs.
        self.cache_manager.lock(handle)
        # Re-check after lock: locking reduces available_size (locked nodes leave evictable pool).
        if kv_needed > self._kv_headroom:
            self.cache_manager.unlock(handle)
            return AdmitResult(AdmitStatus.SKIP)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            self.table_manager.token_pool[table_idx, :cached_len].copy_(
                pending_req.input_ids[:cached_len].pin_memory(), non_blocking=True
            )
            self.table_manager.page_table[table_idx, :cached_len].copy_(
                handle.get_matched_indices()
            )

        return AdmitResult(
            AdmitStatus.ADMITTED,
            self._build_req(pending_req, handle, table_idx, cached_len=cached_len),
        )

    def _build_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        uncached_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, uncached_len)
        is_chunked = chunk_size < uncached_len

        self.token_budget -= chunk_size
        self.tokens_committed += _kv_tokens_needed(
            pending_req.input_len, cached_len, pending_req.decoded_len, pending_req.max_output_len, self.admission_reserve_tokens
        )

        self.table_manager.token_pool[table_idx, cached_len : cached_len + chunk_size].copy_(
            pending_req.input_ids[cached_len : cached_len + chunk_size].pin_memory(),
            non_blocking=True,
        )
        return (ChunkedReq if is_chunked else Req)(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            max_output_len=pending_req.max_output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    admission_reserve_tokens: int = 512
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.enqueue(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def enqueue(self, req: PendingReq) -> None:
        self.pending_list.append(req)
        self._sort_pending()

    def extend_pending(self, reqs: List[PendingReq]) -> None:
        if not reqs:
            return
        self.pending_list.extend(reqs)
        self._sort_pending()

    def prepend_pending(self, reqs: List[PendingReq]) -> None:
        if not reqs:
            return
        self.pending_list = list(reqs) + self.pending_list
        self._sort_pending()

    def _sort_pending(self) -> None:
        self.pending_list.sort(key=lambda req: req.priority_key)

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None

        adder = PrefillAdder(
            token_budget=prefill_budget,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            admission_reserve_tokens=self.admission_reserve_tokens,
        )
        admitted: List[Req] = []
        next_pending: List[PendingReq] = []

        for pending_req in self.pending_list:
            result = adder.try_add_one(pending_req)
            if result.status is AdmitStatus.STOP:
                break
            if result.status is AdmitStatus.SKIP:
                next_pending.append(pending_req)
                continue

            req = result.req
            assert req is not None
            pending_req.chunked_req = req if isinstance(req, ChunkedReq) else None
            if pending_req.chunked_req:
                next_pending.append(pending_req)
            admitted.append(req)

        if not admitted:
            return None

        scanned_count = len(admitted) + len(next_pending)
        self.pending_list = next_pending + self.pending_list[scanned_count:]
        self._sort_pending()
        return Batch(reqs=admitted, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return bool(self.pending_list)
