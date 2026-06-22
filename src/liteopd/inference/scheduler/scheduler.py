from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from liteopd.inference.core import Batch, Req, SamplingParams
from liteopd.inference.env import ENV
from liteopd.inference.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from liteopd.inference.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from liteopd.inference.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

import os

_TRACE_LIMIT = 64
_trace_counter = 0


def _trace_enabled() -> bool:
    return os.environ.get("OPD_DEBUG_TRACE", "0") == "1"


def _trace(message: str) -> None:
    global _trace_counter
    if not _trace_enabled() or _trace_counter >= _TRACE_LIMIT:
        return
    _trace_counter += 1
    print(f"[opd.trace.scheduler] {message}", flush=True)

_MAX_PREEMPTION_RETRIES = 5

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        from liteopd.inference.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager,
            admission_reserve_tokens=config.admission_reserve_tokens,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = self._get_eos_token_ids(config.model_path, self.tokenizer)
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # Profiling stats
        self._preemption_count = 0
        self._loop_count = 0

    @staticmethod
    def _get_eos_token_ids(model_path: str, tokenizer) -> set:
        ids = set()
        if tokenizer.eos_token_id is not None:
            ids.add(tokenizer.eos_token_id)
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model_path)
            if isinstance(gc.eos_token_id, (list, tuple)):
                ids.update(gc.eos_token_id)
            elif gc.eos_token_id is not None:
                ids.add(gc.eos_token_id)
        except Exception:
            pass
        return ids

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

        self._loop_count += 1
        if self._loop_count % 500 == 0:
            self._log_periodic_stats()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                _trace(
                    f"process_last_data uid={req.uid} next_token={next_token} "
                    f"req_tail={req.input_ids[-min(8, len(req.input_ids)):].tolist()}"
                )
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token in self.eos_token_ids
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)
        # Reset DeltaNet recurrent state for this slot (Qwen3.5 only; no-op for others)
        pool = self.engine.ctx.recurrent_pool
        if pool is not None:
            pool.reset_request(req.table_idx)

    def _on_preempt(self, uid: int) -> None:
        """Hook for subclasses to react to preemption (e.g. reset output state)."""
        pass

    @staticmethod
    def _req_to_pending(req: Req) -> PendingReq:
        # Preserve input + already-generated tokens so re-prefill resumes without re-decoding.
        remaining = req.remain_len
        sp = SamplingParams(
            temperature=req.sampling_params.temperature,
            top_k=req.sampling_params.top_k,
            top_p=req.sampling_params.top_p,
            ignore_eos=req.sampling_params.ignore_eos,
            max_tokens=remaining,
        )
        return PendingReq(uid=req.uid, input_ids=req.input_ids[:req.device_len], sampling_params=sp,
                          decoded_len=req.decoded_len)

    def _preempt_longest(self) -> bool:
        """Preempt the decode request occupying the most KV pages. Returns True on success."""
        victim = self.decode_manager.find_longest_preemptable()
        if victim is None:
            return False
        self.decode_manager.remove_req(victim)
        self._free_req_resources(victim)
        pending = self._req_to_pending(victim)
        self.prefill_manager.enqueue(pending)
        logger.debug_rank0("Preempted req %d (device_len=%d)", victim.uid, victim.device_len)
        self._preemption_count += 1
        self._on_preempt(victim.uid)
        return True

    def _abort_prefill_batch(self, batch: Batch) -> None:
        """Return prefill requests from a failed batch back to pending list."""
        reinsert: List[PendingReq] = []
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                pending = self._req_to_pending(req)
                pending.chunked_req = req
                reinsert.append(pending)
            else:
                self.table_manager.free(req.table_idx)
                self.cache_manager.unlock(req.cache_handle)
                reinsert.append(self._req_to_pending(req))
        self.prefill_manager.prepend_pending(reinsert)

    def _prepare_batch(self, batch: Batch, _depth: int = 0) -> ForwardInput | None:
        if _depth > _MAX_PREEMPTION_RETRIES:
            logger.warning_rank0("Preemption depth limit reached, skipping batch")
            return None
        self.engine.graph_runner.pad_batch(batch)
        success = self.cache_manager.allocate_paged(batch.reqs)
        if not success:
            if batch.is_prefill:
                self._abort_prefill_batch(batch)
            if not self._preempt_longest():
                logger.warning_rank0("Cannot preempt: no eligible requests")
                return None
            new_batch = self._schedule_next_batch_inner()
            if new_batch is None:
                return None
            return self._prepare_batch(new_batch, _depth + 1)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch_inner(self) -> Batch | None:
        return (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        batch = self._schedule_next_batch_inner()
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _log_periodic_stats(self) -> None:
        """Log periodic profiling stats during rollout."""
        kv_total = self.cache_manager.total_capacity
        active_kv_tokens = sum(req.device_len for req in self.decode_manager.running_reqs)
        prefix_cache_info = self.cache_manager.prefix_cache.size_info
        logger.info(
            "loop=%d kv=%.1f%% (%d/%d) running=%d pending=%d preemptions=%d "
            "prefix_cache(protected=%d evictable=%d evictable_leaf=%d)",
            self._loop_count,
            100.0 * active_kv_tokens / max(kv_total, 1),
            active_kv_tokens,
            kv_total,
            len(self.decode_manager.running_reqs), len(self.prefill_manager.pending_list),
            self._preemption_count,
            prefix_cache_info.protected_size,
            prefix_cache_info.evictable_size,
            prefix_cache_info.evictable_leaf_size,
        )


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
