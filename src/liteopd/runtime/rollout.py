"""In-process rollout client using embedded inference engine."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from liteopd.inference.core import SamplingParams
from liteopd.inference.distributed import DistributedInfo, set_tp_info
from liteopd.inference.layers import set_rope_device
from liteopd.inference.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from liteopd.inference.models import ModelConfig, create_model
from liteopd.inference.scheduler import Scheduler, SchedulerConfig
from liteopd.inference.utils import cached_load_hf_config, load_tokenizer, torch_dtype
from liteopd.inference.utils.logger import add_file_handler, init_logger

from .weight_sync import share_weights

logger = init_logger(__name__)


_TRACE_LIMIT = 64
_trace_counter = 0


def _trace_enabled() -> bool:
    return os.environ.get("OPD_DEBUG_TRACE", "0") == "1"


def _trace(message: str) -> None:
    global _trace_counter
    if not _trace_enabled() or _trace_counter >= _TRACE_LIMIT:
        return
    _trace_counter += 1
    print(f"[opd.trace.rollout] {message}", flush=True)


class _RequestAllFinished(Exception):
    pass


class InProcessRolloutClient:
    """Rollout client that runs inference in the same process as training."""

    refresh_on_all_ranks: bool = True

    def __init__(
        self,
        hf_model: nn.Module,
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
        self.model = model_path
        self._device = device
        self._tokenizer = tokenizer

        set_tp_info(rank=0, size=1)

        hf_config = cached_load_hf_config(model_path)
        model_config = ModelConfig.from_hf(hf_config)
        self._warn_if_unvalidated_model(hf_config, model_config, model_path)

        set_rope_device(device)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            self._sglang_model = create_model(model_config)

        self._materialize_sglang_model(device, torch.bfloat16)
        share_weights(hf_model, self._sglang_model)
        torch.cuda.empty_cache()

        if cuda_graph_max_bs is None:
            cuda_graph_max_bs = max_running_req

        scheduler_config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            max_running_req=max_running_req,
            memory_ratio=memory_ratio,
            cuda_graph_max_bs=cuda_graph_max_bs,
            page_size=page_size,
            max_seq_len_override=max_seq_len_override,
            num_page_override=num_page_override,
            embedded_mode=True,
            max_extend_tokens=max_extend_tokens,
            attention_backend=attention_backend,
            admission_reserve_tokens=admission_reserve_tokens,
            max_preemptions_per_req=max_preemptions_per_req,
            use_vmm=use_vmm,
        )

        self._scheduler = _EmbeddedScheduler(scheduler_config, self._sglang_model, log_dir=log_dir, rank=rank)

    @staticmethod
    def _warn_if_unvalidated_model(hf_config, model_config: ModelConfig, model_path: str) -> None:
        model_type = getattr(hf_config, "model_type", getattr(getattr(hf_config, "text_config", None), "model_type", "unknown"))
        is_validated = model_type in {"qwen2", "qwen3", "qwen3_5_text", "llama", "gemma3_text"}
        if is_validated:
            return
        logger.warning(
            "InProcessRolloutClient weight sharing/runtime path is only validated on dense Qwen2.5/Qwen3/Llama/Gemma3 models. "
            "Current model may hit load or numerical issues: model_path=%s model_type=%s architectures=%s",
            model_path,
            model_type,
            list(model_config.architectures),
        )

    def _materialize_sglang_model(self, device: torch.device, dtype: torch.dtype):
        """Move sglang model from meta device to real device with empty tensors."""
        sd = self._sglang_model.state_dict()
        materialized = {}
        for k, v in sd.items():
            materialized[k] = torch.empty(v.shape, dtype=dtype, device=device)
        self._sglang_model.load_state_dict(materialized)

    def generate_messages(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        chat_template_kwargs: Dict[str, Any] | None = None,
        max_concurrency: int | None = None,
    ) -> List[str]:
        prompts = []
        for messages in messages_batch:
            kwargs = chat_template_kwargs or {}
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
            max_tokens=max_tokens,
        )

        results = self._scheduler.generate(prompts, sampling_params)
        self.last_output_token_counts = [len(r["token_ids"]) for r in results]
        return [r["text"] for r in results]

    def generate_request_batch(
        self,
        requests_batch: List[Dict[str, Any]],
        max_concurrency: int | None = None,
    ) -> List[str]:
        prompts = []
        sampling_params_list = []
        for req in requests_batch:
            messages = req["messages"]
            kwargs = req.get("chat_template_kwargs") or {}
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            prompts.append(prompt)
            sampling_params_list.append(SamplingParams(
                temperature=req.get("temperature", 0.0),
                top_p=req.get("top_p", 1.0),
                top_k=req.get("top_k", -1),
                max_tokens=req.get("max_tokens", 1024),
            ))

        results = self._scheduler.generate(prompts, sampling_params_list)
        return [r["text"] for r in results]

    def refresh_from_model(self) -> None:
        """Flush prefix cache after optimizer step.

        Weight sharing means the engine already sees updated parameters —
        only the prefix cache needs invalidation.
        """
        self._scheduler.flush_cache()

    def prepare(self) -> None:
        """Prepare for rollout phase. Maps VMM physical memory if enabled."""
        kv_cache = self._scheduler.engine.kv_cache
        if hasattr(kv_cache, 'map_physical'):
            kv_cache.map_physical()

    def release(self) -> None:
        """Release rollout resources. Unmaps VMM physical memory if enabled."""
        self._scheduler.flush_cache()
        kv_cache = self._scheduler.engine.kv_cache
        if hasattr(kv_cache, 'unmap_physical'):
            kv_cache.unmap_physical()

    def shutdown(self) -> None:
        self._scheduler.engine.shutdown()


class _EmbeddedScheduler(Scheduler):
    """Scheduler subclass that uses an externally-provided model in embedded mode."""

    def __init__(self, config: SchedulerConfig, model, log_dir: str | None = None, rank: int = 0):
        from liteopd.inference.engine import Engine

        self.engine = Engine(config, model=model)

        self.device = self.engine.device
        self.stream = self.engine.stream
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)

        from liteopd.inference.scheduler.cache import CacheManager
        from liteopd.inference.scheduler.decode import DecodeManager
        from liteopd.inference.scheduler.prefill import PrefillManager
        from liteopd.inference.scheduler.table import TableManager

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager,
            admission_reserve_tokens=config.admission_reserve_tokens,
        )

        self.finished_reqs = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = self._get_eos_token_ids(config.model_path, self.tokenizer)
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # Offline mode I/O
        self.receive_msg = self.offline_receive_msg
        self.send_result = self.offline_send_result

        # LLM-style offline state
        self.pending_requests: List[tuple] = []
        self.status_map: Dict[int, _RequestStatus] = {}
        self.counter = 0
        self._preemption_count = 0
        self._loop_count = 0

        # Attach file logger if log_dir provided
        if log_dir:
            import logging
            from pathlib import Path
            log_path = str(Path(log_dir) / f"inference_engine.rank{rank}.log")
            add_file_handler(logger, log_path)
            sched_logger = logging.getLogger("opd.inference.scheduler.scheduler")
            for h in sched_logger.handlers:
                h.setLevel(logging.WARNING)
            add_file_handler(sched_logger, log_path)
            kv_total = self.engine.num_pages * self.cache_manager.page_size
            logger.info("Engine ready: kv_pool=%d tokens, max_running_req=%d", kv_total, self.table_manager._max_running_reqs)

    def _tokenize_one(self, prompt: str | List[int]) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=False
            ).view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0 and not self.decode_manager.runnable:
            raise _RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid = self.counter + added
            added += 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = _RequestStatus(
                uid=uid,
                input_ids=input_ids.tolist(),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                status.output_ids.append(msg.next_token)
            _trace(
                f"offline_send_result uid={msg.uid} next_token={msg.next_token} "
                f"finished={msg.finished} output_len={len(status.output_ids)}"
            )

    def run_when_idle(self) -> None:
        pass

    def sync_all_ranks(self) -> None:
        pass

    def _on_preempt(self, uid: int) -> None:
        if uid in self.status_map:
            self.status_map[uid].output_ids = []

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, Any]]:
        import time
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))

        start_time = time.time()
        start_preemptions = self._preemption_count
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(self.engine.stream):
            try:
                while True:
                    self.normal_loop()
            except _RequestAllFinished:
                pass
        torch.cuda.synchronize(self.device)

        elapsed = time.time() - start_time
        total_output_tokens = sum(len(self.status_map[i].output_ids) for i in range(len(prompts)))
        logger.info(
            "Rollout done: prompts=%d out_tokens=%d tok/s=%.0f elapsed=%.1fs preemptions=%d",
            len(prompts), total_output_tokens, total_output_tokens / max(elapsed, 1e-6),
            elapsed, self._preemption_count - start_preemptions,
        )
        kv_total = self.cache_manager.total_capacity
        kv_free = len(self.cache_manager.free_slots) * self.cache_manager.page_size
        kv_used = kv_total - kv_free
        logger.debug(
            "loop=%d kv=%.1f%% (%d/%d) running=%d pending=%d preemptions=%d",
            self._loop_count, 100.0 * kv_used / max(kv_total, 1), kv_used, kv_total,
            len(self.decode_manager.running_reqs), len(self.prefill_manager.pending_list),
            self._preemption_count,
        )

        results: List[Dict[str, Any]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            _trace(
                f"final_result uid={i} token_ids_head={status.output_ids[:16]} "
                f"text_preview={output_text[:80]!r}"
            )
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results

    def flush_cache(self) -> None:
        """Reset cache state for next rollout session."""
        self.cache_manager.free_slots = (
            torch.arange(self.engine.num_pages, dtype=torch.int32, device=self.device)
            * self.cache_manager.page_size
        )
        from liteopd.inference.kvcache import create_prefix_cache
        self.cache_manager.prefix_cache = create_prefix_cache(
            device=self.device, type="radix"
        )
        self.table_manager._free_slots = list(range(self.table_manager._max_running_reqs))
        # Zero page_table for real request slots, but preserve the dummy request's entry
        dummy_idx = self.engine.dummy_req.table_idx
        self.table_manager.page_table[:dummy_idx].zero_()
        self.decode_manager.running_reqs.clear()
        self.finished_reqs.clear()


class _RequestStatus:
    __slots__ = ("uid", "input_ids", "output_ids")

    def __init__(self, uid: int, input_ids: List[int], output_ids: List[int]):
        self.uid = uid
        self.input_ids = input_ids
        self.output_ids = output_ids
