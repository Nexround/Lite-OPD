from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, List

import torch
from liteopd.inference.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from liteopd.inference.core import Batch


_TRACE_LIMIT = 32
_trace_counter = 0


def _trace_enabled() -> bool:
    return os.environ.get("OPD_DEBUG_TRACE", "0") == "1"


def _trace(message: str) -> None:
    global _trace_counter
    if not _trace_enabled() or _trace_counter >= _TRACE_LIMIT:
        return
    _trace_counter += 1
    print(f"[opd.trace.sample] {message}", flush=True)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    greedy_mask: torch.Tensor | None = None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _apply_temperature(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    return logits.div_(temperatures.unsqueeze(dim=1))


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    if top_p is None:
        if top_k is None:
            return logits
        return _apply_top_k_only(logits, top_k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if top_k is not None:
        top_k_mask = logits_sort.size(1) - top_k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - top_p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)


def _apply_top_k_only(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    no_top_k_mask = top_k == logits.shape[1]
    top_k = top_k.masked_fill(no_top_k_mask, 1)
    max_top_k = top_k.max()
    top_k_index = top_k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, top_k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def _random_sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div_(q).argmax(dim=-1).view(-1)


def _native_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    logits = _apply_top_k_top_p(logits, top_k, top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return _random_sample_from_probs(probs)


def _flashinfer_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    if top_k is None and top_p is None:
        return _native_sample(logits, top_k=None, top_p=None)

    if top_p is None:
        assert top_k is not None
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)


def _should_use_flashinfer_sampler() -> bool:
    return os.environ.get("OPD_USE_FLASHINFER_SAMPLER", "0") == "1"


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        greedy = [p.is_greedy for p in params]
        ts = [1.0 if is_greedy else max(p.temperature, MIN_T) for is_greedy, p in zip(greedy, params)]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        greedy_mask = make_device_tensor(greedy, torch.bool, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, greedy_mask=greedy_mask, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                sampled = torch.argmax(logits, dim=-1)
                if logits.shape[0] > 0:
                    top_vals, top_ids = torch.topk(logits[0].float(), k=min(5, logits.shape[-1]))
                    _trace(
                        f"greedy top_ids={top_ids.tolist()} "
                        f"top_vals={[round(float(v), 4) for v in top_vals.tolist()]} "
                        f"sampled0={int(sampled[0].item())}"
                    )
                return sampled
            greedy_sampled = None
            if args.greedy_mask is not None and bool(args.greedy_mask.any().item()):
                greedy_sampled = torch.argmax(logits, dim=-1)
            processed_logits = logits.float().contiguous()
            if logits.shape[0] > 0:
                top_vals, top_ids = torch.topk(processed_logits[0], k=min(5, processed_logits.shape[-1]))
                _trace(
                    f"pre_temp top_ids={top_ids.tolist()} "
                    f"top_vals={[round(float(v), 4) for v in top_vals.tolist()]} "
                    f"temp0={round(float(args.temperatures[0].item()), 4)}"
                )
            processed_logits = _apply_temperature(processed_logits, args.temperatures)
            if _should_use_flashinfer_sampler():
                random_sampled = _flashinfer_sample(processed_logits, args.top_k, args.top_p)
                backend = "flashinfer"
            else:
                random_sampled = _native_sample(processed_logits, args.top_k, args.top_p)
                backend = "native"
            _trace(
                f"backend={backend} sampled_head={random_sampled[: min(8, random_sampled.numel())].tolist()} "
                f"greedy_mask_any={bool(args.greedy_mask.any().item()) if args.greedy_mask is not None else False}"
            )
            if greedy_sampled is None or args.greedy_mask is None:
                return random_sampled
            return torch.where(args.greedy_mask, greedy_sampled, random_sampled, out=greedy_sampled)
