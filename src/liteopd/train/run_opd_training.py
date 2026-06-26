from __future__ import annotations

import argparse
import faulthandler
import math
from collections import deque
import json
import os
import random
import signal
import traceback
import time
from typing import List
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.data import load_examples_from_path, summarize_train_dataset_sources, limit_examples, split_validation_examples

from liteopd.eval.scoring import (
    compute_accuracy_with_rollout_client_batched as _compute_accuracy_batched,
    compute_named_accuracies_with_rollout_client_batched as _compute_named_accuracies_batched,
    distributed_eval as _distributed_eval,
)
from liteopd.losses import chunk_loss_from_hidden_chunk, chunked_entropy_from_hidden, chunked_kl_from_hidden, chunked_kl_from_logits, distillation_loss, sft_loss_from_hidden_chunk
from liteopd.train.config import load_train_config
from liteopd.train.logging import JsonlLogger
from liteopd.train.packing import pack_sequences


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


def build_prompt(tokenizer, question: str, chat_template_kwargs: dict | None = None) -> str:
    messages = build_messages(question)
    kwargs = chat_template_kwargs or {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def get_prompt_text(example: dict) -> str:
    for key in ("question", "problem", "prompt"):
        value = example.get(key)
        if value:
            return value
    raise ValueError(f"example missing prompt field among question/problem/prompt: {example.keys()}")


def get_gold_text(example: dict) -> str:
    for key in ("answer", "solution", "canonical_solution"):
        value = example.get(key)
        if value:
            return value
    raise ValueError(f"example missing answer field among answer/solution/canonical_solution: {example.keys()}")


def get_gold_prefix_text(example: dict, gold_prefix_field: str | None, gold_prefix_max_tokens: int | None, tokenizer) -> str:
    """Return the gold-prefix text for *example*, or an empty string.

    When *gold_prefix_field* is None hybrid mode is disabled and the function
    always returns ``""``.  When the field is present the text is truncated to
    *gold_prefix_max_tokens* tokens (approximate; tokenised without special
    tokens) before being returned.
    """
    if not gold_prefix_field:
        return ""
    text = example.get(gold_prefix_field, "") or ""
    text = str(text).strip()
    if not text:
        return ""
    if gold_prefix_max_tokens is not None:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > gold_prefix_max_tokens:
            # Decode the truncated token sequence back to text
            text = tokenizer.decode(ids[:gold_prefix_max_tokens], skip_special_tokens=False)
    return text


def build_rollout_prompt_with_gold(
    tokenizer,
    question: str,
    gold_prefix: str,
    chat_template_kwargs: dict | None = None,
) -> str:
    """Build the rollout prompt string for hybrid SFT+OPD training.

    Embeds *gold_prefix* as the beginning of the assistant turn so that the
    rollout engine generates only the *continuation* after the gold prefix.
    Uses ``continue_final_message=True`` so the tokeniser does not close the
    assistant block, allowing seamless continuation.

    For samples where *gold_prefix* is empty this degrades to the standard
    ``build_prompt`` path (``add_generation_prompt=True``).
    """
    if not gold_prefix:
        return build_prompt(tokenizer, question, chat_template_kwargs)
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": gold_prefix},
    ]
    kwargs = dict(chat_template_kwargs or {})
    # continue_final_message=True: keep the assistant turn open so the model
    # continues generating from the end of gold_prefix.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
        **kwargs,
    )


def chat_template_kwargs_for_model(model_name: str) -> dict | None:
    if "Qwen3" in model_name:
        return {"enable_thinking": True}
    if "Qwen2.5" in model_name:
        return None
    return None


def rollout_max_concurrency(message_count: int) -> int:
    return max(1, min(message_count, 1024))


def _get_vocab_size(model) -> int:
    if hasattr(model, "module"):
        model = model.module
    cfg = model.config
    if hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    if hasattr(cfg, "text_config"):
        return cfg.text_config.vocab_size
    return model.lm_head.out_features


def should_slice_teacher_logits(student_vocab_size: int, teacher_vocab_size: int) -> bool:
    return teacher_vocab_size > student_vocab_size


def capture_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "sys_used_gb": (total - free) / (1024 ** 3),
        "sys_free_gb": free / (1024 ** 3),
    }


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def log_if_rank0(logger: JsonlLogger, rank: int, record: dict) -> None:
    if rank == 0:
        logger.log(record)


def make_debug_event_logger(debug_logger: JsonlLogger, device: torch.device, rank: int, local_rank: int, world_size: int):
    def debug_event(stage: str, **fields) -> None:
        record = {
            "phase": "debug",
            "time": time.time(),
            "stage": stage,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            **capture_memory_stats(device),
            **fields,
        }
        debug_logger.log(record)

    return debug_event


def all_reduce_param_grads(params: list[torch.nn.Parameter]) -> None:
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return
    world_size = dist.get_world_size()
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def compute_accuracy_with_rollout_client(rollout_client, examples, max_new_tokens: int = 32, top_p: float = 1.0, temperature: float = 0.0, top_k: int | None = None, max_concurrency: int | None = None) -> float:
    return _compute_accuracy_batched(
        rollout_client, examples,
        build_messages=build_messages, get_prompt_text=get_prompt_text,
        get_gold_text=get_gold_text, chat_template_kwargs_for_model=chat_template_kwargs_for_model,
        max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        top_k=top_k, max_concurrency=max_concurrency,
    )


def compute_accuracy_with_rollout_client_batched(rollout_client, examples, max_new_tokens: int = 32, top_p: float = 1.0, temperature: float = 0.0, top_k: int | None = None, max_concurrency: int | None = None, debug_event=None, debug_label: str | None = None) -> float:
    return _compute_accuracy_batched(
        rollout_client, examples,
        build_messages=build_messages, get_prompt_text=get_prompt_text,
        get_gold_text=get_gold_text, chat_template_kwargs_for_model=chat_template_kwargs_for_model,
        max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        top_k=top_k, max_concurrency=max_concurrency,
        debug_event=debug_event, debug_label=debug_label,
    )


def compute_named_accuracies_with_rollout_client_batched(rollout_client, named_examples, max_new_tokens: int = 32, top_p: float = 1.0, temperature: float = 0.0, top_k: int | None = None, max_concurrency: int | None = None, debug_event=None, debug_label: str | None = None):
    return _compute_named_accuracies_batched(
        rollout_client, named_examples,
        build_messages=build_messages, get_prompt_text=get_prompt_text,
        get_gold_text=get_gold_text, chat_template_kwargs_for_model=chat_template_kwargs_for_model,
        max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        top_k=top_k, max_concurrency=max_concurrency,
        debug_event=debug_event, debug_label=debug_label,
    )


def distributed_eval(rollout_client, named_examples, distributed, rank, world_size, device, max_new_tokens, top_p, temperature, top_k, max_concurrency, debug_event, debug_label):
    return _distributed_eval(
        rollout_client, named_examples, distributed, rank, world_size, device,
        build_messages=build_messages, get_prompt_text=get_prompt_text,
        get_gold_text=get_gold_text, chat_template_kwargs_for_model=chat_template_kwargs_for_model,
        max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        top_k=top_k, max_concurrency=max_concurrency,
        debug_event=debug_event, debug_label=debug_label,
    )


def refresh_weights(
    rollout_client,
    rank: int,
    logger,
    step: int,
    phase: str,
) -> float:
    elapsed = 0.0
    refresh_on_all_ranks = bool(getattr(rollout_client, "refresh_on_all_ranks", False))
    if rank == 0 or refresh_on_all_ranks:
        start = time.time()
        rollout_client.refresh_from_model()
        elapsed = time.time() - start
        if rank == 0 and phase != "refresh_after_step":
            logger.log({"phase": phase, "time": time.time(), "step": step, "refresh_seconds": elapsed})
    return elapsed


def batch_rollout_and_loss_with_client(
    rollout_client,
    student,
    teacher,
    tokenizer,
    prompts: List[str],
    messages_batch: List[list[dict]],
    loss_name: str,
    device,
    top_p: float,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
    logger=None,
    step: int | None = None,
    profile_memory: bool = False,
    do_backward: bool = False,
    kl_backward_mode: str = "chunk",
    sync_grads: bool = False,
    responses: List[str] | None = None,
    debug_event=None,
    slice_teacher_logits_to_student: bool = False,
    timing_stats: dict | None = None,
    max_pack_tokens: int = 32768,
    # Hybrid SFT+OPD parameters
    gold_prefixes: List[str] | None = None,
    sft_loss_weight: float = 1.0,
    # Output dict for auxiliary metrics (e.g. student_entropy).
    # Keys are written only when this argument is not None.
    metrics_out: dict | None = None,
) -> torch.Tensor:
    """Rollout, pack, forward, and backward for one training step.

    When *gold_prefixes* is provided (hybrid SFT+OPD mode), each sample is
    split into two regions:

    * **Gold-prefix region** — the gold tokens embedded inside the prompt
      (before the student-generated continuation).  Loss: cross-entropy
      against the gold token IDs, weighted by *sft_loss_weight*.
    * **OPD region** — the student-generated continuation.  Loss: KL
      divergence against the teacher (same as standard OPD).

    Both regions contribute to the same backward pass; the two-stage mode
    accumulates their gradients into ``packed_grad_accum`` before the single
    Stage-2 backbone backward, leaving the backward interface unchanged.
    """
    if responses is None:
        if debug_event is not None:
            debug_event("micro_rollout_generate_start", step=step, sample_count=len(messages_batch), max_new_tokens=max_new_tokens)
        rollout_start = time.perf_counter()
        # Hybrid mode: prompts already contain the gold prefix; use
        # generate_from_prompts to skip a redundant apply_chat_template call.
        if gold_prefixes is not None:
            responses = rollout_client.generate_from_prompts(
                prompts,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_concurrency=len(prompts),
            )
        else:
            responses = rollout_client.generate_messages(
                messages_batch,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                chat_template_kwargs=chat_template_kwargs_for_model(rollout_client.model),
                max_concurrency=len(messages_batch),
            )
        if debug_event is not None:
            debug_event("micro_rollout_generate_end", step=step, response_count=len(responses))
        if dist.is_initialized() and dist.get_world_size() > 1:
            if debug_event is not None:
                debug_event("micro_rollout_barrier_start", step=step)
            dist.barrier()
            if debug_event is not None:
                debug_event("micro_rollout_barrier_end", step=step)
        if timing_stats is not None:
            timing_stats["rollout_seconds"] = timing_stats.get("rollout_seconds", 0.0) + (time.perf_counter() - rollout_start)

    student_ddp = student if hasattr(student, "require_backward_grad_sync") else None
    model_for_forward = student.module if hasattr(student, "module") else student
    lm_head_params = [param for param in model_for_forward.lm_head.parameters() if param.requires_grad]
    loss_values = []
    # Entropy tracking: list of (mean_entropy_nats, opd_token_count) per sample.
    # Used to compute a token-count-weighted mean entropy across the full step.
    entropy_values: list[tuple[float, int]] = [] if metrics_out is not None else []
    if profile_memory and logger is not None and step is not None:
        logger.log({"phase": "memory_profile", "time": time.time(), "step": step, "stage": "after_rollout_generation", **capture_memory_stats(device)})

    student_forward_seconds = 0.0
    student_backward_seconds = 0.0
    pack_sequences_start = time.perf_counter()
    packed_batches = pack_sequences(tokenizer, prompts, responses, device, max_pack_tokens, gold_prefixes=gold_prefixes)
    student_forward_seconds += time.perf_counter() - pack_sequences_start

    total_response_tokens = sum(sum(pb.response_token_counts) for pb in packed_batches)
    for packed in packed_batches:
        is_last_packed = packed is packed_batches[-1]

        teacher_forward_start = time.perf_counter()
        sync_cuda(device)
        with torch.inference_mode():
            teacher_hidden_packed = teacher.model(packed.input_ids, position_ids=packed.position_ids, use_cache=False).last_hidden_state
        sync_cuda(device)
        if timing_stats is not None:
            timing_stats["teacher_prefill_seconds"] = timing_stats.get("teacher_prefill_seconds", 0.0) + (time.perf_counter() - teacher_forward_start)

        student_forward_start = time.perf_counter()
        sync_cuda(device)
        student_hidden_packed = model_for_forward.model(packed.input_ids, position_ids=packed.position_ids, use_cache=False).last_hidden_state
        sync_cuda(device)
        student_forward_seconds += time.perf_counter() - student_forward_start

        packed_grad_accum = torch.zeros_like(student_hidden_packed) if (do_backward and kl_backward_mode == "two_stage") else None
        offset = 0
        for sample_index, (seq_len, prompt_len, response_tokens) in enumerate(
            zip(packed.seq_lengths, packed.prompt_lengths, packed.response_token_counts)
        ):
            gold_len     = packed.gold_prefix_lengths[sample_index]
            is_hybrid    = gold_len > 0

            # ── Region boundaries in the packed sequence ──────────────────
            #
            # Layout of input_ids for a hybrid sample:
            #   [base_prompt | gold_prefix | continuation]
            #    <──────── prompt_len ────────> <opd_tokens>
            #
            # hidden[i] predicts input_ids[i+1], so:
            #   SFT targets: input_ids[sft_start+1 : sft_end+1] = gold tokens
            #   OPD region:  hidden[response_start : response_end]
            #
            response_start = offset + prompt_len - 1          # start of OPD region (unchanged)
            response_end   = offset + seq_len                  # end of OPD region   (unchanged)
            sft_start      = response_start - gold_len         # start of SFT region (= response_start when no gold)
            # sft_end == response_start (SFT and OPD are adjacent, no overlap)

            # OPD token count (continuation only; unchanged from pure-OPD)
            opd_tokens = response_end - response_start
            # Total supervised tokens for intra-sample normalisation
            total_tokens = gold_len + opd_tokens

            # Combined hidden slice: covers SFT region + OPD region
            combined_start = sft_start           # == response_start when no gold
            combined_end   = response_end

            is_last_sample = sample_index == len(packed.seq_lengths) - 1

            # Teacher hidden: OPD region only (not needed for SFT)
            teacher_hidden_slice = teacher_hidden_packed[0, response_start:response_end, :].unsqueeze(0)

            # ── two_stage backward ────────────────────────────────────────
            if kl_backward_mode == "two_stage":
                # Detach the combined region and treat it as a single leaf so
                # that Stage-1 (lm_head + SFT/KL grad computation) and
                # Stage-2 (backbone backward) are completely separated.
                hidden_leaf       = student_hidden_packed[0, combined_start:combined_end, :].unsqueeze(0).detach().requires_grad_(True)
                hidden_grad_accum = torch.zeros_like(hidden_leaf)
                weighted_loss_value = 0.0

                # ── Stage-1a: SFT (gold-prefix CE) ────────────────────────
                if is_hybrid:
                    sft_chunk   = hidden_leaf[:, :gold_len, :]                                         # (1, gold_len, H)
                    target_ids  = packed.input_ids[0, combined_start + 1 : combined_start + gold_len + 1]   # (gold_len,)
                    sft_loss    = sft_loss_from_hidden_chunk(sft_chunk, target_ids, model_for_forward.lm_head)
                    sft_scale   = (gold_len / total_tokens) * sft_loss_weight
                    weighted_loss_value += float((sft_loss * gold_len / total_tokens).detach().cpu())
                    grad_scale_sft = (sft_scale * response_tokens / total_response_tokens) if do_backward else sft_scale
                    sft_grads = torch.autograd.grad(
                        sft_loss * grad_scale_sft, [hidden_leaf, *lm_head_params],
                        retain_graph=False, allow_unused=True,
                    )
                    if sft_grads[0] is not None:
                        hidden_grad_accum.add_(sft_grads[0])
                    for param, grad in zip(lm_head_params, sft_grads[1:]):
                        if grad is not None:
                            if param.grad is None:
                                param.grad = grad.detach().clone()
                            else:
                                param.grad.add_(grad.detach())

                # ── Stage-1b: OPD (chunked KL) ────────────────────────────
                # Index into hidden_leaf with gold_len offset so that both
                # SFT and OPD gradients are accumulated onto the same leaf.
                opd_chunk_tokens = max(1, math.ceil(opd_tokens / 8))
                opd_chunk_ranges = [
                    (s, min(opd_tokens, s + opd_chunk_tokens))
                    for s in range(0, opd_tokens, opd_chunk_tokens)
                ]
                for chunk_index, (start, end) in enumerate(opd_chunk_ranges):
                    token_count      = end - start
                    hidden_leaf_chunk = hidden_leaf[:, gold_len + start : gold_len + end, :]
                    kl_loss = chunk_loss_from_hidden_chunk(
                        hidden_leaf_chunk, teacher_hidden_slice[:, start:end, :],
                        model_for_forward.lm_head, teacher.lm_head,
                        loss_name=loss_name, slice_teacher_logits_to_student=slice_teacher_logits_to_student,
                    )
                    scale = token_count / total_tokens
                    weighted_loss_value += float((kl_loss * scale).detach().cpu())
                    grad_scale = (scale * response_tokens / total_response_tokens) if do_backward else scale
                    kl_grads = torch.autograd.grad(
                        kl_loss * grad_scale, [hidden_leaf, *lm_head_params],
                        retain_graph=False, allow_unused=True,
                    )
                    if kl_grads[0] is not None:
                        hidden_grad_accum.add_(kl_grads[0])
                    for param, grad in zip(lm_head_params, kl_grads[1:]):
                        if grad is not None:
                            if param.grad is None:
                                param.grad = grad.detach().clone()
                            else:
                                param.grad.add_(grad.detach())

                # Scatter the combined gradient back into packed_grad_accum
                if do_backward:
                    packed_grad_accum[0, combined_start:combined_end, :] = hidden_grad_accum.squeeze(0)

            # ── chunk backward ────────────────────────────────────────────
            elif kl_backward_mode == "chunk":
                weighted_loss_value = 0.0
                opd_chunk_tokens = max(1, math.ceil(opd_tokens / 8))
                opd_chunk_ranges = [
                    (s, min(opd_tokens, s + opd_chunk_tokens))
                    for s in range(0, opd_tokens, opd_chunk_tokens)
                ]
                has_opd = len(opd_chunk_ranges) > 0

                # SFT region: single backward; must retain graph if OPD follows
                if is_hybrid:
                    sft_hidden = student_hidden_packed[0, sft_start:response_start, :].unsqueeze(0)
                    target_ids = packed.input_ids[0, sft_start + 1 : response_start + 1]
                    sft_loss   = sft_loss_from_hidden_chunk(sft_hidden, target_ids, model_for_forward.lm_head)
                    sft_scale  = (gold_len / total_tokens) * sft_loss_weight
                    weighted_loss_value += float((sft_loss * gold_len / total_tokens).detach().cpu())
                    if do_backward:
                        student_backward_start = time.perf_counter()
                        # retain_graph=True because the OPD region shares the same
                        # student_hidden_packed computation graph
                        (sft_loss * sft_scale * response_tokens / total_response_tokens).backward(retain_graph=True)
                        student_backward_seconds += time.perf_counter() - student_backward_start

                # OPD region: chunked as in standard OPD
                opd_hidden_slice = student_hidden_packed[0, response_start:response_end, :].unsqueeze(0)
                for chunk_index, (start, end) in enumerate(opd_chunk_ranges):
                    token_count = end - start
                    kl_loss = chunk_loss_from_hidden_chunk(
                        opd_hidden_slice[:, start:end, :],
                        teacher_hidden_slice[:, start:end, :],
                        model_for_forward.lm_head, teacher.lm_head,
                        loss_name=loss_name, slice_teacher_logits_to_student=slice_teacher_logits_to_student,
                    )
                    scale = token_count / total_tokens
                    weighted_loss_value += float((kl_loss * scale).detach().cpu())
                    if do_backward:
                        is_last_chunk = is_last_packed and is_last_sample and chunk_index == len(opd_chunk_ranges) - 1
                        student_backward_start = time.perf_counter()
                        (kl_loss * scale * response_tokens / total_response_tokens).backward(retain_graph=not is_last_chunk)
                        student_backward_seconds += time.perf_counter() - student_backward_start
                    del kl_loss

            # ── sample backward ───────────────────────────────────────────
            else:
                opd_chunk_tokens = max(1, math.ceil(opd_tokens / 8))
                opd_hidden_slice = student_hidden_packed[0, response_start:response_end, :].unsqueeze(0)
                kl_loss = chunked_kl_from_hidden(
                    opd_hidden_slice, teacher_hidden_slice,
                    model_for_forward.lm_head, teacher.lm_head,
                    loss_name=loss_name, chunk_tokens=opd_chunk_tokens,
                    slice_teacher_logits_to_student=slice_teacher_logits_to_student,
                )
                # Scale KL loss by its token fraction
                opd_fraction = opd_tokens / total_tokens if total_tokens > 0 else 1.0
                weighted_loss_value = float(kl_loss.detach().cpu()) * opd_fraction

                if is_hybrid:
                    sft_hidden = student_hidden_packed[0, sft_start:response_start, :].unsqueeze(0)
                    target_ids = packed.input_ids[0, sft_start + 1 : response_start + 1]
                    sft_loss   = sft_loss_from_hidden_chunk(sft_hidden, target_ids, model_for_forward.lm_head)
                    sft_fraction = (gold_len / total_tokens) * sft_loss_weight if total_tokens > 0 else 0.0
                    weighted_loss_value += float(sft_loss.detach().cpu()) * sft_fraction
                    # Combined backward for SFT + OPD in one pass
                    if do_backward:
                        student_backward_start = time.perf_counter()
                        combined = (
                            kl_loss * opd_fraction + sft_loss * sft_fraction
                        ) * response_tokens / total_response_tokens
                        combined.backward()
                        student_backward_seconds += time.perf_counter() - student_backward_start
                else:
                    if do_backward:
                        student_backward_start = time.perf_counter()
                        (kl_loss * response_tokens / total_response_tokens).backward()
                        student_backward_seconds += time.perf_counter() - student_backward_start

            loss_values.append(weighted_loss_value * response_tokens / total_response_tokens)

            # ── Student output entropy (OPD region) ───────────────────────
            # Computed from the same student hidden states used for the KL
            # loss, but inside torch.no_grad() so it does not affect the
            # backward graph or memory budget.
            # Entropy of the SFT (gold-prefix) region is intentionally
            # excluded: the student is given the gold text, so its
            # distribution there is less informative than on its own rollout.
            if metrics_out is not None:
                with torch.no_grad():
                    opd_hidden = student_hidden_packed[0, response_start:response_end, :].unsqueeze(0)
                    sample_entropy = chunked_entropy_from_hidden(
                        opd_hidden,
                        model_for_forward.lm_head,
                        chunk_tokens=max(1, opd_tokens // 8),
                    )
                    entropy_values.append((float(sample_entropy.cpu()), opd_tokens))

            offset += seq_len

        if do_backward and kl_backward_mode == "two_stage":
            student_backward_start = time.perf_counter()
            torch.autograd.backward(student_hidden_packed, grad_tensors=packed_grad_accum)
            student_backward_seconds += time.perf_counter() - student_backward_start

        del student_hidden_packed, teacher_hidden_packed, packed_grad_accum

    # DDP gradient sync: since forward bypasses the DDP wrapper (we call
    # model_for_forward.model(...) directly), DDP's built-in reducer never
    # triggers. We all-reduce all gradients manually after the last packed batch.
    if do_backward and sync_grads and student_ddp is not None:
        all_params = [p for p in student_ddp.module.parameters() if p.requires_grad]
        all_reduce_param_grads(all_params)

    if timing_stats is not None:
        timing_stats["student_forward_seconds"] = timing_stats.get("student_forward_seconds", 0.0) + student_forward_seconds
        timing_stats["student_backward_seconds"] = timing_stats.get("student_backward_seconds", 0.0) + student_backward_seconds
    if debug_event is not None:
        debug_event(
            "loss_timing_breakdown",
            step=step,
            packed_batch_count=len(packed_batches),
            total_response_tokens=total_response_tokens,
            student_forward_seconds=student_forward_seconds,
            student_backward_seconds=student_backward_seconds,
        )
    if not loss_values:
        raise ValueError("no sample losses were produced in batch_rollout_and_loss_with_client")

    if metrics_out is not None and entropy_values:
        # Token-count-weighted mean entropy across all OPD positions in the step.
        total_opd_tokens = sum(t for _, t in entropy_values)
        metrics_out["student_entropy"] = (
            sum(e * t for e, t in entropy_values) / total_opd_tokens
            if total_opd_tokens > 0 else 0.0
        )

    return torch.tensor(sum(loss_values) / len(loss_values), device=device)


def has_bad_gradients(model) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True
    return False


def any_rank_has_bad_gradients(model, distributed: bool, device: torch.device) -> bool:
    local_bad = 1 if has_bad_gradients(model) else 0
    if distributed and dist.is_initialized():
        flag = torch.tensor([local_bad], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    return bool(local_bad)


def save_checkpoint(model, output_dir: Path, step: int, rank: int, tokenizer) -> None:
    if rank != 0:
        return
    target = output_dir / "checkpoints" / f"step_{step}"
    module = model.module if hasattr(model, "module") else model
    from liteopd.train.fused_model import unfuse_state_dict
    state_dict = unfuse_state_dict(module)
    target.mkdir(parents=True, exist_ok=True)
    module.save_pretrained(
        target, state_dict=state_dict, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(target)


def extract_accuracy(metrics: dict[str, dict[str, float | int]], split_name: str) -> float | None:
    split_metrics = metrics.get(split_name)
    if split_metrics is None:
        return None
    return float(split_metrics["accuracy"])


def build_eval_log_record(
    phase: str,
    step: int,
    metrics: dict[str, dict[str, float | int]],
    loss_name: str,
    test_eval_examples: int,
    validation_eval_examples: int,
    examples_per_minute: float | None = None,
) -> dict:
    record = {
        "phase": phase,
        "time": time.time(),
        "step": step,
        "test_accuracy": extract_accuracy(metrics, "test"),
        "validation_accuracy": extract_accuracy(metrics, "validation"),
        "test_eval_examples": test_eval_examples,
        "validation_eval_examples": validation_eval_examples,
        "loss_name": loss_name,
    }
    if examples_per_minute is not None:
        record["examples_per_minute"] = examples_per_minute
    avg_tokens = metrics.get("test", {}).get("avg_gen_tokens") or metrics.get("validation", {}).get("avg_gen_tokens")
    if avg_tokens is not None:
        record["avg_gen_tokens"] = avg_tokens
    return record


class _TrainingSignalGuard:
    def __init__(self, on_signal=None) -> None:
        self._on_signal = on_signal
        self._previous_handlers: dict[int, object] = {}

    def __enter__(self):
        self._previous_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _handler(sig, frame):
            if self._on_signal is not None:
                self._on_signal(sig)
            if sig == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + sig)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.signal(signal.SIGINT, self._previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self._previous_handlers[signal.SIGTERM])
        return False


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    debug_event = None
    distributed = False
    rollout_client = None
    cleanup_started = False

    def _cleanup_runtime() -> None:
        nonlocal cleanup_started, rollout_client, debug_event, distributed
        if cleanup_started:
            return
        cleanup_started = True
        if debug_event is not None:
            debug_event("runtime_cleanup_start")
        if rollout_client is not None and hasattr(rollout_client, "shutdown"):
            try:
                rollout_client.shutdown()
            except BaseException:
                if debug_event is not None:
                    debug_event("runtime_cleanup_rollout_shutdown_failed", traceback=traceback.format_exc())
        if distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                if debug_event is not None:
                    debug_event("runtime_cleanup_destroy_process_group_failed", traceback=traceback.format_exc())
        if debug_event is not None:
            debug_event("runtime_cleanup_end")

    def _on_signal(sig: int) -> None:
        if debug_event is not None:
            debug_event("signal_received", signal=sig)
        _cleanup_runtime()

    try:
        with _TrainingSignalGuard(on_signal=_on_signal):
            cfg = load_train_config(args.config)
            output_dir_override = os.environ.get("OPD_OUTPUT_DIR")
            if output_dir_override:
                cfg.output_dir = output_dir_override
            else:
                from datetime import datetime
                cfg.output_dir = str(Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            distributed, rank, local_rank, world_size = init_distributed()
            logger = JsonlLogger(out_dir / f"train_log.rank{rank}.jsonl")
            debug_logger = JsonlLogger(out_dir / f"train_debug.rank{rank}.jsonl")
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            debug_event = make_debug_event_logger(debug_logger, device, rank, local_rank, world_size)
            debug_event("main_start", config_path=args.config, output_dir=str(out_dir), distributed=distributed)

            train_examples, test_examples = load_examples_from_path(Path(cfg.train_dataset), limit=cfg.train_subset_size)
            if cfg.eval_dataset:
                _, explicit_eval_examples = load_examples_from_path(Path(cfg.eval_dataset), limit=cfg.eval_subset_size)
                if explicit_eval_examples:
                    test_examples = explicit_eval_examples
                else:
                    eval_rows, _ = load_examples_from_path(Path(cfg.eval_dataset), limit=cfg.eval_subset_size)
                    test_examples = eval_rows
            random.seed(42)
            random.shuffle(train_examples)
            train_examples = limit_examples(train_examples, cfg.train_subset_size)
            test_examples = limit_examples(test_examples, cfg.eval_subset_size)
            train_dataset_source_counts = summarize_train_dataset_sources(train_examples)
            train_examples, validation_examples = split_validation_examples(train_examples, cfg.validation_subset_size)
            if not train_examples:
                raise ValueError("train_examples is empty after loading and slicing")
            if not test_examples:
                raise ValueError("test_examples is empty; provide a valid eval dataset or train dataset with a test split")

            log_if_rank0(logger, rank, {"phase": "config", "time": time.time(), "global_batch_size": cfg.global_batch_size, "generation_batch_size": cfg.generation_batch_size, "world_size": world_size, "max_total_tokens": cfg.max_total_tokens, "num_epochs": cfg.num_epochs, "save_every_steps": cfg.save_every_steps, "eval_every_steps": cfg.eval_every_steps, "skip_eval": cfg.skip_eval, "profile_memory": cfg.profile_memory, "kl_backward_mode": cfg.kl_backward_mode, "validation_subset_size": cfg.validation_subset_size, "test_eval_examples": len(test_examples), "validation_eval_examples": len(validation_examples)})

            tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            def _filter_by_prompt_length(examples: list[dict], max_len: int) -> list[dict]:
                chat_kwargs = chat_template_kwargs_for_model(cfg.student_model) or {}
                kept = []
                for ex in examples:
                    prompt = tokenizer.apply_chat_template(
                        build_messages(get_prompt_text(ex)), tokenize=False, add_generation_prompt=True, **chat_kwargs
                    )
                    if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_len:
                        kept.append(ex)
                return kept

            if cfg.max_prompt_length > 0:
                train_examples = _filter_by_prompt_length(train_examples, cfg.max_prompt_length)
                test_examples = _filter_by_prompt_length(test_examples, cfg.max_prompt_length)
                validation_examples = _filter_by_prompt_length(validation_examples, cfg.max_prompt_length)

            student = AutoModelForCausalLM.from_pretrained(
                cfg.student_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
                trust_remote_code=True,
            ).to(device)
            from liteopd.train.fused_model import fuse_model_projections
            fuse_model_projections(student)
            if cfg.enable_gradient_checkpointing:
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(student, "enable_input_require_grads"):
                    student.enable_input_require_grads()
            if distributed:
                if cfg.distributed_strategy == "zero2":
                    pass  # ZeRO2Optimizer manages gradient sync, no wrapper needed
                else:
                    # DDP wrapper is used only as a marker for distributed mode.
                    # Forward bypasses DDP (we call model_for_forward.model(...) directly)
                    # and gradient sync is handled manually via all_reduce_param_grads.
                    student = DDP(
                        student,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=False,
                        static_graph=False,
                    )

            from liteopd.runtime.coordinator import RuntimeCoordinator
            hf_model = student.module if hasattr(student, "module") else student

            # ZeRO-2: create optimizer before RuntimeCoordinator so params become
            # flat buffer views before share_weights() establishes tensor.set_() links.
            if distributed and cfg.distributed_strategy == "zero2":
                from liteopd.train.zero2 import ZeRO2Optimizer
                optimizer = ZeRO2Optimizer(
                    params=[p for p in hf_model.parameters() if p.requires_grad],
                    lr=cfg.learning_rate,
                    rank=rank,
                    world_size=world_size,
                )
                optimizer.init_optimizer_state()
            else:
                optimizer = None  # created later

            teacher = AutoModelForCausalLM.from_pretrained(
                cfg.teacher_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
                trust_remote_code=True,
            )
            if not cfg.offload_teacher:
                teacher = teacher.to(device)
            teacher.eval()
            if cfg.compile_teacher and cfg.offload_teacher:
                if rank == 0:
                    print("[WARN] compile_teacher disabled: incompatible with offload_teacher (device migration triggers recompilation)")
                cfg.compile_teacher = False
            if cfg.compile_teacher:
                teacher.model = torch.compile(teacher.model)

            if optimizer is not None:
                optimizer.release_grad_buffer()
            torch.cuda.empty_cache()

            coordinator = RuntimeCoordinator(
                student_model=hf_model,
                model_path=cfg.student_model,
                device=device,
                tokenizer=tokenizer,
                memory_ratio=cfg.generation_mem_fraction_static,
                max_running_req=cfg.generation_max_running_req,
                cuda_graph_max_bs=cfg.generation_cuda_graph_max_bs,
                page_size=cfg.generation_page_size,
                attention_backend=cfg.generation_attention_backend,
                admission_reserve_tokens=cfg.generation_admission_reserve_tokens,
                max_preemptions_per_req=cfg.generation_max_preemptions_per_req,
                use_vmm=cfg.generation_use_vmm,
                log_dir=str(out_dir),
                rank=rank,
            )
            rollout_client = coordinator.rollout_runtime

            train_examples = train_examples[rank::world_size] if distributed else train_examples
            if distributed and dist.is_initialized():
                shard_len = torch.tensor([len(train_examples)], device=device, dtype=torch.int64)
                dist.all_reduce(shard_len, op=dist.ReduceOp.MIN)
                min_shard_len = int(shard_len.item())
                if len(train_examples) > min_shard_len:
                    train_examples = train_examples[:min_shard_len]
            if not train_examples:
                raise ValueError(f"rank {rank} received an empty train shard; reduce world_size or increase train_subset_size")

            local_batches = max((len(train_examples) + cfg.generation_batch_size - 1) // cfg.generation_batch_size, 1)
            max_local_batches = local_batches
            if distributed and dist.is_initialized():
                batch_tensor = torch.tensor([local_batches], device=device, dtype=torch.int64)
                dist.all_reduce(batch_tensor, op=dist.ReduceOp.MAX)
                max_local_batches = int(batch_tensor.item())
            total_steps = min(cfg.max_steps, max_local_batches * cfg.num_epochs)

            if optimizer is None:
                optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)

            warmup_steps = int(total_steps * cfg.warmup_ratio)
            base_lr = cfg.learning_rate

            def _get_lr(current_step: int) -> float:
                if current_step < warmup_steps:
                    return base_lr * current_step / max(warmup_steps, 1)
                if cfg.cosine_annealing:
                    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
                return base_lr

            optimizer.param_groups[0]["lr"] = _get_lr(0)

            recent_loss_values: deque[float] = deque(maxlen=100)
            slice_teacher_logits = should_slice_teacher_logits(_get_vocab_size(student), _get_vocab_size(teacher))
            debug_event("teacher_logits_slice_policy", enabled=slice_teacher_logits, student_vocab=_get_vocab_size(student), teacher_vocab=_get_vocab_size(teacher))

            pre_eval_metrics = None
            if not cfg.skip_eval:
                rollout_client.prepare()
                debug_event("pre_eval_compute_start", step=0, test_examples=len(test_examples), validation_examples=len(validation_examples))
                pre_eval_metrics = distributed_eval(
                    rollout_client, [("test", test_examples), ("validation", validation_examples)],
                    distributed, rank, world_size, device,
                    max_new_tokens=cfg.max_total_tokens, top_p=cfg.top_p, temperature=cfg.temperature,
                    top_k=cfg.generation_top_k, max_concurrency=rollout_max_concurrency(len(test_examples) + len(validation_examples)),
                    debug_event=debug_event, debug_label="pre_eval",
                )
                debug_event("pre_eval_compute_end", step=0, metrics=pre_eval_metrics)
                log_if_rank0(logger, rank, build_eval_log_record("pre_eval", 0, pre_eval_metrics, cfg.loss, len(test_examples), len(validation_examples)))
                rollout_client.release()

            student.train()
            start_time = time.time()
            processed_examples = 0
            for step in range(1, total_steps + 1):
                try:
                    step_t0 = time.perf_counter()
                    step_timing = {
                        "rollout_seconds": 0.0,
                        "teacher_prefill_seconds": 0.0,
                        "student_forward_seconds": 0.0,
                        "student_backward_seconds": 0.0,
                    }
                    debug_event("step_start", step=step)
                    optimizer.zero_grad(set_to_none=True)
                    debug_event("zero_grad_done", step=step)
                    batch_start = ((step - 1) * cfg.generation_batch_size) % len(train_examples)
                    batch = [train_examples[(batch_start + i) % len(train_examples)] for i in range(cfg.generation_batch_size)]
                    prompt_texts = [get_prompt_text(ex) for ex in batch]
                    messages_batch = [build_messages(prompt_text) for prompt_text in prompt_texts]

                    # ── Hybrid SFT+OPD: extract gold prefixes ─────────────
                    # When cfg.gold_prefix_field is set, each sample's prompt
                    # is extended with its gold prefix so the rollout engine
                    # generates only the continuation after it.
                    is_hybrid_step = bool(cfg.gold_prefix_field)
                    if is_hybrid_step:
                        gold_prefixes_batch = [
                            get_gold_prefix_text(ex, cfg.gold_prefix_field, cfg.gold_prefix_max_tokens, tokenizer)
                            for ex in batch
                        ]
                        # Build prompts that embed the gold prefix; these are
                        # passed to generate_from_prompts and to pack_sequences.
                        prompts = [
                            build_rollout_prompt_with_gold(
                                tokenizer, pt, gp,
                                chat_template_kwargs_for_model(cfg.student_model),
                            )
                            for pt, gp in zip(prompt_texts, gold_prefixes_batch)
                        ]
                        # Normalise: treat empty-gold samples as pure OPD
                        gold_prefixes_batch = [gp or "" for gp in gold_prefixes_batch]
                    else:
                        gold_prefixes_batch = None
                        prompts = [
                            build_prompt(tokenizer, pt, chat_template_kwargs_for_model(cfg.student_model))
                            for pt in prompt_texts
                        ]

                    debug_event("batch_ready", step=step, batch_start=batch_start, batch_size=len(batch), prompt_chars=sum(len(p) for p in prompts))
                    debug_event("rollout_generate_start", step=step, sample_count=len(messages_batch), max_concurrency=rollout_max_concurrency(len(messages_batch)))
                    rollout_start = time.perf_counter()
                    sync_cuda(device)
                    if cfg.distributed_strategy == "zero2" and hasattr(optimizer, 'release_grad_buffer'):
                        optimizer.release_grad_buffer()
                        torch.cuda.empty_cache()
                        debug_event("mem_after_grad_buffer_release", step=step)
                    rollout_client.prepare()
                    if is_hybrid_step:
                        # Prompts already contain the gold prefix; skip the
                        # redundant apply_chat_template inside generate_messages.
                        responses = rollout_client.generate_from_prompts(
                            prompts,
                            max_tokens=cfg.max_total_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            top_k=cfg.generation_top_k,
                            max_concurrency=rollout_max_concurrency(len(prompts)),
                        )
                    else:
                        responses = rollout_client.generate_messages(
                            messages_batch,
                            max_tokens=cfg.max_total_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            top_k=cfg.generation_top_k,
                            chat_template_kwargs=chat_template_kwargs_for_model(rollout_client.model),
                            max_concurrency=rollout_max_concurrency(len(messages_batch)),
                        )
                    sync_cuda(device)
                    if distributed and dist.is_initialized():
                        debug_event("rollout_barrier_start", step=step)
                        dist.barrier()
                        debug_event("rollout_barrier_end", step=step)
                    step_timing["rollout_seconds"] += time.perf_counter() - rollout_start
                    debug_event("rollout_generate_end", step=step, response_count=len(responses), response_chars=sum(len(response) for response in responses))
                    rollout_client.release()
                    if cfg.profile_memory:
                        logger.log({"phase": "memory_profile", "time": time.time(), "step": step, "stage": "after_rollout_generation", **capture_memory_stats(device)})
                    micro_losses = []
                    step_metrics: dict = {}
                    # packing mode: pass all prompts/responses at once; pack_sequences handles grouping
                    if cfg.offload_teacher:
                        teacher.to(device)
                    loss = batch_rollout_and_loss_with_client(
                        rollout_client, student, teacher, tokenizer,
                        prompts, messages_batch, cfg.loss, device,
                        top_p=cfg.top_p, temperature=cfg.temperature, top_k=cfg.generation_top_k,
                        max_new_tokens=cfg.max_total_tokens, logger=logger, step=step,
                        profile_memory=cfg.profile_memory, do_backward=True,
                        kl_backward_mode=cfg.kl_backward_mode,
                        sync_grads=True, responses=responses, debug_event=debug_event,
                        slice_teacher_logits_to_student=slice_teacher_logits,
                        timing_stats=step_timing,
                        max_pack_tokens=cfg.max_pack_tokens,
                        gold_prefixes=gold_prefixes_batch,
                        sft_loss_weight=cfg.sft_loss_weight,
                        metrics_out=step_metrics,
                    )
                    if cfg.offload_teacher:
                        teacher.to("cpu")
                        torch.cuda.empty_cache()
                    micro_losses.append(float(loss.detach().cpu()))
                    loss_value = sum(micro_losses) / len(micro_losses)
                    processed_examples += cfg.generation_batch_size * world_size
                    debug_event("step_loss_ready", step=step, loss_value=float(loss_value))
                    debug_event("bad_grad_check_start", step=step)
                    bad_grad_check_start = time.perf_counter()
                    has_bad_grads = any_rank_has_bad_gradients(student, distributed, device)
                    step_timing["student_backward_seconds"] += time.perf_counter() - bad_grad_check_start
                    debug_event("bad_grad_check_end", step=step, has_bad_grads=has_bad_grads)
                    refresh_barrier_accounted = False
                    if has_bad_grads:
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        log_if_rank0(logger, rank, {"phase": "skip_step", "time": time.time(), "step": step, "reason": "non_finite_gradient", "loss_name": cfg.loss})
                        debug_event("skip_step", step=step, reason="non_finite_gradient")
                    else:
                        if cfg.distributed_strategy == "zero2" and hasattr(optimizer, 'prepare_grad_buffer'):
                            optimizer.prepare_grad_buffer()
                            debug_event("mem_after_grad_buffer_prepare", step=step)
                            optimizer.reduce_scatter_grads()
                        debug_event("optimizer_step_start", step=step)
                        optimizer.step()
                        optimizer.param_groups[0]["lr"] = _get_lr(step)
                        debug_event("optimizer_step_end", step=step)
                        debug_event("refresh_after_step_start", step=step)
                        refresh_start = time.perf_counter()
                        refresh_weights(rollout_client, rank, logger, step, "refresh_after_step")
                        if distributed and dist.is_initialized():
                            debug_event("refresh_barrier_start", step=step)
                            dist.barrier()
                            debug_event("refresh_barrier_end", step=step)
                            refresh_barrier_accounted = True
                        _ = time.perf_counter() - refresh_start
                        debug_event("refresh_after_step_end", step=step)
                    if distributed and dist.is_initialized() and not refresh_barrier_accounted:
                        debug_event("train_barrier_start", step=step)
                        dist.barrier()
                        debug_event("train_barrier_end", step=step)
                    step_total_seconds = time.perf_counter() - step_t0
                    residual_seconds = step_total_seconds - step_timing["rollout_seconds"] - step_timing["teacher_prefill_seconds"] - step_timing["student_forward_seconds"] - step_timing["student_backward_seconds"]
                except BaseException:
                    debug_event("step_exception", step=step, traceback=traceback.format_exc())
                    raise

                elapsed_minutes = max((time.time() - start_time) / 60.0, 1e-6)
                examples_per_minute = processed_examples / elapsed_minutes

                recent_loss_values.append(loss_value)
                if step % cfg.log_every == 0:
                    train_log_record = {
                        "phase": "train",
                        "time": time.time(),
                        "step": step,
                        "loss_name": cfg.loss,
                        "loss": loss_value,
                        "loss_avg_100": sum(recent_loss_values) / len(recent_loss_values),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "examples_per_minute": examples_per_minute,
                        "step_total_seconds": step_total_seconds,
                        "rollout_seconds": step_timing["rollout_seconds"],
                        "teacher_prefill_seconds": step_timing["teacher_prefill_seconds"],
                        "student_forward_seconds": step_timing["student_forward_seconds"],
                        "student_backward_seconds": step_timing.get("student_backward_seconds", 0.0),
                        "other_seconds": residual_seconds,
                        "avg_gen_tokens": sum(getattr(rollout_client, 'last_output_token_counts', None) or [len(tokenizer.encode(r, add_special_tokens=False)) for r in responses]) / max(len(responses), 1),
                    }
                    # Auxiliary metrics collected inside batch_rollout_and_loss_with_client
                    if "student_entropy" in step_metrics:
                        train_log_record["student_entropy"] = step_metrics["student_entropy"]
                    log_if_rank0(logger, rank, train_log_record)
                debug_event("step_end", step=step, examples_per_minute=examples_per_minute)

                if step % cfg.save_every_steps == 0:
                    save_checkpoint(student, out_dir, step, rank, tokenizer)

                if not cfg.skip_eval and step % cfg.eval_every_steps == 0:
                    student.eval()
                    optimizer.zero_grad(set_to_none=True)
                    refresh_weights(rollout_client, rank, logger, step, "refresh_before_eval")
                    rollout_client.prepare()
                    debug_event("eval_compute_start", step=step, test_examples=len(test_examples), validation_examples=len(validation_examples))
                    eval_metrics = distributed_eval(
                        rollout_client, [("test", test_examples), ("validation", validation_examples)],
                        distributed, rank, world_size, device,
                        max_new_tokens=cfg.max_total_tokens, top_p=cfg.top_p, temperature=cfg.temperature,
                        top_k=cfg.generation_top_k, max_concurrency=rollout_max_concurrency(len(test_examples) + len(validation_examples)),
                        debug_event=debug_event, debug_label="eval",
                    )
                    debug_event("eval_compute_end", step=step, metrics=eval_metrics)
                    log_if_rank0(logger, rank, build_eval_log_record("eval", step, eval_metrics, cfg.loss, len(test_examples), len(validation_examples), examples_per_minute=examples_per_minute))
                    rollout_client.release()
                    student.train()

            post_eval_metrics = None
            if not cfg.skip_eval:
                optimizer.zero_grad(set_to_none=True)
                refresh_weights(rollout_client, rank, logger, total_steps, "refresh_before_post_eval")
                rollout_client.prepare()
                debug_event("post_eval_compute_start", step=total_steps, test_examples=len(test_examples), validation_examples=len(validation_examples))
                post_eval_metrics = distributed_eval(
                    rollout_client, [("test", test_examples), ("validation", validation_examples)],
                    distributed, rank, world_size, device,
                    max_new_tokens=cfg.max_total_tokens, top_p=cfg.top_p, temperature=cfg.temperature,
                    top_k=cfg.generation_top_k, max_concurrency=rollout_max_concurrency(len(test_examples) + len(validation_examples)),
                    debug_event=debug_event, debug_label="post_eval",
                )
                debug_event("post_eval_compute_end", step=total_steps, metrics=post_eval_metrics)
                log_if_rank0(logger, rank, build_eval_log_record("post_eval", total_steps, post_eval_metrics, cfg.loss, len(test_examples), len(validation_examples), examples_per_minute=examples_per_minute))
                rollout_client.release()
            save_checkpoint(student, out_dir, total_steps, rank, tokenizer)

            if rank == 0:
                debug_event("summary_write_start", step=total_steps)
                with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump({"pre_test_accuracy": extract_accuracy(pre_eval_metrics or {}, "test"), "pre_validation_accuracy": extract_accuracy(pre_eval_metrics or {}, "validation"), "post_test_accuracy": extract_accuracy(post_eval_metrics or {}, "test"), "post_validation_accuracy": extract_accuracy(post_eval_metrics or {}, "validation"), "loss": cfg.loss, "world_size": world_size, "startup_seconds": None, "examples_per_minute": examples_per_minute, "total_steps": total_steps, "skip_eval": cfg.skip_eval, "validation_subset_size": cfg.validation_subset_size, "test_eval_examples": len(test_examples), "validation_eval_examples": len(validation_examples)}, f, ensure_ascii=False, indent=2)
                debug_event("summary_write_end", step=total_steps)

            if distributed and dist.is_initialized():
                debug_event("final_barrier_start", step=total_steps)
                dist.barrier()
                debug_event("final_barrier_end", step=total_steps)
    except BaseException:
        if debug_event is not None:
            debug_event("main_exception", traceback=traceback.format_exc())
        raise
    finally:
        _cleanup_runtime()


if __name__ == "__main__":
    main()
