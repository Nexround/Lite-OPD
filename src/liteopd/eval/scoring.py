"""Evaluation and scoring utilities for OPD."""
from __future__ import annotations

import os
import re
import time
from typing import Any

import torch
import torch.distributed as dist


def extract_number(text: str) -> str | None:
    """Extract the last number from text."""
    matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a model response."""
    if not text or not text.strip():
        return ""
    for pattern in [r"Final answer\s*:\s*(.+)$", r"Answer\s*:\s*(.+)$"]:
        matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
        if matches:
            return matches[-1].strip()
    boxed_matches = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r'\boxed{':
            start = i + 7
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                boxed_matches.append(text[start:j-1])
            i = j
        else:
            i += 1
    if boxed_matches:
        return boxed_matches[-1].strip()
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else ""


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    return re.sub(r"\s+", "", text.replace("$", "").strip())


def score_response(example: dict, response: str, get_gold_text) -> bool:
    """Score a model response against the gold answer.

    Uses math_verify if available, falls back to string matching.
    `get_gold_text` is a callable that extracts the gold answer from an example dict.
    """
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        gold_text = get_gold_text(example)
        gold_parsed = parse(f"\\boxed{{{gold_text}}}", extraction_config=(LatexExtractionConfig(),))
        pred_parsed = parse(response, extraction_config=(ExprExtractionConfig(), LatexExtractionConfig()))
        if gold_parsed and pred_parsed:
            return any(verify(g, p) for g in gold_parsed for p in pred_parsed)
        return False
    except Exception:
        pred = extract_number(extract_final_answer(response))
        gold_text = get_gold_text(example)
        gold = extract_number(gold_text or "")
        return pred is not None and gold is not None and normalize_answer(pred) == normalize_answer(gold)


def compute_accuracy_with_rollout_client(
    rollout_client,
    examples,
    *,
    build_messages,
    get_prompt_text,
    get_gold_text,
    chat_template_kwargs_for_model,
    max_new_tokens: int = 32,
    top_p: float = 1.0,
    temperature: float = 0.0,
    top_k: int | None = None,
    max_concurrency: int | None = None,
    debug_event=None,
    debug_label: str | None = None,
) -> float:
    """Evaluate accuracy using the rollout client for generation."""
    return compute_accuracy_with_rollout_client_batched(
        rollout_client, examples,
        build_messages=build_messages,
        get_prompt_text=get_prompt_text,
        get_gold_text=get_gold_text,
        chat_template_kwargs_for_model=chat_template_kwargs_for_model,
        max_new_tokens=max_new_tokens, top_p=top_p,
        temperature=temperature, top_k=top_k,
        max_concurrency=max_concurrency,
        debug_event=debug_event, debug_label=debug_label,
    )


def compute_accuracy_with_rollout_client_batched(
    rollout_client,
    examples,
    *,
    build_messages,
    get_prompt_text,
    get_gold_text,
    chat_template_kwargs_for_model,
    max_new_tokens: int = 32,
    top_p: float = 1.0,
    temperature: float = 0.0,
    top_k: int | None = None,
    max_concurrency: int | None = None,
    debug_event=None,
    debug_label: str | None = None,
) -> float:
    """Batch-evaluate accuracy using the rollout client."""
    eval_max_new_tokens_env = os.environ.get("OPD_EVAL_MAX_NEW_TOKENS")
    if eval_max_new_tokens_env:
        max_new_tokens = int(eval_max_new_tokens_env)
        if max_new_tokens <= 0:
            raise ValueError("OPD_EVAL_MAX_NEW_TOKENS must be positive when set")
    eval_request_batch_size_env = os.environ.get("OPD_EVAL_REQUEST_BATCH_SIZE")
    eval_request_batch_size = None if not eval_request_batch_size_env else int(eval_request_batch_size_env)
    if eval_request_batch_size is not None and eval_request_batch_size <= 0:
        raise ValueError("OPD_EVAL_REQUEST_BATCH_SIZE must be positive when set")
    correct = 0
    total_response_chars = 0
    max_response_chars = 0
    min_response_chars = None
    batch_size = eval_request_batch_size or len(examples)
    for batch_start in range(0, len(examples), batch_size):
        batch_examples = examples[batch_start : batch_start + batch_size]
        messages_batch = [build_messages(get_prompt_text(ex)) for ex in batch_examples]
        default_concurrency = len(messages_batch) if messages_batch else 1
        requested_concurrency = max_concurrency if max_concurrency is not None else default_concurrency
        effective_concurrency = min(requested_concurrency, default_concurrency)
        if debug_event is not None:
            debug_event(
                f"{debug_label or 'rollout_eval'}_generate_start",
                example_count=len(examples),
                batch_start=batch_start,
                batch_size=len(batch_examples),
                max_concurrency=effective_concurrency,
                max_new_tokens=max_new_tokens,
            )
        generate_start = time.perf_counter()
        responses = rollout_client.generate_messages(
            messages_batch,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            chat_template_kwargs=chat_template_kwargs_for_model(rollout_client.model),
            max_concurrency=effective_concurrency,
        )
        generate_seconds = time.perf_counter() - generate_start
        response_lengths = [len(response) for response in responses]
        total_response_chars += sum(response_lengths)
        if response_lengths:
            max_response_chars = max(max_response_chars, max(response_lengths))
            local_min = min(response_lengths)
            min_response_chars = local_min if min_response_chars is None else min(min_response_chars, local_min)
        if debug_event is not None:
            debug_event(
                f"{debug_label or 'rollout_eval'}_generate_end",
                example_count=len(examples),
                batch_start=batch_start,
                batch_size=len(batch_examples),
                response_count=len(responses),
                generate_seconds=generate_seconds,
                response_chars_total=sum(response_lengths),
                response_chars_max=max(response_lengths) if response_lengths else 0,
                response_chars_min=min(response_lengths) if response_lengths else 0,
            )
        for ex, response in zip(batch_examples, responses):
            if score_response(ex, response, get_gold_text):
                correct += 1
    if debug_event is not None:
        debug_event(
            f"{debug_label or 'rollout_eval'}_score_end",
            example_count=len(examples),
            correct=correct,
            accuracy=correct / max(len(examples), 1),
            response_chars_total=total_response_chars,
            response_chars_max=max_response_chars,
            response_chars_min=min_response_chars or 0,
            eval_request_batch_size=eval_request_batch_size,
        )
    return correct / max(len(examples), 1)


def compute_named_accuracies_with_rollout_client_batched(
    rollout_client,
    named_examples: list[tuple[str, list[dict]]],
    *,
    build_messages,
    get_prompt_text,
    get_gold_text,
    chat_template_kwargs_for_model,
    max_new_tokens: int = 32,
    top_p: float = 1.0,
    temperature: float = 0.0,
    top_k: int | None = None,
    max_concurrency: int | None = None,
    debug_event=None,
    debug_label: str | None = None,
) -> dict[str, dict[str, float | int]]:
    """Evaluate accuracy on multiple named splits using the rollout client."""
    if not named_examples:
        raise ValueError("named_examples must not be empty")
    flat_examples: list[tuple[str, dict]] = []
    metrics: dict[str, dict[str, float | int]] = {}
    for name, examples in named_examples:
        if not examples:
            continue
        if name in metrics:
            raise ValueError(f"duplicate eval split name: {name}")
        metrics[name] = {"correct": 0, "total": len(examples)}
        flat_examples.extend((name, example) for example in examples)
    if not flat_examples:
        raise ValueError("named_examples must contain at least one non-empty example list")

    eval_max_new_tokens_env = os.environ.get("OPD_EVAL_MAX_NEW_TOKENS")
    if eval_max_new_tokens_env:
        max_new_tokens = int(eval_max_new_tokens_env)
        if max_new_tokens <= 0:
            raise ValueError("OPD_EVAL_MAX_NEW_TOKENS must be positive when set")
    eval_request_batch_size_env = os.environ.get("OPD_EVAL_REQUEST_BATCH_SIZE")
    eval_request_batch_size = None if not eval_request_batch_size_env else int(eval_request_batch_size_env)
    if eval_request_batch_size is not None and eval_request_batch_size <= 0:
        raise ValueError("OPD_EVAL_REQUEST_BATCH_SIZE must be positive when set")

    total_response_chars = 0
    max_response_chars = 0
    min_response_chars = None
    total_response_tokens = 0
    batch_size = eval_request_batch_size or len(flat_examples)
    split_sizes = {name: data["total"] for name, data in metrics.items()}
    for batch_start in range(0, len(flat_examples), batch_size):
        batch_items = flat_examples[batch_start : batch_start + batch_size]
        messages_batch = [build_messages(get_prompt_text(example)) for _, example in batch_items]
        default_concurrency = len(messages_batch) if messages_batch else 1
        requested_concurrency = max_concurrency if max_concurrency is not None else default_concurrency
        effective_concurrency = min(requested_concurrency, default_concurrency)
        if debug_event is not None:
            debug_event(
                f"{debug_label or 'rollout_eval'}_generate_start",
                example_count=len(flat_examples),
                split_sizes=split_sizes,
                batch_start=batch_start,
                batch_size=len(batch_items),
                max_concurrency=effective_concurrency,
                max_new_tokens=max_new_tokens,
            )
        generate_start = time.perf_counter()
        responses = rollout_client.generate_messages(
            messages_batch,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            chat_template_kwargs=chat_template_kwargs_for_model(rollout_client.model),
            max_concurrency=effective_concurrency,
        )
        generate_seconds = time.perf_counter() - generate_start
        response_lengths = [len(response) for response in responses]
        total_response_chars += sum(response_lengths)
        if response_lengths:
            max_response_chars = max(max_response_chars, max(response_lengths))
            local_min = min(response_lengths)
            min_response_chars = local_min if min_response_chars is None else min(min_response_chars, local_min)
        token_counts = getattr(rollout_client, 'last_output_token_counts', None) or [len(rollout_client._tokenizer.encode(r, add_special_tokens=False)) for r in responses]
        total_response_tokens += sum(token_counts)
        if debug_event is not None:
            debug_event(
                f"{debug_label or 'rollout_eval'}_generate_end",
                example_count=len(flat_examples),
                split_sizes=split_sizes,
                batch_start=batch_start,
                batch_size=len(batch_items),
                response_count=len(responses),
                generate_seconds=generate_seconds,
                response_chars_total=sum(response_lengths),
                response_chars_max=max(response_lengths) if response_lengths else 0,
                response_chars_min=min(response_lengths) if response_lengths else 0,
            )
        for (split_name, example), response in zip(batch_items, responses):
            if score_response(example, response, get_gold_text):
                metrics[split_name]["correct"] = int(metrics[split_name]["correct"]) + 1

    for split_name, data in metrics.items():
        correct_count = int(data["correct"])
        total_count = int(data["total"])
        data["accuracy"] = correct_count / max(total_count, 1)
    total_examples = sum(d["total"] for d in metrics.values())
    for data in metrics.values():
        data["avg_gen_tokens"] = total_response_tokens / max(total_examples, 1)
    if debug_event is not None:
        score_summary = {name: {"correct": int(d["correct"]), "total": int(d["total"]), "accuracy": d["accuracy"]} for name, d in metrics.items()}
        debug_event(
            f"{debug_label or 'rollout_eval'}_score_end",
            example_count=len(flat_examples),
            split_scores=score_summary,
            response_chars_total=total_response_chars,
            response_chars_max=max_response_chars,
            response_chars_min=min_response_chars or 0,
            eval_request_batch_size=eval_request_batch_size,
        )
    return metrics


def distributed_eval(
    rollout_client,
    named_examples: list[tuple[str, list[dict]]],
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    *,
    build_messages,
    get_prompt_text,
    get_gold_text,
    chat_template_kwargs_for_model,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    top_k: int | None,
    max_concurrency: int | None,
    debug_event,
    debug_label: str,
) -> dict[str, dict[str, float | int]]:
    """Evaluate across all ranks, each rank handles its shard. All-reduces correct/total."""
    sharded = [(name, examples[rank::world_size] if distributed else examples) for name, examples in named_examples]
    sharded = [(name, exs) for name, exs in sharded if exs]
    if not sharded:
        local_metrics = {}
        for name, examples in named_examples:
            local_metrics[name] = {"correct": 0, "total": 0, "accuracy": 0.0}
    else:
        local_metrics = compute_named_accuracies_with_rollout_client_batched(
            rollout_client,
            sharded,
            build_messages=build_messages,
            get_prompt_text=get_prompt_text,
            get_gold_text=get_gold_text,
            chat_template_kwargs_for_model=chat_template_kwargs_for_model,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
            max_concurrency=max_concurrency,
            debug_event=debug_event,
            debug_label=debug_label,
        )
    if distributed and dist.is_initialized():
        all_names = [name for name, _ in named_examples]
        counts = torch.zeros(len(all_names) * 3, device=device, dtype=torch.int64)
        for i, name in enumerate(all_names):
            m = local_metrics.get(name, {"correct": 0, "total": 0, "avg_gen_tokens": 0.0})
            counts[i * 3] = int(m.get("correct", 0))
            counts[i * 3 + 1] = int(m.get("total", 0))
            counts[i * 3 + 2] = int(round(m.get("avg_gen_tokens", 0.0) * int(m.get("total", 0))))
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        metrics = {}
        for i, name in enumerate(all_names):
            correct_val = int(counts[i * 3].item())
            total_val = int(counts[i * 3 + 1].item())
            total_tokens = int(counts[i * 3 + 2].item())
            metrics[name] = {"correct": correct_val, "total": total_val, "accuracy": correct_val / max(total_val, 1), "avg_gen_tokens": total_tokens / max(total_val, 1)}
        return metrics
    return local_metrics
