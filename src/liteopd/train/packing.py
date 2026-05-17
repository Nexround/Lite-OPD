from __future__ import annotations

from dataclasses import dataclass, field

import torch

_PAD_BUCKET_SIZE = 512


def _round_up_to_bucket(n: int) -> int:
    return ((n + _PAD_BUCKET_SIZE - 1) // _PAD_BUCKET_SIZE) * _PAD_BUCKET_SIZE


@dataclass
class PackedBatch:
    input_ids: torch.Tensor    # (1, total_tokens)
    position_ids: torch.Tensor # (1, total_tokens), resets to 0 at each sequence boundary
    seq_lengths: list[int]     # token count per sequence (prompt + response)
    prompt_lengths: list[int]  # prompt token count per sequence
    response_token_counts: list[int]  # response token count per sequence (for loss weighting)


def pack_sequences(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    max_pack_tokens: int = 32768,
) -> list[PackedBatch]:
    """Greedily pack sequences into micro-batches up to max_pack_tokens each."""
    # Tokenize prompt and response separately, then concatenate IDs.
    # This avoids BPE merge artifacts at the prompt/response boundary that
    # would make prompt_len inaccurate if we tokenized the concatenation.
    seqs: list[tuple[torch.Tensor, int, int]] = []  # (ids, seq_len, prompt_len)
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]
        ids = torch.cat([prompt_ids, response_ids])
        seqs.append((ids, ids.shape[0], prompt_ids.shape[0]))

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Greedy bin-packing: fill each micro-batch until adding next seq would exceed limit
    batches: list[PackedBatch] = []
    current_ids: list[torch.Tensor] = []
    current_pos: list[torch.Tensor] = []
    current_seq_lengths: list[int] = []
    current_prompt_lengths: list[int] = []
    current_response_counts: list[int] = []
    current_total = 0

    def flush():
        if not current_ids:
            return
        cat_ids = torch.cat(current_ids)
        cat_pos = torch.cat(current_pos)
        # Pad to bucket boundary to limit flex_attention recompilation
        padded_len = _round_up_to_bucket(cat_ids.shape[0])
        if padded_len > cat_ids.shape[0]:
            pad_size = padded_len - cat_ids.shape[0]
            cat_ids = torch.cat([cat_ids, cat_ids.new_full((pad_size,), pad_token_id)])
            cat_pos = torch.cat([cat_pos, cat_pos.new_zeros(pad_size)])
        batches.append(PackedBatch(
            input_ids=cat_ids.unsqueeze(0).to(device),
            position_ids=cat_pos.unsqueeze(0).to(device),
            seq_lengths=list(current_seq_lengths),
            prompt_lengths=list(current_prompt_lengths),
            response_token_counts=list(current_response_counts),
        ))
        current_ids.clear()
        current_pos.clear()
        current_seq_lengths.clear()
        current_prompt_lengths.clear()
        current_response_counts.clear()

    for ids, seq_len, prompt_len in seqs:
        if current_total + seq_len > max_pack_tokens and current_ids:
            flush()
            current_total = 0
        current_ids.append(ids)
        current_pos.append(torch.arange(seq_len, dtype=torch.long))
        current_seq_lengths.append(seq_len)
        current_prompt_lengths.append(prompt_len)
        current_response_counts.append(seq_len - prompt_len)
        current_total += seq_len

    flush()
    return batches
