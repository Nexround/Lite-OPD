"""
Multi-GPU gradient parity test: two_stage vs sample backward with ZeRO-2.

Verifies that the two_stage backward produces identical gradients to the naive
sample-mode backward in a distributed ZeRO-2 setting. Each rank processes
different data (simulating real training), and we compare gradients after
reduce_scatter_grads().

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/check_twostage_grad_parity_distributed.py \
        --student-model /path/to/model \
        --output /tmp/twostage_grad_parity_dist.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.losses import chunk_loss_from_hidden_chunk, chunked_kl_from_hidden
from liteopd.train.packing import pack_sequences
from liteopd.train.zero2 import ZeRO2Optimizer


PROMPTS_PER_RANK = [
    [
        "What is 2+2?",
        "Explain the Pythagorean theorem in one sentence.",
        "Write a Python function to compute factorial.",
    ],
    [
        "What is the capital of France?",
        "Describe how a neural network learns in two sentences.",
        "What are the first 10 prime numbers?",
    ],
]
RESPONSES_PER_RANK = [
    [
        "2+2 equals 4.",
        "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    ],
    [
        "The capital of France is Paris.",
        "A neural network learns by adjusting its weights through backpropagation, minimizing the difference between predicted and actual outputs. Each training iteration updates parameters in the direction that reduces the loss function.",
        "The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.",
    ],
]


def run_backward_two_stage(student_hidden, teacher_hidden, lm_head, packed, loss_name, total_response_tokens):
    """Two-stage backward: detach hidden, accumulate grads, backward through body."""
    lm_params = [p for p in lm_head.parameters() if p.requires_grad]
    packed_grad_accum = torch.zeros_like(student_hidden)
    offset = 0
    for sl, pl, rt in zip(packed.seq_lengths, packed.prompt_lengths, packed.response_token_counts):
        rs = offset + pl - 1
        re = offset + sl
        s_slice = student_hidden[0, rs:re, :].unsqueeze(0)
        t_slice = teacher_hidden[0, rs:re, :].unsqueeze(0)
        seq_tokens = s_slice.shape[1]
        ct = max(1, math.ceil(seq_tokens / 8))
        chunk_ranges = [(start, min(seq_tokens, start + ct)) for start in range(0, seq_tokens, ct)]
        total_tokens = sum((end - start) for start, end in chunk_ranges)

        hidden_leaf = s_slice.detach().requires_grad_(True)
        hga = torch.zeros_like(hidden_leaf)
        for start, end in chunk_ranges:
            tc = end - start
            tl = chunk_loss_from_hidden_chunk(
                hidden_leaf[:, start:end, :], t_slice[:, start:end, :],
                lm_head, lm_head, loss_name=loss_name,
            )
            scale = tc / total_tokens
            grad_scale = scale * rt / total_response_tokens
            grads = torch.autograd.grad(tl * grad_scale, [hidden_leaf, *lm_params], retain_graph=False, allow_unused=True)
            if grads[0] is not None:
                hga.add_(grads[0])
            for param, g in zip(lm_params, grads[1:]):
                if g is not None:
                    if param.grad is None:
                        param.grad = g.detach().clone()
                    else:
                        param.grad.add_(g.detach())
        packed_grad_accum[0, rs:re, :] = hga.squeeze(0)
        offset += sl

    torch.autograd.backward(student_hidden, grad_tensors=packed_grad_accum)


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = {}
    if "Qwen3" in tokenizer.name_or_path:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def run_backward_sample(student_hidden, teacher_hidden, lm_head, packed, loss_name, total_response_tokens):
    """Naive sample-mode backward."""
    num_samples = len(packed.seq_lengths)
    offset = 0
    for i, (sl, pl, rt) in enumerate(zip(packed.seq_lengths, packed.prompt_lengths, packed.response_token_counts)):
        rs = offset + pl - 1
        re = offset + sl
        s_slice = student_hidden[0, rs:re, :].unsqueeze(0)
        t_slice = teacher_hidden[0, rs:re, :].unsqueeze(0)
        seq_tokens = s_slice.shape[1]
        ct = max(1, math.ceil(seq_tokens / 8))
        loss = chunked_kl_from_hidden(s_slice, t_slice, lm_head, lm_head, loss_name=loss_name, chunk_tokens=ct)
        is_last = i == num_samples - 1
        (loss * rt / total_response_tokens).backward(retain_graph=not is_last)
        offset += sl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--loss-name", default="reverse_kl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pack-tokens", type=int, default=4096)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Loading model: {args.student_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    student.eval()

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = ZeRO2Optimizer(
        params=trainable_params, lr=1e-5, rank=rank, world_size=world_size,
    )
    optimizer.init_optimizer_state()

    prompts_raw = PROMPTS_PER_RANK[rank % len(PROMPTS_PER_RANK)]
    responses = RESPONSES_PER_RANK[rank % len(RESPONSES_PER_RANK)]
    prompts = [build_prompt(tokenizer, q) for q in prompts_raw]
    packed_batches = pack_sequences(tokenizer, prompts, responses, device, args.max_pack_tokens)

    if rank == 0:
        print(f"Rank 0: {len(packed_batches)} batch(es), "
              f"tokens: {sum(sum(pb.seq_lengths) for pb in packed_batches)}")

    # --- Run sample mode ---
    for p in student.parameters():
        p.grad = None

    for packed in packed_batches:
        with torch.no_grad():
            teacher_hidden = student.model(
                packed.input_ids, position_ids=packed.position_ids, use_cache=False
            ).last_hidden_state
            torch.manual_seed(42 + rank)
            teacher_hidden = teacher_hidden + torch.randn_like(teacher_hidden) * 0.1

        student_hidden = student.model(
            packed.input_ids, position_ids=packed.position_ids, use_cache=False
        ).last_hidden_state
        total_resp = sum(packed.response_token_counts)
        run_backward_sample(student_hidden, teacher_hidden, student.lm_head, packed, args.loss_name, total_resp)

    optimizer.prepare_grad_buffer()
    optimizer.reduce_scatter_grads()
    sample_grad_shard = optimizer._grad_shard_out.clone()

    # Restore params after reduce_scatter overwrote _flat_params_padded
    dist.all_gather_into_tensor(
        optimizer._flat_params_padded, optimizer._param_shard_snapshot, group=optimizer._pg
    )

    # --- Run two_stage mode ---
    for p in student.parameters():
        p.grad = None

    for packed in packed_batches:
        with torch.no_grad():
            teacher_hidden = student.model(
                packed.input_ids, position_ids=packed.position_ids, use_cache=False
            ).last_hidden_state
            torch.manual_seed(42 + rank)
            teacher_hidden = teacher_hidden + torch.randn_like(teacher_hidden) * 0.1

        student_hidden = student.model(
            packed.input_ids, position_ids=packed.position_ids, use_cache=False
        ).last_hidden_state
        total_resp = sum(packed.response_token_counts)
        run_backward_two_stage(student_hidden, teacher_hidden, student.lm_head, packed, args.loss_name, total_resp)

    optimizer.prepare_grad_buffer()
    optimizer.reduce_scatter_grads()
    twostage_grad_shard = optimizer._grad_shard_out.clone()

    # --- Compare gradient shards ---
    dot = (sample_grad_shard.float() * twostage_grad_shard.float()).sum().item()
    norm_a = sample_grad_shard.float().norm().item()
    norm_b = twostage_grad_shard.float().norm().item()
    cos_sim = dot / (norm_a * norm_b + 1e-12)
    l2_rel = (sample_grad_shard.float() - twostage_grad_shard.float()).norm().item() / (norm_a + 1e-12)
    max_abs = (sample_grad_shard - twostage_grad_shard).abs().max().item()

    passed = cos_sim > 0.9999

    print(f"\nRank {rank} gradient shard comparison:")
    print(f"  Shard size: {sample_grad_shard.numel():,}")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  L2 relative diff:  {l2_rel:.6e}")
    print(f"  Max abs diff:      {max_abs:.6e}")
    print(f"  Norm sample:       {norm_a:.6e}")
    print(f"  Norm two_stage:    {norm_b:.6e}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "passed": passed,
            "world_size": world_size,
            "cosine_similarity": cos_sim,
            "l2_relative_diff": l2_rel,
            "max_abs_diff": max_abs,
            "norm_sample": norm_a,
            "norm_twostage": norm_b,
            "shard_size": sample_grad_shard.numel(),
            "loss_name": args.loss_name,
            "model": args.student_model,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to: {output_path}")

    dist.barrier()
    dist.destroy_process_group()

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()