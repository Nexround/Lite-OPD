"""
End-to-end gradient parity test: two_stage vs sample (naive) backward.

Loads the real student model, runs a forward pass, then compares gradients
produced by the two backward modes on the same hidden states. This catches
bugs in the two_stage detach/accumulate/backward logic that synthetic tests miss.

Usage:
    python tests/check_twostage_grad_parity.py \
        --student-model /path/to/model \
        --output /tmp/twostage_grad_parity.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.losses import chunk_loss_from_hidden_chunk, chunked_kl_from_hidden
from liteopd.train.packing import pack_sequences


PROMPTS = [
    "What is 2+2?",
    "Explain the Pythagorean theorem in one sentence.",
    "Write a Python function to compute factorial.",
    "What is the capital of France?",
    "Describe how a neural network learns in two sentences.",
    "What are the first 10 prime numbers?",
]
RESPONSES = [
    "2+2 equals 4.",
    "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "The capital of France is Paris.",
    "A neural network learns by adjusting its weights through backpropagation, minimizing the difference between predicted and actual outputs. Each training iteration updates parameters in the direction that reduces the loss function.",
    "The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.",
]


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = {}
    if "Qwen3" in tokenizer.name_or_path:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def run_backward_sample(
    student_hidden_packed: torch.Tensor,
    teacher_hidden_packed: torch.Tensor,
    lm_head,
    packed_batch,
    loss_name: str,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Naive sample-mode backward: compute loss per sample, backward immediately."""
    total_response_tokens = sum(packed_batch.response_token_counts)
    num_samples = len(packed_batch.seq_lengths)
    offset = 0
    loss_values = []

    for sample_idx, (seq_len, prompt_len, response_tokens) in enumerate(zip(
        packed_batch.seq_lengths, packed_batch.prompt_lengths, packed_batch.response_token_counts
    )):
        response_start = offset + prompt_len - 1
        response_end = offset + seq_len
        student_hidden_slice = student_hidden_packed[0, response_start:response_end, :].unsqueeze(0)
        teacher_hidden_slice = teacher_hidden_packed[0, response_start:response_end, :].unsqueeze(0)

        seq_tokens = student_hidden_slice.shape[1]
        chunk_tokens = max(1, math.ceil(seq_tokens / 8))

        sample_loss = chunked_kl_from_hidden(
            student_hidden_slice, teacher_hidden_slice,
            lm_head, lm_head,
            loss_name=loss_name, chunk_tokens=chunk_tokens,
        )
        weighted_loss_value = float(sample_loss.detach().cpu())
        is_last = sample_idx == num_samples - 1
        (sample_loss * response_tokens / total_response_tokens).backward(retain_graph=not is_last)
        loss_values.append(weighted_loss_value * response_tokens / total_response_tokens)
        offset += seq_len

    total_loss = sum(loss_values) / len(loss_values)
    grads = {}
    for name, param in lm_head.named_parameters():
        if param.grad is not None:
            grads[f"lm_head.{name}"] = param.grad.clone()
    return total_loss, grads


def run_backward_two_stage(
    student_hidden_packed: torch.Tensor,
    teacher_hidden_packed: torch.Tensor,
    lm_head,
    packed_batch,
    loss_name: str,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Two-stage backward: detach hidden, accumulate grads, then backward through body."""
    total_response_tokens = sum(packed_batch.response_token_counts)
    lm_head_params = [p for p in lm_head.parameters() if p.requires_grad]
    packed_grad_accum = torch.zeros_like(student_hidden_packed)
    offset = 0
    loss_values = []

    for seq_len, prompt_len, response_tokens in zip(
        packed_batch.seq_lengths, packed_batch.prompt_lengths, packed_batch.response_token_counts
    ):
        response_start = offset + prompt_len - 1
        response_end = offset + seq_len
        student_hidden_slice = student_hidden_packed[0, response_start:response_end, :].unsqueeze(0)
        teacher_hidden_slice = teacher_hidden_packed[0, response_start:response_end, :].unsqueeze(0)

        seq_tokens = student_hidden_slice.shape[1]
        chunk_tokens = max(1, math.ceil(seq_tokens / 8))
        chunk_ranges = [(start, min(seq_tokens, start + chunk_tokens)) for start in range(0, seq_tokens, chunk_tokens)]
        total_tokens = sum((end - start) for start, end in chunk_ranges)

        hidden_leaf = student_hidden_slice.detach().requires_grad_(True)
        hidden_grad_accum = torch.zeros_like(hidden_leaf)
        weighted_loss_value = 0.0

        for start, end in chunk_ranges:
            token_count = end - start
            token_loss = chunk_loss_from_hidden_chunk(
                hidden_leaf[:, start:end, :],
                teacher_hidden_slice[:, start:end, :],
                lm_head, lm_head,
                loss_name=loss_name,
            )
            scale = token_count / total_tokens
            weighted_loss_value += float((token_loss * scale).detach().cpu())
            grad_scale = scale * response_tokens / total_response_tokens
            grad_outputs = torch.autograd.grad(
                token_loss * grad_scale, [hidden_leaf, *lm_head_params],
                retain_graph=False, allow_unused=True,
            )
            if grad_outputs[0] is not None:
                hidden_grad_accum.add_(grad_outputs[0])
            for param, grad in zip(lm_head_params, grad_outputs[1:]):
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad.detach().clone()
                    else:
                        param.grad.add_(grad.detach())

        packed_grad_accum[0, response_start:response_end, :] = hidden_grad_accum.squeeze(0)
        loss_values.append(weighted_loss_value * response_tokens / total_response_tokens)
        offset += seq_len

    torch.autograd.backward(student_hidden_packed, grad_tensors=packed_grad_accum)

    total_loss = sum(loss_values) / len(loss_values)
    grads = {}
    for name, param in lm_head.named_parameters():
        if param.grad is not None:
            grads[f"lm_head.{name}"] = param.grad.clone()
    return total_loss, grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--loss-name", default="reverse_kl", choices=["forward_kl", "reverse_kl", "jsd"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pack-tokens", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Loading model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    student.eval()

    prompts = [build_prompt(tokenizer, q) for q in PROMPTS]
    packed_batches = pack_sequences(tokenizer, prompts, RESPONSES, device, args.max_pack_tokens)
    print(f"Packed into {len(packed_batches)} batch(es), "
          f"total tokens: {sum(sum(pb.seq_lengths) for pb in packed_batches)}")

    report = []
    for batch_idx, packed in enumerate(packed_batches):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  Sequences: {len(packed.seq_lengths)}, total tokens: {sum(packed.seq_lengths)}")

        # Forward pass for teacher (use student weights but add noise to create divergence)
        with torch.no_grad():
            teacher_hidden = student.model(
                packed.input_ids, position_ids=packed.position_ids, use_cache=False
            ).last_hidden_state
            # Add noise to teacher hidden to create non-zero KL divergence
            torch.manual_seed(42 + batch_idx)
            teacher_hidden = teacher_hidden + torch.randn_like(teacher_hidden) * 0.1

        # --- Run sample (naive) mode ---
        student.zero_grad()
        student_hidden_sample = student.model(
            packed.input_ids, position_ids=packed.position_ids, use_cache=False
        ).last_hidden_state
        loss_sample, grads_sample = run_backward_sample(
            student_hidden_sample, teacher_hidden, student.lm_head, packed, args.loss_name
        )
        # Collect body grads
        body_grads_sample = {}
        for name, param in student.model.named_parameters():
            if param.grad is not None:
                body_grads_sample[name] = param.grad.clone()
        grads_sample.update({f"body.{k}": v for k, v in body_grads_sample.items()})

        # --- Run two_stage mode ---
        student.zero_grad()
        student_hidden_twostage = student.model(
            packed.input_ids, position_ids=packed.position_ids, use_cache=False
        ).last_hidden_state
        loss_twostage, grads_twostage = run_backward_two_stage(
            student_hidden_twostage, teacher_hidden, student.lm_head, packed, args.loss_name
        )
        body_grads_twostage = {}
        for name, param in student.model.named_parameters():
            if param.grad is not None:
                body_grads_twostage[name] = param.grad.clone()
        grads_twostage.update({f"body.{k}": v for k, v in body_grads_twostage.items()})

        # --- Compare ---
        # Use cosine similarity of the full gradient vector as primary metric.
        # Per-param relative diff can be large for bf16 due to accumulation order,
        # but cosine similarity captures whether the gradient *direction* is correct.
        dot_product = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0

        batch_report = {
            "batch_idx": batch_idx,
            "num_sequences": len(packed.seq_lengths),
            "total_tokens": sum(packed.seq_lengths),
            "loss_sample": loss_sample,
            "loss_twostage": loss_twostage,
            "loss_abs_diff": abs(loss_sample - loss_twostage),
            "grad_comparisons": [],
        }

        all_keys = sorted(set(grads_sample.keys()) | set(grads_twostage.keys()))
        max_rel_diff = 0.0
        for key in all_keys:
            g_sample = grads_sample.get(key)
            g_twostage = grads_twostage.get(key)
            if g_sample is None or g_twostage is None:
                batch_report["grad_comparisons"].append({
                    "name": key,
                    "status": "missing",
                    "in_sample": g_sample is not None,
                    "in_twostage": g_twostage is not None,
                })
                continue

            gs_f = g_sample.float()
            gt_f = g_twostage.float()
            dot_product += (gs_f * gt_f).sum().item()
            norm_a_sq += (gs_f * gs_f).sum().item()
            norm_b_sq += (gt_f * gt_f).sum().item()

            abs_diff = (g_sample - g_twostage).abs()
            norm_sample = g_sample.norm().item()
            norm_twostage = g_twostage.norm().item()
            max_abs = abs_diff.max().item()
            mean_abs = abs_diff.mean().item()
            denom = max(norm_sample, 1e-12)
            rel_diff = (g_sample - g_twostage).norm().item() / denom
            max_rel_diff = max(max_rel_diff, rel_diff)

            batch_report["grad_comparisons"].append({
                "name": key,
                "norm_sample": norm_sample,
                "norm_twostage": norm_twostage,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "rel_diff": rel_diff,
            })

        cosine_sim = dot_product / (norm_a_sq**0.5 * norm_b_sq**0.5 + 1e-12)
        l2_rel = ((norm_a_sq + norm_b_sq - 2 * dot_product) ** 0.5) / (norm_a_sq**0.5 + 1e-12)
        batch_report["cosine_similarity"] = cosine_sim
        batch_report["l2_relative_diff"] = l2_rel
        batch_report["max_per_param_rel_diff"] = max_rel_diff
        report.append(batch_report)

        status = "PASS" if cosine_sim > 0.9999 else "FAIL"
        print(f"  Loss sample={loss_sample:.6f}, two_stage={loss_twostage:.6f}, diff={abs(loss_sample - loss_twostage):.2e}")
        print(f"  Cosine similarity: {cosine_sim:.8f} [{status}]")
        print(f"  L2 relative diff: {l2_rel:.6e}")
        print(f"  Max per-param rel diff: {max_rel_diff:.6e}")
        print(f"  Compared {len(all_keys)} parameter groups")

    # Summary
    min_cosine = min(b["cosine_similarity"] for b in report)
    passed = min_cosine > 0.9999
    print(f"\n{'='*60}")
    print(f"Min cosine similarity: {min_cosine:.8f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"(threshold: cosine > 0.9999 for bf16)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "passed": passed,
        "min_cosine_similarity": min_cosine,
        "loss_name": args.loss_name,
        "model": args.student_model,
        "batches": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to: {output_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
