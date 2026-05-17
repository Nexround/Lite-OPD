from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from liteopd.losses import chunk_loss_from_hidden_chunk, chunked_kl_from_hidden


def make_tensors(seq_len: int, hidden_size: int, vocab_size: int, device: torch.device):
    torch.manual_seed(seq_len)
    student_input = torch.randn(1, seq_len, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    teacher_hidden = torch.randn(1, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    trunk = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=torch.bfloat16)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=torch.bfloat16)
    return student_input, teacher_hidden, trunk, lm_head


def get_param_grad_norms(trunk: nn.Linear, lm_head: nn.Linear, student_input: torch.Tensor) -> dict[str, float]:
    result = {"student_input": float(student_input.grad.norm().cpu()) if student_input.grad is not None else 0.0}
    if trunk.weight.grad is not None:
        result["trunk.weight"] = float(trunk.weight.grad.norm().cpu())
    else:
        result["trunk.weight"] = 0.0
    if lm_head.weight.grad is not None:
        result["lm_head.weight"] = float(lm_head.weight.grad.norm().cpu())
    else:
        result["lm_head.weight"] = 0.0
    return result


def run_sample_mode(seq_len: int, hidden_size: int, vocab_size: int, device: torch.device):
    student_input, teacher_hidden, trunk, lm_head = make_tensors(seq_len, hidden_size, vocab_size, device)
    student_hidden = trunk(student_input)
    chunk_tokens = max(1, (seq_len + 7) // 8)
    loss = chunked_kl_from_hidden(student_hidden, teacher_hidden, lm_head, lm_head, loss_name="forward_kl", chunk_tokens=chunk_tokens)
    loss.backward()
    grads = get_param_grad_norms(trunk, lm_head, student_input)
    return float(loss.detach().cpu()), grads


def run_two_stage_mode(seq_len: int, hidden_size: int, vocab_size: int, device: torch.device):
    student_input, teacher_hidden, trunk, lm_head = make_tensors(seq_len, hidden_size, vocab_size, device)
    student_hidden = trunk(student_input)
    student_hidden_leaf = student_hidden.detach().requires_grad_(True)
    chunk_tokens = max(1, (seq_len + 7) // 8)
    chunk_ranges = [(start, min(seq_len, start + chunk_tokens)) for start in range(0, seq_len, chunk_tokens)]
    total_tokens = sum((end - start) for start, end in chunk_ranges)
    weighted_loss_value = 0.0
    hidden_grad_accum = torch.zeros_like(student_hidden_leaf)
    for start, end in chunk_ranges:
        token_count = end - start
        token_loss = chunk_loss_from_hidden_chunk(
            student_hidden_leaf[:, start:end, :],
            teacher_hidden[:, start:end, :],
            lm_head,
            lm_head,
            loss_name="forward_kl",
        )
        weighted_chunk_loss = token_loss * (token_count / total_tokens)
        weighted_loss_value += float(weighted_chunk_loss.detach().cpu())
        grads = torch.autograd.grad(weighted_chunk_loss, [student_hidden_leaf, lm_head.weight], retain_graph=False, allow_unused=True)
        if grads[0] is not None:
            hidden_grad_accum.add_(grads[0])
        if grads[1] is not None:
            if lm_head.weight.grad is None:
                lm_head.weight.grad = grads[1].detach().clone()
            else:
                lm_head.weight.grad.add_(grads[1].detach())
    torch.autograd.backward(student_hidden, grad_tensors=hidden_grad_accum)
    grads = get_param_grad_norms(trunk, lm_head, student_input)
    return weighted_loss_value, grads


def compare_grad_norms(sample_grads, two_stage_grads):
    keys = sorted(set(sample_grads) | set(two_stage_grads))
    results = []
    for key in keys:
        a = sample_grads.get(key, 0.0)
        b = two_stage_grads.get(key, 0.0)
        denom = max(abs(a), 1e-12)
        rel_diff = abs(a - b) / denom
        results.append({"name": key, "sample": a, "two_stage": b, "rel_diff": rel_diff})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for total_tokens in [5000, 10000, 15000]:
        sample_loss, sample_grads = run_sample_mode(total_tokens, args.hidden_size, args.vocab_size, device)
        two_stage_loss, two_stage_grads = run_two_stage_mode(total_tokens, args.hidden_size, args.vocab_size, device)
        report.append({
            "total_tokens": total_tokens,
            "sample_loss": sample_loss,
            "two_stage_loss": two_stage_loss,
            "loss_abs_diff": abs(sample_loss - two_stage_loss),
            "grads": compare_grad_norms(sample_grads, two_stage_grads),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "cases": len(report), "hidden_size": args.hidden_size, "vocab_size": args.vocab_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
