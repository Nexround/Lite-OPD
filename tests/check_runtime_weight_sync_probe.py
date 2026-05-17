"""Probe weight sharing and first rollout using the real v1 training path.

Key detail:
- v1 training calls `fuse_model_projections(student)` before creating the
  in-process rollout runtime.
- `share_weights(...)` assumes the HF model is already fused.

This script reproduces that path and checks:
1. key shared tensors alias the same storage;
2. those tensors stay finite;
3. a short sampled rollout no longer collapses immediately.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.runtime.rollout import InProcessRolloutClient
from liteopd.runtime.weight_sync import _get_hf_param
from liteopd.train.fused_model import fuse_model_projections


@dataclass
class CompareRow:
    key: str
    same_shape: bool
    same_ptr: bool
    hf_finite: bool
    engine_finite: bool
    max_abs_diff: float


def qwen_chat_template_kwargs(model_name: str) -> dict | None:
    if "Qwen3" in model_name:
        return {"enable_thinking": True}
    return None


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def get_engine_tensor(model, key: str) -> torch.Tensor:
    parts = key.split(".")
    obj = model
    for part in parts:
        if part == "op_list":
            continue
        if part.isdigit():
            obj = obj.op_list[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def compare_shared_tensors(hf_model, engine_model) -> list[CompareRow]:
    rows: list[CompareRow] = []
    sample_keys = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.5.self_attn.qkv_proj.weight",
        "model.layers.5.mlp.gate_up_proj.weight",
    ]
    for key in sample_keys:
        hf_tensor = _get_hf_param(hf_model, key)
        engine_tensor = get_engine_tensor(engine_model, key)
        same_ptr = hf_tensor.untyped_storage().data_ptr() == engine_tensor.untyped_storage().data_ptr()
        rows.append(
            CompareRow(
                key=key,
                same_shape=tuple(hf_tensor.shape) == tuple(engine_tensor.shape),
                same_ptr=same_ptr,
                hf_finite=bool(torch.isfinite(hf_tensor).all().item()),
                engine_finite=bool(torch.isfinite(engine_tensor).all().item()),
                max_abs_diff=float((hf_tensor - engine_tensor).abs().max().item()),
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/model")
    parser.add_argument("--prompt", default="Compute 2+3. Answer with just the final number.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--memory-ratio", type=float, default=0.35)
    parser.add_argument("--attention-backend", default="fi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
    ).to(device)
    fuse_model_projections(hf_model)
    hf_model.eval()

    client = InProcessRolloutClient(
        hf_model=hf_model,
        model_path=args.model,
        device=device,
        tokenizer=tokenizer,
        memory_ratio=args.memory_ratio,
        max_running_req=8,
        cuda_graph_max_bs=0,
        page_size=1,
        attention_backend=args.attention_backend,
    )

    print("=== shared tensor checks ===")
    rows = compare_shared_tensors(hf_model, client._sglang_model)
    for row in rows:
        print(
            f"{row.key}: same_shape={row.same_shape} same_ptr={row.same_ptr} "
            f"hf_finite={row.hf_finite} engine_finite={row.engine_finite} "
            f"max_abs_diff={row.max_abs_diff:.6f}"
        )

    print("\n=== sampled rollout ===")
    print(f"attention_backend={args.attention_backend}")
    client.prepare()
    outputs = client.generate_messages(
        [build_messages(args.prompt)],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        chat_template_kwargs=qwen_chat_template_kwargs(args.model),
    )
    client.release()
    print(repr(outputs[0]))

    client.shutdown()


if __name__ == "__main__":
    main()
