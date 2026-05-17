"""Regression smoke test for fused Qwen runtime weight sharing.

This covers the exact class of bug where Qwen3 `q_norm/k_norm` weights were not
shared into the embedded runtime model, producing NaN logits and repeated
token_id=0 / "!" outputs.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.runtime.rollout import InProcessRolloutClient
from liteopd.runtime.weight_sync import _get_hf_param
from liteopd.train.fused_model import fuse_model_projections


def qwen_chat_template_kwargs(model_name: str) -> dict | None:
    if "Qwen3" in model_name:
        return {"enable_thinking": True}
    return None


def get_engine_tensor(model, key: str) -> torch.Tensor:
    obj = model
    for part in key.split("."):
        if part == "op_list":
            continue
        if part.isdigit():
            obj = obj.op_list[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/model")
    parser.add_argument("--prompt", default="Compute 2+3. Answer with just the final number.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--attention-backend", default="fi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
    ).to(device)
    fuse_model_projections(model)
    model.eval()

    client = InProcessRolloutClient(
        hf_model=model,
        model_path=args.model,
        device=device,
        tokenizer=tokenizer,
        memory_ratio=0.35,
        max_running_req=8,
        cuda_graph_max_bs=0,
        page_size=1,
        attention_backend=args.attention_backend,
    )

    critical_keys = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.5.self_attn.q_norm.weight",
        "model.layers.5.self_attn.k_norm.weight",
    ]
    for key in critical_keys:
        hf_tensor = _get_hf_param(model, key)
        engine_tensor = get_engine_tensor(client._sglang_model, key)
        assert tuple(hf_tensor.shape) == tuple(engine_tensor.shape), f"shape mismatch: {key}"
        assert hf_tensor.untyped_storage().data_ptr() == engine_tensor.untyped_storage().data_ptr(), f"not shared: {key}"
        assert torch.isfinite(engine_tensor).all().item(), f"non-finite engine tensor: {key}"

    client.prepare()
    outputs = client.generate_messages(
        [[{"role": "user", "content": args.prompt}]],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        chat_template_kwargs=qwen_chat_template_kwargs(args.model),
    )
    client.release()
    client.shutdown()

    text = outputs[0]
    assert text, "empty runtime output"
    assert text != "!" * len(text), f"runtime collapsed to repeated token 0: {text!r}"
    print(repr(text))


if __name__ == "__main__":
    main()
