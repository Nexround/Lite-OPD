"""Probe whether repeated '!' comes from model behavior or runtime sampling corruption.

This script compares:
1. HF generate (greedy / sampled)
2. v1 in-process runtime (greedy / sampled, CUDA graph optional)

It prints raw token ids and simple stats so we can tell whether the runtime is
collapsing to one repeated token in a way HF does not.
"""
from __future__ import annotations

import argparse
import gc
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.runtime.rollout import InProcessRolloutClient


def qwen_chat_template_kwargs(model_name: str) -> dict | None:
    if "Qwen3" in model_name:
        return {"enable_thinking": True}
    return None


def build_prompt(tokenizer, model_name: str, user_text: str) -> str:
    kwargs = qwen_chat_template_kwargs(model_name) or {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def summarize_ids(tokenizer, token_ids: list[int], label: str) -> None:
    unique = len(set(token_ids))
    counts = Counter(token_ids).most_common(8)
    decoded_top = [(tid, tokenizer.decode([tid]).replace("\n", "\\n")) for tid, _ in counts]
    print(f"{label}:")
    print(f"  length={len(token_ids)} unique={unique}")
    print(f"  first_32_ids={token_ids[:32]}")
    print(f"  top_counts={counts}")
    print(f"  top_decoded={decoded_top}")
    print(f"  text_preview={tokenizer.decode(token_ids[:80])!r}")


def hf_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> list[int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0.0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
    return output[0, inputs["input_ids"].shape[1] :].tolist()


def hf_first_step_top(model, tokenizer, prompt: str, k: int = 10) -> None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.numel()))
    pairs = [
        (int(tid), tokenizer.decode([int(tid)]).replace("\n", "\\n"), float(p))
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]
    print("HF first-step top candidates:")
    for tid, text, p in pairs:
        print(f"  id={tid:<6} token={text!r:<12} prob={p:.6f}")


def runtime_generate(
    model,
    tokenizer,
    model_path: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    cuda_graph_max_bs: int,
) -> list[int]:
    client = None
    try:
        client = InProcessRolloutClient(
            hf_model=model,
            model_path=model_path,
            device=model.device,
            tokenizer=tokenizer,
            memory_ratio=0.35,
            max_running_req=8,
            cuda_graph_max_bs=cuda_graph_max_bs,
            page_size=1,
            attention_backend="fi",
        )
        client.prepare()
        sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k is not None else -1,
            "max_tokens": max_tokens,
        }
        from liteopd.inference.core import SamplingParams

        raw = client._scheduler.generate([prompt], SamplingParams(**sampling))
        client.release()
        return raw[0]["token_ids"]
    finally:
        if client is not None:
            client.shutdown()
            del client
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/model")
    parser.add_argument("--prompt", default="Compute 2+3. Answer with just the final number.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--graph-bs", type=int, default=8)
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
        attn_implementation="eager",
    ).to(device)
    model.eval()

    prompt = build_prompt(tokenizer, args.model, args.prompt)
    print(f"Prompt preview: {prompt[:200]!r}")
    hf_first_step_top(model, tokenizer, prompt)

    hf_greedy = hf_generate(
        model, tokenizer, prompt,
        max_tokens=args.max_tokens, temperature=0.0, top_p=args.top_p, top_k=args.top_k,
    )
    summarize_ids(tokenizer, hf_greedy, "HF greedy")

    hf_sample = hf_generate(
        model, tokenizer, prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
    )
    summarize_ids(tokenizer, hf_sample, "HF sampled")

    rt_greedy = runtime_generate(
        model, tokenizer, args.model, prompt,
        max_tokens=args.max_tokens, temperature=0.0, top_p=args.top_p, top_k=args.top_k,
        cuda_graph_max_bs=0,
    )
    summarize_ids(tokenizer, rt_greedy, "Runtime greedy")

    rt_sample_off = runtime_generate(
        model, tokenizer, args.model, prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        cuda_graph_max_bs=0,
    )
    summarize_ids(tokenizer, rt_sample_off, "Runtime sampled graph_off")

    rt_sample_on = runtime_generate(
        model, tokenizer, args.model, prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        cuda_graph_max_bs=args.graph_bs,
    )
    summarize_ids(tokenizer, rt_sample_on, "Runtime sampled graph_on")


if __name__ == "__main__":
    main()
