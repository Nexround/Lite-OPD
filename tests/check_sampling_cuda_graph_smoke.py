"""Smoke test for sampling path with and without CUDA graph.

Purpose:
1. Isolate the v1 in-process rollout sampling path from trainer/loss/teacher.
2. Check whether `temperature > 0` failures correlate with CUDA graph replay.
3. Provide a minimal repro script under `src/v1/tests`.

Usage:
  PYTHONPATH=src .venv-v1/bin/python src/v1/tests/check_sampling_cuda_graph_smoke.py
  PYTHONPATH=src .venv-v1/bin/python src/v1/tests/check_sampling_cuda_graph_smoke.py \
      --model /path/to/model \
      --temperature 0.6 --top-p 0.9 --top-k 20
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteopd.runtime.rollout import InProcessRolloutClient


DEFAULT_PROMPTS = [
    "Compute 2+3. Answer with just the final number.",
    "Compute 7*8. Answer with just the final number.",
    "Compute 15-6. Answer with just the final number.",
    "Compute 12/3. Answer with just the final number.",
]


def qwen_chat_template_kwargs(model_name: str) -> dict | None:
    if "Qwen3" in model_name:
        return {"enable_thinking": True}
    return None


def build_messages_batch(prompts: list[str]) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def release_client(client: InProcessRolloutClient | None) -> None:
    if client is None:
        return
    try:
        client.shutdown()
    finally:
        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_case(
    hf_model,
    tokenizer,
    model_path: str,
    device: torch.device,
    *,
    cuda_graph_max_bs: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    rounds: int,
    memory_ratio: float,
    max_running_req: int,
    attention_backend: str,
) -> dict:
    case_name = "graph_on" if cuda_graph_max_bs > 0 else "graph_off"
    client = None
    t0 = time.time()
    try:
        client = InProcessRolloutClient(
            hf_model=hf_model,
            model_path=model_path,
            device=device,
            tokenizer=tokenizer,
            memory_ratio=memory_ratio,
            max_running_req=max_running_req,
            cuda_graph_max_bs=cuda_graph_max_bs,
            page_size=1,
            attention_backend=attention_backend,
        )
        init_seconds = time.time() - t0
        messages_batch = build_messages_batch(DEFAULT_PROMPTS)
        outputs: list[list[str]] = []
        per_round_seconds: list[float] = []

        for round_idx in range(rounds):
            client.prepare()
            round_t0 = time.time()
            responses = client.generate_messages(
                messages_batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                chat_template_kwargs=qwen_chat_template_kwargs(model_path),
            )
            torch.cuda.synchronize(device)
            per_round_seconds.append(time.time() - round_t0)
            outputs.append(responses)
            client.release()
            print(
                f"[{case_name}] round={round_idx + 1}/{rounds} "
                f"seconds={per_round_seconds[-1]:.2f} "
                f"lengths={[len(r) for r in responses]}"
            )

        return {
            "case": case_name,
            "ok": True,
            "cuda_graph_max_bs": cuda_graph_max_bs,
            "init_seconds": init_seconds,
            "round_seconds": per_round_seconds,
            "outputs": outputs,
        }
    except Exception as exc:
        return {
            "case": case_name,
            "ok": False,
            "cuda_graph_max_bs": cuda_graph_max_bs,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        release_client(client)


def print_case_summary(result: dict) -> None:
    print(f"\n=== {result['case']} ===")
    print(f"ok: {result['ok']}")
    print(f"cuda_graph_max_bs: {result['cuda_graph_max_bs']}")
    if result["ok"]:
        print(f"init_seconds: {result['init_seconds']:.2f}")
        print(f"round_seconds: {[round(x, 2) for x in result['round_seconds']]}")
        first_round = result["outputs"][0]
        for idx, text in enumerate(first_round):
            preview = text.replace("\n", "\\n")[:160]
            print(f"sample[{idx}]: {preview!r}")
    else:
        print(f"error_type: {result['error_type']}")
        print(f"error: {result['error']}")
        print(result["traceback"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/model")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--memory-ratio", type=float, default=0.35)
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument("--attention-backend", default="fi")
    parser.add_argument("--graph-on-bs", type=int, default=8)
    parser.add_argument("--case", choices=["graph_off", "graph_on", "both"], default="both")
    return parser.parse_args()


def _run_single_case_from_args(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0")

    print(f"Loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
    ).to(device)
    hf_model.eval()

    cuda_graph_max_bs = 0 if args.case == "graph_off" else args.graph_on_bs
    print(
        "Test config:",
        {
            "case": args.case,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "rounds": args.rounds,
            "graph_on_bs": args.graph_on_bs,
        },
    )
    result = run_case(
        hf_model,
        tokenizer,
        args.model,
        device,
        cuda_graph_max_bs=cuda_graph_max_bs,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        rounds=args.rounds,
        memory_ratio=args.memory_ratio,
        max_running_req=args.max_running_req,
        attention_backend=args.attention_backend,
    )
    print_case_summary(result)
    return 0 if result["ok"] else 1


def _run_case_subprocess(args: argparse.Namespace, case: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        __file__,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--max-tokens",
        str(args.max_tokens),
        "--rounds",
        str(args.rounds),
        "--memory-ratio",
        str(args.memory_ratio),
        "--max-running-req",
        str(args.max_running_req),
        "--attention-backend",
        args.attention_backend,
        "--graph-on-bs",
        str(args.graph_on_bs),
        "--case",
        case,
    ]
    env = dict(**os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def main() -> None:
    args = parse_args()
    if args.case in {"graph_off", "graph_on"}:
        raise SystemExit(_run_single_case_from_args(args))

    proc_off = _run_case_subprocess(args, "graph_off")
    proc_on = _run_case_subprocess(args, "graph_on")

    print("=== graph_off subprocess stdout ===")
    print(proc_off.stdout)
    if proc_off.stderr:
        print("=== graph_off subprocess stderr ===")
        print(proc_off.stderr)

    print("=== graph_on subprocess stdout ===")
    print(proc_on.stdout)
    if proc_on.stderr:
        print("=== graph_on subprocess stderr ===")
        print(proc_on.stderr)

    print("=== verdict ===")
    if proc_off.returncode == 0 and proc_on.returncode != 0:
        print("Sampling works with CUDA graph OFF but fails with CUDA graph ON.")
    elif proc_off.returncode != 0 and proc_on.returncode == 0:
        print("Sampling fails with CUDA graph OFF but works with CUDA graph ON.")
    elif proc_off.returncode != 0 and proc_on.returncode != 0:
        print("Sampling fails in both modes. Bug is not specific to CUDA graph.")
    else:
        print("Sampling succeeds in both modes for this smoke case.")


if __name__ == "__main__":
    main()
