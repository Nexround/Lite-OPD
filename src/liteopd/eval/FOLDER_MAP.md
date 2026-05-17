# src/opd/eval

## Purpose
Evaluation and scoring utilities for math/reasoning tasks. Provides answer extraction, normalization, scoring (with optional `math_verify` library), and distributed evaluation across DP ranks.

## Key Parts
- `scoring.py`: All evaluation logic:
  - `extract_number(text)`: Extracts the last numeric value from text.
  - `extract_final_answer(text)`: Extracts answer from "Final answer:", "Answer:", or `\boxed{}` patterns.
  - `normalize_answer(text)`: Strips whitespace and `$` for comparison.
  - `score_response(example, response, get_gold_text)`: Scores a response against gold. Uses `math_verify` if available, falls back to string matching.
  - `compute_accuracy_with_rollout_client_batched(...)`: Batch-evaluates accuracy using the rollout client for generation. Accepts `build_messages`, `get_prompt_text`, `get_gold_text`, `qwen_chat_template_kwargs` as dependency-injected callables.
  - `compute_named_accuracies_with_rollout_client_batched(...)`: Evaluates multiple named splits in a single flat batch, returns per-split metrics including `avg_gen_tokens`.
  - `distributed_eval(...)`: Shards examples across DP ranks, evaluates locally, then all-reduces correct/total counts.
- `__init__.py`: Re-exports all public functions from `scoring.py`.

## Entry Points
- `from liteopd.eval import score_response, distributed_eval`: primary imports.
- Eval functions are designed for dependency injection — they receive prompt-building callables rather than importing them, so they stay decoupled from the training module.

## Outbound Dependencies
- `torch`, `torch.distributed`: for all-reduce in `distributed_eval`.
- Optional `math_verify`: symbolic math verification (graceful fallback to string matching).

## Inbound Dependents
- `liteopd.train.run_opd_training`: thin wrappers bind prompt helpers and call these functions.

## Notes
- `OPD_EVAL_MAX_NEW_TOKENS` env var overrides the `max_new_tokens` parameter.
- `OPD_EVAL_REQUEST_BATCH_SIZE` env var controls sub-batching of eval requests (useful for memory-constrained setups).
