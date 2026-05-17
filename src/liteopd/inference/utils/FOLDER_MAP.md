# src/opd/inference/utils

## Purpose
Shared utilities for the inference engine: logging, tokenizer loading, math helpers, and misc torch utilities.

## Key Parts
- `logger.py`: `init_logger()` — returns a logger with `debug_rank0` / `info_rank0` / `warning_rank0` methods that suppress output on non-primary TP ranks.
- `hf.py`: `cached_load_hf_config()` — loads and caches HF `AutoConfig` by model path.
- `arch.py`: `load_tokenizer()` — loads HF tokenizer with caching.
- `misc.py`: `div_ceil`, `div_even`, `align_down`, `is_sm90_supported`, `is_sm100_supported`.
- `registry.py`: Generic class registry used by model and backend factories.
- `torch_utils.py`: `torch_dtype()` context manager, `nvtx_annotate`.

## Entry Points
- `init_logger(__name__)`: used in every submodule.
- `load_tokenizer(model_path)`: used by `Scheduler.__init__` and `InProcessRolloutClient`.
- `cached_load_hf_config(model_path)`: used by `InProcessRolloutClient` and `Engine`.

## Outbound Dependencies
- `torch`, `transformers`.

## Inbound Dependents
- All `liteopd.inference.*` submodules.
