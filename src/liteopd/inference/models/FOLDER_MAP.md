# src/opd/inference/models

## Purpose
Per-architecture model definitions for the inference engine. Each model implements the forward pass using the shared layer building blocks from `liteopd.inference.layers`.

## Key Parts
- `llama.py`: Llama / Llama-2 / Llama-3 decoder.
- `qwen2.py`: Qwen2 / Qwen2.5 decoder.
- `qwen3.py`: Qwen3 decoder.
- `qwen3_moe.py`: Qwen3-MoE decoder (sparse MoE layers).
- `gemma3.py`: Gemma 3 decoder. 4 RMSNorms per layer, per-layer dual RoPE (global/local), GeGLU activation, scaled embedding. Uses standalone `RMSNorm` (not fused) due to 4-norm residual pattern.
- `base.py`: `BaseLLMModel` — abstract base with `forward()` and weight-loading interface.
- `config.py`: `ModelConfig` / `RotaryConfig` — parsed from HF config; architecture-agnostic representation.
- `register.py`: Architecture registry mapping HF `model_type` strings to model classes.
- `weight.py`: Weight loading and name remapping from HF checkpoint format. Applies Gemma3 norm weight transform (+1).
- `utils.py`: Shared sub-layers (`GatedMLP`, `RopeAttn`) reused across architectures.

## Entry Points
- `create_model(model_config)`: instantiates the correct model class on meta device.
- `load_weight(model, model_path)`: loads HF checkpoint weights into the model.
- `ModelConfig.from_hf(hf_config)`: converts a HF config object to `ModelConfig`.

## Outbound Dependencies
- `liteopd.inference.layers`: all layer primitives.
- `liteopd.inference.core`: `Context`, `Batch`.
- `transformers`: HF config parsing only.

## Inbound Dependents
- `liteopd.inference.engine.Engine`: calls `create_model` and `load_weight`.
- `liteopd.runtime.rollout.InProcessRolloutClient`: calls `create_model` for the sglang model.
