# src/opd/inference/attention

## Purpose
Attention backend abstraction: selects and dispatches to the appropriate attention kernel (FlashAttention 2/3, TRT-LLM, or fallback) based on hardware and configuration.

## Key Parts
- `base.py`: `BaseAttnBackend`, `BaseAttnMetadata` — abstract interfaces all backends implement.
- `fa.py`: FlashAttention backend (FA2 / FA3 via `flash_attn`).
- `fi.py`: FlashInfer backend.
- `trtllm.py`: TensorRT-LLM attention backend.
- `utils.py`: Shared helpers (e.g. metadata construction).
- `__init__.py`: `create_attention_backend(config)` factory; auto-selects based on SM version and available libraries.

## Entry Points
- `create_attention_backend(config)`: called by `Engine.__init__`.
- `BaseAttnBackend.prepare_metadata(batch)`: called by `Scheduler._prepare_batch` before each forward.
- `BaseAttnBackend.forward(...)`: called inside model layer forward.

## Outbound Dependencies
- Optional: `flash_attn`, `tensorrt_llm`.
- `liteopd.inference.core`: `Batch`, `Context`.

## Inbound Dependents
- `liteopd.inference.engine.Engine`: constructs and holds the backend.
- `liteopd.inference.layers.attention.AttentionLayer`: calls `forward` during model execution.
