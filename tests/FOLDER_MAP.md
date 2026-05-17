# tests

## Purpose
Tests and smoke scripts for the Lite-OPD framework. Covers loss correctness, runtime weight sharing, sampling quality, and end-to-end training smoke tests.

## Key Parts
- `test_losses.py`: Verifies `opd.losses` — finiteness, gradient correctness, shape properties.
- `smoke_test.py`: Minimal numerical sanity check for loss functions.
- `run_opd_smoke.py`: End-to-end smoke test that imports and runs `opd.train.run_opd_training`.
- `check_kl_backward_parity.py`: Verifies two KL backward paths (logits vs hidden) produce consistent gradients.
- `check_sglang_offline_engine.py`: Lightweight capability check for the offline inference engine.
- `check_qwen_runtime_sync_smoke.py`: Regression test for fused Qwen weight sharing — catches `q_norm/k_norm` sync issues that cause NaN logits.
- `check_runtime_weight_sync_probe.py`: Diagnostic probe verifying shared tensor aliases and sampled rollout correctness.
- `check_sampling_cuda_graph_smoke.py`: Validates sampling behavior with CUDA graph capture/replay.
- `check_sampling_quality_probe.py`: Compares HF reference vs Lite-OPD runtime token generation.
- `check_vllm_sampling_fragment.py`: Fragment-level validation of native/flashinfer sampler paths.
- `check_vllm_sampling_rollout_fragment.py`: Multi-step rollout fragment validation.
- `count_lines.py`: Utility script counting Python and C/C++ lines under `src/opd/`.

## Entry Points
- `PYTHONPATH=src python tests/<script>.py`: run any individual test.
- `PYTHONPATH=src python tests/test_losses.py`: primary unit test for loss functions.
- `PYTHONPATH=src python tests/run_opd_smoke.py --config <yaml>`: end-to-end smoke.

## Outbound Dependencies
- `opd.*`: all tests import from the `opd` package.
- `torch`: tensor operations and gradient checks.

## Notes
- Runtime numerical stability issues should get a dedicated smoke/probe script here rather than debugging logic in the training loop.
- Tests requiring GPU will fail gracefully on CPU-only machines.
