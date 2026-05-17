# Benchmark: Lite-OPD vs ms-swift

[English](README.md) | [Chinese](README_zh.md)

## Environment

- Hardware: 2× H20 96GB
- ms-swift version: 3.5.1 (using vLLM colocate mode)
- Loss: reverse KL (corresponds to `gkd` + `lmbda=1.0` in ms-swift)
- `global_batch_size=128` (`per_device_train_batch_size=1 × gradient_accumulation_steps=64 × 2 GPUs`)
- `max_completion_length=16384`
- `learning_rate=5e-6`, constant schedule
- ZeRO-2, gradient checkpointing

## Comparison Conditions

Both frameworks use essentially the same training configuration (model, data, hyperparameters, hardware). ms-swift uses vLLM colocate mode (inference and training on the same GPUs), corresponding to Lite-OPD's embedded inference engine architecture.

Lite-OPD configs: [`configs/qwen3_1b7_8b.yaml`](../../configs/qwen3_1b7_8b.yaml) and [`configs/qwen25_1b5_7b.yaml`](../../configs/qwen25_1b5_7b.yaml). ms-swift configs: `run_qwen3_1b7_8b.sh` and `run_qwen25_1b5_7b.sh` in this directory.

## Results

| Configuration | Lite-OPD | ms-swift | Speedup |
|---------------|-----|----------|---------|
| Qwen3-1.7B → Qwen3-8B | ~384s/step | ~474s/step | 1.23× |
| Qwen2.5-1.5B → Qwen2.5-7B | ~93s/step | ~100s/step | 1.08× |

## Reproduction

ms-swift scripts:
- `run_qwen3_1b7_8b.sh`
- `run_qwen25_1b5_7b.sh`

Lite-OPD configs:
- `configs/qwen3_1b7_8b.yaml`
- `configs/qwen25_1b5_7b.yaml`
