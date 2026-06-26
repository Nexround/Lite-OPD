# Datasets

[English](README.md) | [中文](README_zh.md)

This directory contains sample data subsets for quick validation.

## Sources

### Training Data (`train/`)

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `math_metamath_openr1_1to1_1280.parquet` | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) + [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 1280 | 1:1 mixed sampling from both datasets; fields: `problem`, `solution` |
| `openr1_math_all_1280.parquet` | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 1280 | Full OpenR1 math data with `messages` field (multi-turn conversation format) |

### Evaluation Data (`eval/`)

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `math_hf_200.jsonl` | [MATH](https://huggingface.co/datasets/lighteval/MATH) | 200 | High school math competition problems; fields: `problem`, `solution`, `level`, `type` |
| `omni_math_4_5_200.jsonl` | [OmniMath](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | 200 | Difficulty 4-5 math problem subset |
| `omni_math_5_6_200.jsonl` | [OmniMath](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | 200 | Difficulty 5-6 math problem subset |

## Data Format

Training data must contain `problem` and `solution` fields (or a `messages` field). Both parquet and jsonl formats are supported.

Evaluation data must contain `problem` and `solution` fields (jsonl format). During evaluation, the model generates a response to `problem`, and correctness is determined by comparing against the answer in `solution` using a math verifier.

For complete field specifications, supported file formats, hybrid SFT+OPD requirements, and multi-source mixing, see **[docs/dataset_format.md](../docs/dataset_format.md)**.
