# Datasets

[English](README.md) | [中文](README_zh.md)

本目录包含用于快速验证的示例数据子集。

## 来源

### 训练数据（`train/`）

| 文件 | 来源 | 条数 | 说明 |
|------|------|------|------|
| `math_metamath_openr1_1to1_1280.parquet` | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) + [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 1280 | 两数据集 1:1 混合采样，字段：`problem`, `solution` |
| `openr1_math_all_1280.parquet` | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 1280 | 完整 OpenR1 数学数据，含 `messages` 字段（多轮对话格式） |

### 评测数据（`eval/`）

| 文件 | 来源 | 条数 | 说明 |
|------|------|------|------|
| `math_hf_200.jsonl` | [MATH](https://huggingface.co/datasets/lighteval/MATH) | 200 | 高中数学竞赛题，字段：`problem`, `solution`, `level`, `type` |
| `omni_math_4_5_200.jsonl` | [OmniMath](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | 200 | 难度 4-5 的数学题子集 |
| `omni_math_5_6_200.jsonl` | [OmniMath](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | 200 | 难度 5-6 的数学题子集 |

## 数据格式

训练数据需包含 `problem` 和 `solution` 字段（或 `messages` 字段）。支持 parquet 和 jsonl 格式。

评测数据需包含 `problem` 和 `solution` 字段（jsonl 格式）。评测时模型对 `problem` 生成回答，通过数学验证器与 `solution` 中的答案比对判定正确性。

完整的字段规范、支持的文件格式、混合 SFT+OPD 要求及多来源数据混合，请参阅 **[docs/dataset_format_zh.md](../docs/dataset_format_zh.md)**。
