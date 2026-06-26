# 数据集格式与要求

[English](dataset_format.md) | [中文](dataset_format_zh.md)

## 概述

Lite-OPD 支持三种数据集文件格式，以及两种逻辑角色（训练集和评估集）。以下各节详细说明了每种格式、必须字段与可选字段，以及可直接复制使用的示例。

---

## 支持的文件格式

| 格式 | 路径示例 | 说明 |
|---|---|---|
| **JSONL** | `data/train.jsonl` | 每行一个 JSON 对象，UTF-8 编码。 |
| **Parquet** | `data/train.parquet` | 单文件或**目录**下的多个 `.parquet` 文件。 |
| **HuggingFace DatasetDict** | `data/hf_dataset/` | 通过 `dataset.save_to_disk()` 保存的目录，必须包含 `"train"` split，可选包含 `"test"` split。 |

加载器根据路径扩展名和目录内容自动选择格式：

```
path/
├── *.parquet                            →  parquet 目录模式（文件按字典序排序）
├── dataset_info.json + state.json       →  HuggingFace DatasetDict
file.parquet                             →  单 parquet 文件
file.jsonl                               →  JSONL
```

其他任何扩展名都会抛出 `FileNotFoundError`。

---

## 训练数据集

### 必须字段

每条样本必须包含**恰好一个提示字段**和**恰好一个答案字段**：

| 角色 | 接受的字段名（按优先级顺序检查） |
|---|---|
| **提示（Prompt）** | `question`、`problem`、`prompt` |
| **答案（Answer）** | `answer`、`solution`、`canonical_solution` |

加载器按上表顺序检查字段名，取第一个非空值。如果找不到对应字段，训练时会抛出 `ValueError`。

### 最简合法样本示例

**JSONL — `question` / `answer` 风格：**

```jsonl
{"question": "12 × 13 等于多少？", "answer": "156"}
{"question": "求解：2x + 3 = 11", "answer": "x = 4"}
```

**JSONL — `problem` / `solution` 风格：**

```jsonl
{"problem": "火车以 60 km/h 的速度行驶 2.5 小时，走了多远？", "solution": "150 km"}
{"problem": "求 84 的所有质因数。", "solution": "2, 3, 7"}
```

**Parquet** — 列名与 JSONL 相同，每行对应一条样本。

### `messages` 字段（多轮对话格式）

采用 OpenR1 / 多轮对话格式的数据集，可以将提示内容放在 `messages` 列表中，而不是扁平字段。示例：

```jsonl
{
  "messages": [
    {"role": "user", "content": "证明 √2 是无理数。"},
    {"role": "assistant", "content": "假设 √2 = p/q（最简分数）..."}
  ],
  "solution": "反证法证明 ..."
}
```

> **注意：** 当存在 `messages` 字段时，框架直接通过聊天模板将其用于构建 rollout 提示。`solution` / `answer` 字段仍需存在，用于精度评估。

### 可选字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `_train_dataset_source` | `str` | 标记样本来源（如 `"metamath"`、`"openr1"`），仅用于分来源日志，不影响训练。 |

### 混合 SFT + OPD 模式（附加字段）

当配置文件中设置了 `gold_prefix_field` 时，每条样本可额外包含一个**金标前缀**字段。完整说明请参见 [混合 SFT + OPD](hybrid_sft_opd_zh.md)。

| 字段 | 类型 | 说明 |
|---|---|---|
| `gold_prefix`（或配置中指定的字段名） | `str` | 锚定助手回复的文本。学生模型从该文本末尾开始生成延续内容。为空或缺失时，该样本退化为标准 OPD 样本（无 SFT 区域）。 |

**含金标前缀的示例：**

```jsonl
{
  "question": "求解：x² - 5x + 6 = 0",
  "gold_prefix": "对该表达式进行因式分解。\n(x - 2)(x - 3) = 0\n因此",
  "answer": "x = 2 或 x = 3"
}
```

---

## 评估数据集

评估数据集与训练数据集使用相同的字段约定，但为**只读**——训练期间不会对其进行 rollout，只计算精度。

### 必须字段

| 角色 | 接受的字段名（按优先级顺序检查） |
|---|---|
| **提示（Prompt）** | `question`、`problem`、`prompt` |
| **标准答案** | `answer`、`solution`、`canonical_solution` |

### 支持的格式

评估数据集可以是 JSONL、Parquet（单文件或目录）或 HuggingFace DatasetDict。若使用 DatasetDict，`"test"` split 会被加载为评估集，`"train"` split 则用作训练集。

### 评估样本示例

```jsonl
{"problem": "计算 15! / 13!。", "solution": "210"}
{"problem": "5 本书在书架上有多少种排列方式？", "solution": "120"}
```

元数据字段（`level`、`type`、`difficulty` 等）会被评分器忽略，允许存在：

```jsonl
{
  "problem": "求 1 到 100 所有整数之和。",
  "solution": "5050",
  "level": "简单",
  "type": "算术"
}
```

---

## 评分逻辑

评估时，评分器通过以下流程将模型生成的回复与标准答案进行比对：

1. **`math_verify`（首选）** — 若已安装 `math_verify` 包，将对标准答案和模型回复分别进行符号解析；任意预测结果与任意标准答案匹配即视为正确。
2. **正则回退** — 当 `math_verify` 不可用或解析失败时：
   - 在回复中搜索 `\boxed{...}` 表达式（取最后一个匹配项），或 `Final answer: ...` / `Answer: ...` 格式的行。
   - 提取文本后与标准答案进行数值比对。

因此，标准答案应为**精确的数值或符号值**（如 `"156"`、`"x = 4"`、`"\\frac{1}{2}"`），而非自由格式的句子。

---

## 数据集大小与限制

配置参数 `train_subset_size` 和 `eval_subset_size` 限制加载的样本数量：

| 配置参数 | 默认值 | 效果 |
|---|---|---|
| `train_subset_size` | `-1`（不限制） | 最多加载 N 条训练样本 |
| `eval_subset_size` | `-1`（不限制） | 最多加载 N 条评估样本 |

若 `validation_subset_size > 0`，则会从训练样本中切分出前 N 条作为内联验证集（不需要单独的评估文件）。

---

## 多来源混合训练

若需混合多个来源的数据，可将它们合并为一个 JSONL 或 Parquet 文件，并为每行添加 `_train_dataset_source` 标记：

```jsonl
{"question": "...", "answer": "...", "_train_dataset_source": "metamath"}
{"problem": "...", "solution": "...", "_train_dataset_source": "openr1"}
```

训练日志在启动时会汇报各来源的样本数：

```
train dataset sources: {'metamath': 640, 'openr1': 640}
```

---

## 快速参考

### 各模式最少必填字段

| 模式 | 训练集必填字段 | 评估集必填字段 |
|---|---|---|
| 标准 OPD | `question`/`problem`/`prompt` + `answer`/`solution`/`canonical_solution` | 同左 |
| 混合 SFT+OPD | 以上全部 + `gold_prefix`（字段名可配置） | 同左（金标前缀被忽略） |

### 字段查找顺序（取第一个非空值）

```
提示：question → problem → prompt
答案：answer → solution → canonical_solution
```

### 支持的数据集路径格式

```yaml
train_dataset: datasets/train/my_data.jsonl          # JSONL
train_dataset: datasets/train/my_data.parquet        # 单 Parquet 文件
train_dataset: datasets/train/parquet_dir/           # Parquet 目录
train_dataset: datasets/train/hf_dataset_dir/        # HuggingFace DatasetDict
```

---

## 内置示例数据

仓库内置了少量样本，可用于快速验证：

| 路径 | 行数 | 字段 | 用途 |
|---|---|---|---|
| `datasets/train/math_metamath_openr1_1to1_1280.parquet` | 1 280 | `problem`、`solution` | 训练 |
| `datasets/train/openr1_math_all_1280.parquet` | 1 280 | `messages`、`solution` | 训练 |
| `datasets/eval/math_hf_200.jsonl` | 200 | `problem`、`solution`、`level`、`type` | 评估 |
| `datasets/eval/omni_math_4_5_200.jsonl` | 200 | `problem`、`answer` | 评估 |
| `datasets/eval/omni_math_5_6_200.jsonl` | 200 | `problem`、`answer` | 评估 |
