# Dataset Format and Requirements

[English](dataset_format.md) | [中文](dataset_format_zh.md)

## Overview

Lite-OPD supports three dataset file formats and two logical dataset roles (training and evaluation). The sections below document each format, the required and optional fields, and examples you can copy directly.

---

## Supported File Formats

| Format | Path example | Notes |
|---|---|---|
| **JSONL** | `data/train.jsonl` | One JSON object per line. UTF-8 encoded. |
| **Parquet** | `data/train.parquet` | Single file or a **directory** of `.parquet` files. |
| **HuggingFace DatasetDict** | `data/hf_dataset/` | A directory saved with `dataset.save_to_disk()`. Must contain a `"train"` split; may optionally include a `"test"` split. |

The loader is selected automatically based on the path extension and directory contents:

```
path/
├── *.parquet   →  parquet directory mode (files sorted lexicographically)
├── dataset_info.json + state.json  →  HuggingFace DatasetDict
file.parquet    →  single-file parquet
file.jsonl      →  JSONL
```

Any other extension raises `FileNotFoundError`.

---

## Training Dataset

### Required fields

Each example must contain exactly **one prompt field** and exactly **one answer field**:

| Role | Accepted field names (checked in order) |
|---|---|
| **Prompt** | `question`, `problem`, `prompt` |
| **Answer** | `answer`, `solution`, `canonical_solution` |

The loader checks these names in the order listed; the first non-empty value wins. If none is found the example raises a `ValueError` at training time.

### Minimal valid examples

**JSONL — `question` / `answer` style:**

```jsonl
{"question": "What is 12 × 13?", "answer": "156"}
{"question": "Solve: 2x + 3 = 11", "answer": "x = 4"}
```

**JSONL — `problem` / `solution` style:**

```jsonl
{"problem": "If a train travels 60 km/h for 2.5 hours, how far does it go?", "solution": "150 km"}
{"problem": "Find all prime factors of 84.", "solution": "2, 3, 7"}
```

**Parquet** — the same column names apply; each row is one example.

### `messages` field (multi-turn format)

Datasets in OpenR1 / multi-turn conversation format may store the prompt inside a `messages` list instead of a flat field. Example:

```jsonl
{
  "messages": [
    {"role": "user", "content": "Prove that √2 is irrational."},
    {"role": "assistant", "content": "Assume √2 = p/q in lowest terms..."}
  ],
  "solution": "Proof by contradiction ..."
}
```

> **Note:** When a `messages` field is present the framework uses it to build the rollout prompt directly via the chat template. The `solution` / `answer` field is still required for accuracy evaluation.

### Optional fields

| Field | Type | Description |
|---|---|---|
| `_train_dataset_source` | `str` | Labels the origin of the example (e.g., `"metamath"`, `"openr1"`). Used only for per-source logging; has no effect on training. |

### Hybrid SFT + OPD mode (additional field)

When `gold_prefix_field` is set in the config, each example may additionally contain a **gold prefix** field. See [Hybrid SFT + OPD](hybrid_sft_opd.md) for full details.

| Field | Type | Description |
|---|---|---|
| `gold_prefix` *(or whatever name you set in config)* | `str` | The text that anchors the assistant response. The student generates its continuation from the end of this text. Empty or missing values are treated as standard OPD samples (no SFT region). |

**Example with gold prefix:**

```jsonl
{
  "question": "Solve: x² - 5x + 6 = 0",
  "gold_prefix": "Let me factor this expression.\n(x - 2)(x - 3) = 0\nSo",
  "answer": "x = 2 or x = 3"
}
```

---

## Evaluation Dataset

Evaluation datasets follow the same field conventions as training data but are **read-only** — no rollouts are generated from them during training, only accuracy is measured.

### Required fields

| Role | Accepted field names (checked in order) |
|---|---|
| **Prompt** | `question`, `problem`, `prompt` |
| **Gold answer** | `answer`, `solution`, `canonical_solution` |

### Supported formats

Evaluation datasets may be JSONL, Parquet (single file or directory), or HuggingFace DatasetDict. When a DatasetDict is used the `"test"` split is loaded as the evaluation set and `"train"` becomes the training set.

### Example evaluation records

```jsonl
{"problem": "Compute 15! / 13!.", "solution": "210"}
{"problem": "How many ways can 5 books be arranged on a shelf?", "solution": "120"}
```

Metadata fields (`level`, `type`, `difficulty`, etc.) are ignored by the scorer and may be present without issue:

```jsonl
{
  "problem": "Find the sum of all integers from 1 to 100.",
  "solution": "5050",
  "level": "easy",
  "type": "arithmetic"
}
```

---

## Scoring Logic

During evaluation the scorer compares the model's generated response against the gold answer using the following pipeline:

1. **`math_verify` (primary)** — if the `math_verify` package is installed, both the gold answer and the model response are parsed symbolically; the response is marked correct when any parsed prediction matches any parsed gold expression.
2. **Regex fallback** — when `math_verify` is unavailable or parsing fails:
   - The response is searched for a `\boxed{...}` expression (last match wins) or a `Final answer: ...` / `Answer: ...` line.
   - The extracted text is then compared numerically against the gold answer.

This means gold answers should be **exact numeric or symbolic values** (e.g., `"156"`, `"x = 4"`, `"\\frac{1}{2}"`) rather than free-form sentences.

---

## Dataset Size and Limits

The `train_subset_size` and `eval_subset_size` config parameters cap the number of examples loaded:

| Config parameter | Default | Effect |
|---|---|---|
| `train_subset_size` | `-1` (no cap) | Load at most N training examples |
| `eval_subset_size` | `-1` (no cap) | Load at most N evaluation examples |

A `validation_subset_size > 0` splits off the first N training examples as an inline validation set (not drawn from a separate eval file).

---

## Multi-Source Training Datasets

To mix data from multiple sources, concatenate them into a single JSONL or Parquet file and tag each row with `_train_dataset_source`:

```jsonl
{"question": "...", "answer": "...", "_train_dataset_source": "metamath"}
{"problem": "...", "solution": "...", "_train_dataset_source": "openr1"}
```

The training log reports per-source example counts at startup:

```
train dataset sources: {'metamath': 640, 'openr1': 640}
```

---

## Quick Reference

### Minimum fields per mode

| Mode | Required training fields | Required eval fields |
|---|---|---|
| Standard OPD | `question`/`problem`/`prompt` + `answer`/`solution`/`canonical_solution` | same |
| Hybrid SFT+OPD | all of the above + `gold_prefix` (name configurable) | same (gold prefix ignored) |

### Field lookup order (first non-empty wins)

```
Prompt:  question → problem → prompt
Answer:  answer → solution → canonical_solution
```

### Supported dataset paths

```
train_dataset: datasets/train/my_data.jsonl         # JSONL
train_dataset: datasets/train/my_data.parquet       # single Parquet
train_dataset: datasets/train/parquet_dir/          # directory of Parquet files
train_dataset: datasets/train/hf_dataset_dir/       # HuggingFace DatasetDict
```

---

## Bundled Sample Data

The repository ships small subsets for quick validation:

| Path | Rows | Fields | Use |
|---|---|---|---|
| `datasets/train/math_metamath_openr1_1to1_1280.parquet` | 1 280 | `problem`, `solution` | Training |
| `datasets/train/openr1_math_all_1280.parquet` | 1 280 | `messages`, `solution` | Training |
| `datasets/eval/math_hf_200.jsonl` | 200 | `problem`, `solution`, `level`, `type` | Evaluation |
| `datasets/eval/omni_math_4_5_200.jsonl` | 200 | `problem`, `answer` | Evaluation |
| `datasets/eval/omni_math_5_6_200.jsonl` | 200 | `problem`, `answer` | Evaluation |
