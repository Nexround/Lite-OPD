# src/opd/data

## Purpose
Dataset loading and preprocessing utilities for training and evaluation. Handles reading from multiple formats (parquet, JSONL, HF DatasetDict) and splitting into train/validation sets.

## Key Parts
- `loading.py`: Core data loading functions:
  - `load_examples_from_path(path, limit)`: Loads examples from a directory of parquets, HF DatasetDict, single parquet, or JSONL. Returns `(train_rows, test_rows)`.
  - `summarize_train_dataset_sources(examples)`: Counts examples by `_train_dataset_source` field.
  - `limit_examples(examples, limit)`: Truncates a list to at most `limit` entries.
  - `split_validation_examples(train_examples, validation_subset_size)`: Splits off the first N examples as a validation set.
- `__init__.py`: Re-exports all public functions from `loading.py`.

## Entry Points
- `from liteopd.data import load_examples_from_path`: primary import used by the training script.

## Outbound Dependencies
- `pandas`: parquet file reading.
- `datasets` (`load_from_disk`): HF DatasetDict loading.

## Inbound Dependents
- `liteopd.train.run_opd_training`: loads train/test examples at startup.
