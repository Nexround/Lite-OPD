"""Dataset loading utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def load_examples_from_path(path: Path, limit: int):
    """Load training/eval examples from a dataset path.

    Supports: directories of parquet files, HF DatasetDict on disk,
    single .parquet files, and .jsonl files.

    Returns (train_rows, test_rows). test_rows is empty unless the path
    is a DatasetDict with a "test" split.
    """
    effective_limit = None if limit is None or limit < 0 else limit
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            rows = []
            for parquet_file in parquet_files:
                rows.extend(pd.read_parquet(parquet_file).to_dict(orient="records"))
                if effective_limit is not None and len(rows) >= effective_limit:
                    break
            return rows[:effective_limit] if effective_limit is not None else rows, []
        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys"):
            train = list(dataset["train"])[:effective_limit] if "train" in dataset and effective_limit is not None else (list(dataset["train"]) if "train" in dataset else [])
            test = list(dataset["test"])[:effective_limit] if "test" in dataset and effective_limit is not None else (list(dataset["test"]) if "test" in dataset else [])
            return train, test
        rows = list(dataset)[:effective_limit] if effective_limit is not None else list(dataset)
        return rows, []
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
        return rows[:effective_limit] if effective_limit is not None else rows, []
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if effective_limit is not None and len(rows) >= effective_limit:
                    break
        return rows, []
    raise FileNotFoundError(f"unsupported dataset path: {path}")


def summarize_train_dataset_sources(examples: list[dict]) -> dict[str, int]:
    """Count examples by their _train_dataset_source field."""
    counts: dict[str, int] = {}
    for example in examples:
        source = example.get("_train_dataset_source")
        if not source:
            continue
        counts[str(source)] = counts.get(str(source), 0) + 1
    return counts


def limit_examples(examples: list[dict], limit: int) -> list[dict]:
    """Truncate examples list to at most `limit` entries."""
    if limit is None or limit < 0:
        return list(examples)
    return list(examples[:limit])


def split_validation_examples(train_examples: list[dict], validation_subset_size: int) -> tuple[list[dict], list[dict]]:
    """Split off the first `validation_subset_size` examples as validation set."""
    if validation_subset_size == 0:
        return list(train_examples), []
    if validation_subset_size >= len(train_examples):
        raise ValueError(
            "validation_subset_size must be smaller than the available train example count "
            f"after train_subset_size filtering: validation_subset_size={validation_subset_size}, "
            f"train_examples={len(train_examples)}"
        )
    return list(train_examples[validation_subset_size:]), list(train_examples[:validation_subset_size])
