"""Evaluation utilities for OPD."""
from .scoring import (
    compute_accuracy_with_rollout_client,
    compute_accuracy_with_rollout_client_batched,
    compute_named_accuracies_with_rollout_client_batched,
    distributed_eval,
    extract_final_answer,
    extract_number,
    normalize_answer,
    score_response,
)

__all__ = [
    "extract_number",
    "extract_final_answer",
    "normalize_answer",
    "score_response",
    "compute_accuracy_with_rollout_client",
    "compute_accuracy_with_rollout_client_batched",
    "compute_named_accuracies_with_rollout_client_batched",
    "distributed_eval",
]
