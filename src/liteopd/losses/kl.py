from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12

SUPPORTED_LOSSES = {"forward_kl", "reverse_kl", "jsd"}


# ---------------------------------------------------------------------------
# SFT (cross-entropy) loss — used for the gold-prefix region in hybrid
# SFT+OPD training.
# ---------------------------------------------------------------------------

def sft_loss_from_hidden_chunk(
    student_chunk_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    student_lm_head,
) -> torch.Tensor:
    """Cross-entropy loss for the gold-prefix region in hybrid SFT+OPD training.

    Projects a chunk of student hidden states through the LM head, then
    computes mean next-token cross-entropy against ``target_ids``.

    Args:
        student_chunk_hidden: ``(1, tokens, hidden_dim)`` float tensor.
            Requires grad when called from the two-stage backward path.
        target_ids: ``(tokens,)`` int64 tensor of ground-truth next-token IDs.
            Corresponds to ``input_ids[sft_start+1 : sft_end+1]``.
        student_lm_head: The student's language-model head (``nn.Linear`` or
            equivalent ``ParallelLMHead``).

    Returns:
        Scalar mean cross-entropy loss over the chunk's token positions.
    """
    # (1, tokens, vocab) → (tokens, vocab) for cross_entropy
    student_chunk_logits = student_lm_head(student_chunk_hidden).float().squeeze(0)
    return F.cross_entropy(student_chunk_logits, target_ids.long())


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)


def distillation_loss(teacher_probs: torch.Tensor, student_probs: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"unknown loss: {loss_name}, supported: {SUPPORTED_LOSSES}")
    teacher_probs = _normalize_probs(teacher_probs)
    student_probs = _normalize_probs(student_probs)

    if loss_name == "forward_kl":
        return (teacher_probs * (teacher_probs.clamp_min(EPS).log() - student_probs.clamp_min(EPS).log())).sum(dim=-1).mean()
    if loss_name == "reverse_kl":
        return (student_probs * (student_probs.clamp_min(EPS).log() - teacher_probs.clamp_min(EPS).log())).sum(dim=-1).mean()
    # jsd
    mixture = 0.5 * (teacher_probs + student_probs)
    return 0.5 * forward_kl_loss(teacher_probs, mixture) + 0.5 * forward_kl_loss(student_probs, mixture)


def forward_kl_loss(teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    return distillation_loss(teacher_probs=teacher_probs, student_probs=student_probs, loss_name="forward_kl")


def reverse_kl_loss(teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    return distillation_loss(teacher_probs=teacher_probs, student_probs=student_probs, loss_name="reverse_kl")


def jsd_loss(teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    return distillation_loss(teacher_probs=teacher_probs, student_probs=student_probs, loss_name="jsd")


def chunked_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_name: str,
    chunk_tokens: int = 32,
) -> torch.Tensor:
    seq_len = student_logits.shape[1]
    total_loss = student_logits.new_zeros(())
    total_tokens = 0

    for start in range(0, seq_len, chunk_tokens):
        end = min(seq_len, start + chunk_tokens)
        student_chunk = student_logits[:, start:end, :].float()
        teacher_chunk = teacher_logits[:, start:end, :].float()
        token_count = student_chunk.shape[0] * student_chunk.shape[1]
        if token_count == 0:
            continue

        teacher_probs = F.softmax(teacher_chunk, dim=-1)
        student_probs = F.softmax(student_chunk, dim=-1)
        token_loss = distillation_loss(teacher_probs=teacher_probs, student_probs=student_probs, loss_name=loss_name)

        total_loss = total_loss + token_loss * token_count
        total_tokens += token_count

    if total_tokens == 0:
        return total_loss
    return total_loss / total_tokens


def chunked_kl_from_hidden(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_lm_head,
    teacher_lm_head,
    loss_name: str,
    chunk_tokens: int = 32,
    slice_teacher_logits_to_student: bool = False,
) -> torch.Tensor:
    seq_len = student_hidden.shape[1]
    total_loss = student_hidden.new_zeros(())
    total_tokens = 0

    for start in range(0, seq_len, chunk_tokens):
        end = min(seq_len, start + chunk_tokens)
        token_loss = chunk_loss_from_hidden_chunk(
            student_hidden[:, start:end, :],
            teacher_hidden[:, start:end, :],
            student_lm_head,
            teacher_lm_head,
            loss_name,
            slice_teacher_logits_to_student=slice_teacher_logits_to_student,
        )
        token_count = student_hidden[:, start:end, :].shape[0] * student_hidden[:, start:end, :].shape[1]

        total_loss = total_loss + token_loss * token_count
        total_tokens += token_count

    if total_tokens == 0:
        return total_loss
    return total_loss / total_tokens


def chunk_loss_from_hidden_chunk(
    student_chunk_hidden: torch.Tensor,
    teacher_chunk_hidden: torch.Tensor,
    student_lm_head,
    teacher_lm_head,
    loss_name: str,
    slice_teacher_logits_to_student: bool = False,
) -> torch.Tensor:
    student_chunk_logits = student_lm_head(student_chunk_hidden).float()
    with torch.no_grad():
        teacher_chunk_logits = teacher_lm_head(teacher_chunk_hidden).float()

    if slice_teacher_logits_to_student:
        if teacher_chunk_logits.shape[-1] < student_chunk_logits.shape[-1]:
            raise AssertionError(
                "teacher logits cannot be sliced down to student vocab because teacher vocab is smaller: "
                f"student={student_chunk_logits.shape[-1]}, teacher={teacher_chunk_logits.shape[-1]}"
            )
        teacher_chunk_logits = teacher_chunk_logits[..., : student_chunk_logits.shape[-1]]

    if student_chunk_logits.shape[-1] != teacher_chunk_logits.shape[-1]:
        raise AssertionError(
            "student and teacher logits must share the same vocab dimension for KL computation: "
            f"student={student_chunk_logits.shape[-1]}, teacher={teacher_chunk_logits.shape[-1]}"
        )

    teacher_probs = F.softmax(teacher_chunk_logits, dim=-1)
    student_probs = F.softmax(student_chunk_logits, dim=-1)
    return distillation_loss(teacher_probs=teacher_probs, student_probs=student_probs, loss_name=loss_name)
