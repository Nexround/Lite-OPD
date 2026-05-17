from __future__ import annotations

import torch
import torch.nn.functional as F

from liteopd.losses import distillation_loss, forward_kl_loss, jsd_loss, reverse_kl_loss


def test_losses_are_finite() -> None:
    teacher = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    student = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    assert torch.isfinite(forward_kl_loss(teacher, student))
    assert torch.isfinite(reverse_kl_loss(teacher, student))
    assert torch.isfinite(jsd_loss(teacher, student))


def test_forward_kl_gradient() -> None:
    teacher_probs = torch.tensor([[0.62, 0.27, 0.11]], dtype=torch.float64)
    student_logits = torch.tensor([[0.35, -0.15, -0.40]], dtype=torch.float64, requires_grad=True)
    student_probs = F.softmax(student_logits, dim=-1)
    loss = distillation_loss(teacher_probs, student_probs, "forward_kl")
    grad = torch.autograd.grad(loss, student_logits)[0]
    expected = student_probs - teacher_probs
    assert torch.allclose(grad, expected, atol=1e-10, rtol=1e-8)


def test_reverse_kl_gradient() -> None:
    teacher_probs = torch.tensor([[0.62, 0.27, 0.11]], dtype=torch.float64)
    student_logits = torch.tensor([[0.35, -0.15, -0.40]], dtype=torch.float64, requires_grad=True)
    student_probs = F.softmax(student_logits, dim=-1)
    loss = distillation_loss(teacher_probs, student_probs, "reverse_kl")
    grad = torch.autograd.grad(loss, student_logits)[0]
    log_ratio = student_probs.log() - teacher_probs.log()
    baseline = (student_probs * (log_ratio + 1)).sum(dim=-1, keepdim=True)
    expected = student_probs * (log_ratio + 1) - student_probs * baseline
    assert torch.allclose(grad, expected, atol=1e-10, rtol=1e-8)


def test_jsd_bounds() -> None:
    teacher = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    student = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    jsd = jsd_loss(teacher, student)
    fkl = forward_kl_loss(teacher, student)
    assert jsd.item() >= 0.0
    assert jsd.item() <= fkl.item()


def test_invalid_loss_raises() -> None:
    teacher = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    student = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    try:
        distillation_loss(teacher, student, "overlap_linear")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_losses_are_finite()
    test_forward_kl_gradient()
    test_reverse_kl_gradient()
    test_jsd_bounds()
    test_invalid_loss_raises()
    print("all tests passed")
