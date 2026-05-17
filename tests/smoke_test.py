from __future__ import annotations

import torch

from liteopd.losses import forward_kl_loss, reverse_kl_loss, jsd_loss


def main() -> None:
    teacher = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    student = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    print("fkl", float(forward_kl_loss(teacher, student)))
    print("rkl", float(reverse_kl_loss(teacher, student)))
    print("jsd", float(jsd_loss(teacher, student)))


if __name__ == "__main__":
    main()
