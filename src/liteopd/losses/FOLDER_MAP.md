# src/opd/losses

## Purpose
Distillation loss implementations: Forward KL, Reverse KL, JSD, and overlap-focused research losses. Provides both direct-probability and chunked (logits/hidden-state) computation paths.

## Key Parts
- `kl.py`: All distillation loss implementations:
  - `FKL / RKL / JSD` baseline losses.
  - `teacher_ratio_sq`: teacher-side ratio control variant.
  - `overlap_*` family: `overlap_linear`, `overlap_power_beta_*`, `overlap_temp_*`, `overlap_log1p_lambda_*`, `overlap_alpha_linear_*`, `overlap_topk_linear_*`.
  - `chunked_kl_from_logits`: chunked loss from pre-computed logits.
  - `chunked_kl_from_hidden` / `chunk_loss_from_hidden_chunk`: two-stage chunk-level backward — Stage 1 computes grad of loss w.r.t. hidden per chunk, Stage 2 backprops accumulated grad through backbone.
  - `slice_teacher_logits_to_student`: truncates teacher vocab to student vocab size.
- `__init__.py`: Exports the loss functions used by the training loop.

## Entry Points
- `distillation_loss(loss_name, ...)`: dispatches to the correct loss by name.
- `chunked_kl_from_hidden(...)` / `chunk_loss_from_hidden_chunk(...)`: used in the two-stage backward path.
- `chunked_kl_from_logits(...)`: used in the direct-logits backward path.

## Outbound Dependencies
- `torch`, `torch.nn.functional`: softmax, log-softmax, tensor operations.

## Inbound Dependents
- `liteopd.train.run_opd_training`: calls loss functions each training step.
- Test scripts (`tests/test_losses.py`): verify correctness and numerical properties.

## Notes
- `reverse_kl_loss` implements `KL(student || teacher)` — argument order matters.
- `slice_teacher_logits_to_student` only handles teacher vocab > student vocab (truncation), not general tokenizer alignment.
