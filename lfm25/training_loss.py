"""Loss utilities for completion-only causal language-model training."""

from __future__ import annotations

from typing import Any


LOSS_MODE = "per_example_completion_mean"


def normalized_completion_cross_entropy(
    logits: Any,
    labels: Any,
    *,
    sample_weight: Any | None = None,
    first_supervised_token_weight: float = 1.0,
) -> Any:
    """Average causal completion loss per row, then average rows by sample weight."""

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("labels must have shape [batch, sequence] matching logits")
    if not math_is_finite_positive(first_supervised_token_weight):
        raise ValueError("first_supervised_token_weight must be finite and greater than zero")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].to(device=logits.device)
    supervised = shifted_labels.ne(-100)
    supervised_counts = supervised.sum(dim=1)
    if bool(supervised_counts.eq(0).any().item()):
        raise ValueError("every example must have at least one supervised completion token")

    token_losses = functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=-100,
    )
    token_weights = supervised.to(dtype=token_losses.dtype)
    if first_supervised_token_weight != 1.0:
        first_positions = supervised.to(dtype=torch.int64).argmax(dim=1)
        row_indices = torch.arange(supervised.shape[0], device=supervised.device)
        token_weights[row_indices, first_positions] = first_supervised_token_weight
    row_losses = (token_losses * token_weights).sum(dim=1) / token_weights.sum(dim=1)

    if sample_weight is None:
        weights = torch.ones_like(row_losses)
    else:
        weights = torch.as_tensor(sample_weight, device=row_losses.device, dtype=row_losses.dtype)
        if weights.ndim != 1 or weights.shape[0] != row_losses.shape[0]:
            raise ValueError("sample_weight must contain one value per example")
        if bool((~torch.isfinite(weights)).any().item()) or bool(weights.lt(0).any().item()):
            raise ValueError("sample_weight values must be finite and non-negative")
    weight_sum = weights.sum()
    if not bool(torch.isfinite(weight_sum).item()) or bool(weight_sum.le(0).item()):
        raise ValueError("sample_weight must have a positive sum")
    return (row_losses * weights).sum() / weight_sum


def math_is_finite_positive(value: float) -> bool:
    import math

    return math.isfinite(value) and value > 0
