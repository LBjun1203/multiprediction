"""Losses and retrieval metrics for same-timepoint pretraining."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as F


def symmetric_info_nce(
    z_ct: torch.Tensor,
    z_bio: torch.Tensor,
    temperature: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric diagonal-label InfoNCE loss and ``[B,B]`` logits."""
    if z_ct.ndim != 2 or z_bio.ndim != 2 or z_ct.shape != z_bio.shape:
        raise ValueError(
            "z_ct and z_bio must have identical [batch, projection_dim] shapes"
        )
    if z_ct.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two paired samples per batch")
    if z_ct.device != z_bio.device:
        raise ValueError("z_ct and z_bio must be on the same device")
    value = torch.as_tensor(temperature, dtype=torch.float32, device=z_ct.device)
    if value.numel() != 1 or not torch.isfinite(value).item() or value.item() <= 0:
        raise ValueError("temperature must be one finite positive scalar")
    # Compute similarities in float32 even under autocast to keep softmax stable.
    normalized_ct = F.normalize(z_ct.float(), dim=-1)
    normalized_bio = F.normalize(z_bio.float(), dim=-1)
    logits = normalized_ct @ normalized_bio.transpose(0, 1) / value
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_ct = F.cross_entropy(logits, labels)
    loss_bio = F.cross_entropy(logits.transpose(0, 1), labels)
    loss = (loss_ct + loss_bio) * 0.5
    if not torch.isfinite(loss).item():
        raise ValueError("Contrastive loss is non-finite")
    return loss, logits


def retrieval_recalls(
    logits: torch.Tensor, ks: Iterable[int] = (1, 5)
) -> dict[str, float]:
    """Compute paired CT-to-bio and bio-to-CT Recall@K."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Retrieval logits must be a square [B,B] matrix")
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device).view(-1, 1)
    result: dict[str, float] = {}
    for requested in ks:
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise ValueError("Recall K values must be positive integers")
        k = min(requested, batch_size)
        ct_indices = logits.topk(k, dim=1).indices
        bio_indices = logits.transpose(0, 1).topk(k, dim=1).indices
        result[f"ct_to_bio_recall_at_{requested}"] = float(
            ct_indices.eq(labels).any(dim=1).float().mean().item()
        )
        result[f"bio_to_ct_recall_at_{requested}"] = float(
            bio_indices.eq(labels).any(dim=1).float().mean().item()
        )
    return result


def masked_huber_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Robust same-timepoint CT-to-biochemistry regression control loss."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("mask must contain only 0 or 1")
    valid = mask.sum()
    if valid.item() <= 0:
        raise ValueError("masked regression batch contains no observed targets")
    elementwise = F.huber_loss(prediction.float(), target.float(), reduction="none", delta=delta)
    return (elementwise * mask.float()).sum() / valid.float()
