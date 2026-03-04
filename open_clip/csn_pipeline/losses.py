from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss; supports 2+ views."""

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float | None = None):
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = str(contrast_mode)
        self.base_temperature = float(base_temperature if base_temperature is not None else temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim < 3:
            raise ValueError("features must be [batch, n_views, dim]")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        device = features.device

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("labels shape does not match features batch size")

        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"unknown contrast_mode={self.contrast_mode}")

        logits = torch.div(anchor_feature @ contrast_feature.T, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_counts = mask.sum(1)
        if torch.any(pos_counts <= 0):
            raise RuntimeError("SupConLoss has anchors with zero positive samples in batch.")

        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class ImageTextAgreementLoss(nn.Module):
    """Symmetric image-text contrastive loss (CLIP-style)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if image_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError("image_embeddings and text_embeddings must have same batch size")

        img = F.normalize(image_embeddings, dim=-1)
        txt = F.normalize(text_embeddings, dim=-1)

        logits = (img @ txt.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i + loss_t)


def two_view_supcon_loss(loss_fn: SupConLoss, v1: torch.Tensor, v2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    features = torch.stack([F.normalize(v1, dim=-1), F.normalize(v2, dim=-1)], dim=1)
    return loss_fn(features, labels)
