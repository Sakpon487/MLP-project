from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Simple 2-layer MLP projection head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SharedCSNMask(nn.Module):
    """Shared CSN mask with sigmoid gating across projection dimensions."""

    def __init__(self, proj_dim: int, mask_init: float = 0.0):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.full((proj_dim,), float(mask_init), dtype=torch.float32))

    def mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask().unsqueeze(0)
