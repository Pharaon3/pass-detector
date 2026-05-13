"""
Per-time-step classifier head.

Input:  [B, T, hidden_dim]
Output: [B, T, num_classes]  (raw logits)
"""

from __future__ import annotations

import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
