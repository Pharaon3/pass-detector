"""
Temporal aggregation over frame features.

Default: Transformer encoder on sequence of 2048-D vectors.
Alternative: lightweight TCN (1D convs) for easy swapping in config.

Input:  [B, T, in_dim]   (in_dim == 2048 from ResNet GAP)
Output: [B, T, hidden_dim]
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, in_dim] -> [B, T, hidden_dim]
        h = self.input_proj(x)
        h = self.pos(h)
        # TransformerEncoder expects src_key_padding_mask [B, T] True = ignore
        return self.encoder(h, src_key_padding_mask=mask)


class TemporalTCN(nn.Module):
    """Dilated 1D conv stack; keeps length T."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        layers = []
        dil = 1
        for _ in range(num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dil,
                        dilation=dil,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            dil = min(dil * 2, 16)
        self.blocks = nn.ModuleList(layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, C] -> conv1d expects [B, C, T]
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = h + blk(h)
        h = h.transpose(1, 2)
        h = self.out_norm(h)
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        return h


class TemporalModel(nn.Module):
    """
    Thin facade so train / infer code can swap transformer vs TCN from config.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        model_type: Literal["transformer", "tcn"] = "transformer",
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        if model_type == "transformer":
            self.core: nn.Module = TemporalTransformer(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        elif model_type == "tcn":
            self.core = TemporalTCN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown temporal model_type: {model_type}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.core(x, mask)
