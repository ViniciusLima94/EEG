"""
Causal Conv1D model for EEG sliding-window classification.

Usage
-----
from src.models.conv1d import CausalConv1D, make_model
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Trim(nn.Module):
    """Remove n timesteps from the right to enforce causal padding."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.n] if self.n > 0 else x


class CausalConv1D(nn.Module):
    """
    Stacked causal 1-D convolutional classifier.

    Input  : (batch, C, T)  — channels first, causal window
    Output : (batch,)       — raw logit (use sigmoid for probability)

    Architecture per layer
    ----------------------
    Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size-1)
    → _Trim(kernel_size-1)   # drop right-side padding → causal
    → [BatchNorm1d]
    → Activation
    → [Dropout]

    After all conv layers:
    AdaptiveAvgPool1d(1) → flatten → Linear(last_ch, 1)
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        n_layers: int,
        kernel_size: int,
        grow_filters: bool,
        dropout_rate: float,
        activation: str,
        use_bn: bool,
    ):
        super().__init__()
        Act = self.ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_ch = in_channels

        for i in range(n_layers):
            out_ch = min(n_filters * (2**i) if grow_filters else n_filters, 256)
            padding = kernel_size - 1
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                _Trim(padding),
            ]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(Act())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = self.conv_stack(x)                            # (B, last_ch, T)
        return self.head(self.pool(x).squeeze(-1)).squeeze(1)  # (B,)


def make_model(params: dict) -> CausalConv1D:
    """
    Reconstruct a CausalConv1D from a params dict (as saved to JSON).

    Expected keys: n_channels, n_filters, n_layers, kernel_size,
                   grow_filters, dropout, activation, use_bn.
    """
    return CausalConv1D(
        in_channels=params["n_channels"],
        n_filters=params["n_filters"],
        n_layers=params["n_layers"],
        kernel_size=params["kernel_size"],
        grow_filters=params["grow_filters"],
        dropout_rate=params["dropout"],
        activation=params["activation"],
        use_bn=params["use_bn"],
    )
