"""
Causal Conv1D models for EEG sliding-window classification.

Usage
-----
from src.model.conv1d import CausalConv1D, DSCausalConv1D, make_model
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


class _DSBlock(nn.Module):
    """
    One depthwise-separable causal conv block.

    Step 1 — depthwise temporal:
        Conv1d(C, C×D, k, groups=C) + causal trim + [BN] + act
        Each input channel gets D independent temporal filters.

    Step 2 — pointwise spatial:
        Conv1d(C×D, out_ch, 1) + [BN] + act + [Dropout]
        Mixes the C×D feature maps into out_ch spatial combinations.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        depth_mult: int,
        dropout_rate: float,
        activation: str,
        use_bn: bool,
    ):
        super().__init__()
        Act = self.ACTIVATIONS[activation]
        pad = kernel_size - 1
        mid_ch = in_ch * depth_mult

        dw: list[nn.Module] = [
            nn.Conv1d(in_ch, mid_ch, kernel_size=kernel_size, padding=pad, groups=in_ch),
            _Trim(pad),
        ]
        if use_bn:
            dw.append(nn.BatchNorm1d(mid_ch))
        dw.append(Act())
        self.dw = nn.Sequential(*dw)

        pw: list[nn.Module] = [nn.Conv1d(mid_ch, out_ch, kernel_size=1)]
        if use_bn:
            pw.append(nn.BatchNorm1d(out_ch))
        pw.append(Act())
        if dropout_rate > 0:
            pw.append(nn.Dropout(dropout_rate))
        self.pw = nn.Sequential(*pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSCausalConv1D(nn.Module):
    """
    Depthwise-separable causal Conv1D classifier for EEG.

    Input  : (batch, C, T)  — channels first, causal window
    Output : (batch,)       — raw logit

    Each block:
      depthwise temporal conv (D filters per channel, causal)
      → pointwise spatial mix (learned spatial combination)

    Fewer parameters than standard Conv1D; closer to LGB's per-channel
    filter-bank + linear combination structure.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        n_layers: int,
        kernel_size: int,
        depth_mult: int = 2,
        grow_filters: bool = True,
        dropout_rate: float = 0.3,
        activation: str = "gelu",
        use_bn: bool = True,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = in_channels

        for i in range(n_layers):
            out_ch = min(n_filters * (2**i) if grow_filters else n_filters, 256)
            blocks.append(
                _DSBlock(in_ch, out_ch, kernel_size, depth_mult,
                         dropout_rate, activation, use_bn)
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = self.blocks(x)                               # (B, last_ch, T)
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
