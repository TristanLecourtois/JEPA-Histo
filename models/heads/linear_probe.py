"""
Linear evaluation head for frozen encoder representations.

A single linear layer attached to a frozen encoder constitutes the canonical
linear probe protocol.  This module also provides a lightweight MLP probe
for ablation studies comparing linear vs. non-linear transfer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class LinearProbe(nn.Module):
    """Single linear layer classifier on top of frozen features.

    Args:
        in_dim:      Input feature dimension (encoder output dimension).
        num_classes: Number of target classes.
    """

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from pre-extracted features.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.linear(x)


class MLPProbe(nn.Module):
    """Two-layer MLP probe for non-linear transfer evaluation.

    Used in ablation studies to measure the gap between linear and
    non-linear probing, which can indicate representation disentanglement.

    Args:
        in_dim:      Input feature dimension.
        hidden_dim:  Hidden layer width.
        num_classes: Number of target classes.
        dropout:     Dropout probability applied before the output layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
