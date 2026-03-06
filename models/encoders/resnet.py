"""
ResNet encoder for self-supervised histopathology learning.

Provides a thin wrapper around torchvision ResNet models that exposes a
consistent interface with the ViT encoder:

  - ``forward(x)`` → ``(patch_tokens, None)``
  - ``forward_features(x)`` → global average-pooled features
  - ``output_dim`` property

ResNet variants serve as the convolutional baseline in ablation experiments.

Reference:
  He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetEncoder(nn.Module):
    """ResNet feature extractor with a uniform interface.

    Args:
        arch:         One of ``"resnet18"`` | ``"resnet50"`` | ``"resnet101"``.
        pretrained:   Initialise with ImageNet weights (for transfer baseline).
        zero_init_residual: Zero-initialise the last BN in each residual block.
    """

    _OUT_DIMS: Dict[str, int] = {
        "resnet18":  512,
        "resnet34":  512,
        "resnet50":  2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if arch not in self._OUT_DIMS:
            raise ValueError(
                f"Unknown ResNet architecture: {arch!r}.  "
                f"Available: {list(self._OUT_DIMS)}"
            )

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(tvm, arch)(weights=weights,
                                      zero_init_residual=zero_init_residual)

        # Strip the final FC layer; keep the GAP layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self._out_dim = self._OUT_DIMS[arch]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Extract spatial feature maps, flattened as pseudo-token sequences.

        The mask argument is accepted but ignored to maintain API compatibility
        with the ViT encoder.

        Args:
            x:    Float tensor of shape ``(B, C, H, W)``.
            mask: Ignored.

        Returns:
            patch_tokens: Shape ``(B, 1, D)`` — single global feature vector
                          wrapped as a length-1 token sequence.
            None
        """
        feats = self.features(x)      # (B, D, 1, 1)
        feats = feats.flatten(1)      # (B, D)
        return feats.unsqueeze(1), None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return global average-pooled features.

        Args:
            x: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, D)``.
        """
        patch_tokens, _ = self.forward(x)
        return patch_tokens.squeeze(1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


def build_resnet(cfg: dict) -> ResNetEncoder:
    """Instantiate a ResNet encoder from a config dictionary.

    Args:
        cfg: ``model.encoder`` sub-dict from the YAML config.

    Returns:
        Initialised :class:`ResNetEncoder`.
    """
    return ResNetEncoder(
        arch=cfg.get("arch", "resnet50"),
        pretrained=cfg.get("pretrained", False),
        zero_init_residual=cfg.get("zero_init_residual", True),
    )
