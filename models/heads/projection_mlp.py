from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.encoders.vit import Block, get_2d_sincos_pos_embed


class DINOHead(nn.Module):
    """DINO projection head.

    Implements the 3-layer MLP + weight-normalised linear prototype layer
    described in Caron et al. (2021).  The last layer uses weight normalisation
    instead of full Batch Norm to ensure the prototype vectors stay on the
    unit hypersphere.

    Args:
        in_dim:          Input feature dimension.
        out_dim:         Number of prototype dimensions (typically 65 536).
        hidden_dim:      Width of hidden MLP layers.
        bottleneck_dim:  Bottleneck dimension before the prototype layer.
        nlayers:         Number of MLP layers (including bottleneck).
        norm_last_layer: Freeze the norm of the last layer prototype vectors.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        nlayers = max(1, nlayers)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            for _ in range(nlayers - 2):
                layers += [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # Weight-normalised prototype layer
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad_(False)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to prototype space.

        Args:
            x: Shape ``(B, D)``.

        Returns:
            Logit tensor of shape ``(B, out_dim)``.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MAEDecoder(nn.Module):
    """Lightweight Transformer decoder for MAE pixel reconstruction.

    The decoder is deliberately shallower and narrower than the encoder.
    It reconstructs normalised pixel values for masked patch positions.

    Args:
        num_patches:      Total number of patch tokens.
        encoder_embed_dim: Embedding dimension of the encoder.
        decoder_embed_dim: Hidden dimension of the decoder.
        decoder_depth:     Number of Transformer blocks.
        decoder_num_heads: Number of attention heads.
        mlp_ratio:         MLP expansion ratio.
        patch_size:        Patch side length in pixels.
        in_channels:       Number of image channels.
        norm_layer:        Normalisation layer constructor.
        grid_h:            Number of patch rows.
        grid_w:            Number of patch columns.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        in_channels: int = 3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        grid_h: int = 12,
        grid_w: int = 12,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        pos_emb = get_2d_sincos_pos_embed(
            decoder_embed_dim, grid_h, grid_w, cls_token=True
        )
        self.register_buffer("decoder_pos_embed", pos_emb.unsqueeze(0))

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_channels
        )

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct pixel values for all patch positions.

        Args:
            x:           Encoded (visible) patch tokens of shape
                         ``(B, N_keep + 1, D_enc)`` (includes [CLS] at index 0).
            ids_restore: Permutation tensor of shape ``(B, N)`` that restores
                         the original token order.

        Returns:
            Reconstructed pixel patches of shape
            ``(B, N, patch_size**2 * C)``.
        """
        x = self.decoder_embed(x)               # (B, N_keep+1, D_dec)

        # Append mask tokens for missing positions
        n_mask  = ids_restore.shape[1] - (x.shape[1] - 1)  # exclude CLS
        mask_tokens = self.mask_token.expand(x.shape[0], n_mask, -1)
        x_no_cls    = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, N, D_dec)
        x_no_cls    = torch.gather(
            x_no_cls, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_no_cls.shape[-1])
        )
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)      # re-attach CLS

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]                         # remove CLS token
        return x


class ProjectionMLP(nn.Module):
    """Two-layer projection MLP (SimCLR / Barlow-Twins style).

    Args:
        in_dim:     Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim:    Output projection dimension.
        use_bn:     Apply Batch Normalisation after each linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 128,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        layers: list = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
