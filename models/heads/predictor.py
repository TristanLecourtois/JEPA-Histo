"""
I-JEPA Predictor network.

The predictor is a *narrow* Transformer that operates entirely in latent space.
Given a set of context token embeddings and the positional embeddings of target
tokens, it predicts the target encoder's output at those positions—without ever
seeing the raw target pixels.

Design choices (following Assran et al. 2023):
  - The predictor has fewer heads / layers than the encoder to form a
    representational bottleneck (forces the predictor to ``compress'').
  - Positional queries for target positions are injected as learned
    positional shifts on top of sinusoidal embeddings.
  - No [CLS] token.

Reference:
  Assran et al., "Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture", CVPR 2023.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from models.encoders.vit import Block, get_2d_sincos_pos_embed


class JEPAPredictor(nn.Module):
    """Latent-space predictor for I-JEPA.

    Args:
        num_patches:         Total number of patch tokens in the grid (H*W).
        encoder_embed_dim:   Embedding dimension of the context encoder.
        predictor_embed_dim: Internal hidden dimension of the predictor
                             (narrower than the encoder).
        depth:               Number of Transformer blocks.
        num_heads:           Number of attention heads.
        mlp_ratio:           MLP expansion ratio.
        drop_path_rate:      Stochastic depth rate.
        grid_h:              Number of patch rows.
        grid_w:              Number of patch columns.
        norm_layer:          Normalisation layer constructor.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        predictor_embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        grid_h: int = 12,
        grid_w: int = 12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_patches         = num_patches
        self.grid_h              = grid_h
        self.grid_w              = grid_w

        # Project encoder tokens into predictor space
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Mask / query tokens — one learnable vector per position query
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Fixed sinusoidal positional embedding shared across context & targets
        pos_emb = get_2d_sincos_pos_embed(predictor_embed_dim, grid_h, grid_w)
        self.register_buffer("pos_embed", pos_emb.unsqueeze(0))  # (1, N, D)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(predictor_embed_dim)

        # Project back to encoder embedding space for the MSE loss
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

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
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target token representations from context tokens.

        Args:
            context_tokens: Shape ``(B, N_ctx, D_enc)`` — encoder outputs for
                            context positions.
            context_mask:   Boolean tensor of shape ``(B, N)`` or ``(N,)``
                            marking which token positions are context (True).
            target_mask:    Boolean tensor of shape ``(B, N)`` or ``(N,)``
                            marking which positions are targets (True).

        Returns:
            Predicted embeddings of shape ``(B, N_tgt, D_enc)`` for all
            target positions (in positional order).
        """
        B = context_tokens.shape[0]
        N = self.num_patches

        if context_mask.dim() == 1:
            context_mask = context_mask.unsqueeze(0).expand(B, -1)
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0).expand(B, -1)

        # Project context tokens to predictor dim and add positional embeddings
        ctx = self.predictor_embed(context_tokens)          # (B, N_ctx, D_pred)
        ctx = ctx + self.pos_embed.expand(B, -1, -1)[context_mask].reshape(
            B, -1, self.predictor_embed_dim
        )

        # Build target query tokens (mask token + positional embedding)
        n_tgt = target_mask[0].sum().item()
        queries = self.mask_token.expand(B, n_tgt, -1)      # (B, N_tgt, D_pred)
        queries = queries + self.pos_embed.expand(B, -1, -1)[target_mask].reshape(
            B, n_tgt, self.predictor_embed_dim
        )

        # Concatenate context tokens and target queries, process jointly
        x = torch.cat([ctx, queries], dim=1)                # (B, N_ctx+N_tgt, D_pred)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Extract only the target positions (last N_tgt tokens) and project back
        pred = x[:, ctx.shape[1]:, :]                       # (B, N_tgt, D_pred)
        pred = self.predictor_proj(pred)                    # (B, N_tgt, D_enc)
        return pred


def build_predictor(cfg: dict, encoder_embed_dim: int, grid_h: int, grid_w: int) -> JEPAPredictor:
    """Build the JEPA predictor from a config dictionary.

    Args:
        cfg:               ``model.predictor`` sub-dict from the YAML config.
        encoder_embed_dim: Output dimension of the context encoder.
        grid_h:            Number of patch rows.
        grid_w:            Number of patch columns.

    Returns:
        Initialised :class:`JEPAPredictor`.
    """
    return JEPAPredictor(
        num_patches         = grid_h * grid_w,
        encoder_embed_dim   = encoder_embed_dim,
        predictor_embed_dim = cfg.get("predictor_embed_dim", 192),
        depth               = cfg.get("depth", 6),
        num_heads           = cfg.get("num_heads", 6),
        mlp_ratio           = cfg.get("mlp_ratio", 4.0),
        drop_path_rate      = cfg.get("drop_path_rate", 0.0),
        grid_h              = grid_h,
        grid_w              = grid_w,
    )
