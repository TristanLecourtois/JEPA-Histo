"""
Image JEPA (I-JEPA) for histopathology self-supervised pretraining.

Architecture overview:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Input image                                                    │
  │      │                                                          │
  │      ├──[context mask]──► Context Encoder (ViT)                │
  │      │                         │                                │
  │      │                         ▼                                │
  │      │                   Predictor (narrow ViT)                 │
  │      │                   + target positional queries            │
  │      │                         │                                │
  │      │                         ▼  predicted target embeddings   │
  │      │                                                          │
  │      └──[target mask]───► Target Encoder (EMA of context)      │
  │                                 │                               │
  │                                 ▼  actual target embeddings     │
  │                                                                 │
  │  Loss = MSE(predicted, actual)  [in normalised feature space]  │
  └─────────────────────────────────────────────────────────────────┘

Key properties:
  - Predictions are in *latent* space → no pixel-level reconstruction.
  - Target encoder is an EMA copy → no gradient through targets.
  - Block masking creates structured prediction tasks that encourage
    semantic representations over low-level texture matching.

Reference:
  Assran et al., "Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture", CVPR 2023.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.vit import VisionTransformer, build_vit
from models.heads.predictor import JEPAPredictor, build_predictor
from utils.patching import create_jepa_masks


def _clip_mask_to_min(masks: torch.Tensor) -> torch.Tensor:
    """Clip boolean masks so every batch item has the same number of True tokens.

    Random block sampling can produce slightly different token counts per image.
    Batched boolean-index reshapes require uniform counts, so we drop excess
    True positions (keeping the first min_count ones in raster order).

    Args:
        masks: Bool tensor of shape ``(B, N)``.

    Returns:
        Clipped bool tensor of shape ``(B, N)`` where each row has exactly
        ``min_count`` True values.
    """
    min_count = int(masks.sum(dim=1).min().item())
    clipped = torch.zeros_like(masks)
    for i in range(masks.shape[0]):
        indices = masks[i].nonzero(as_tuple=True)[0][:min_count]
        clipped[i, indices] = True
    return clipped


class IJEPA(nn.Module):
    """Full I-JEPA model: context encoder + EMA target encoder + predictor.

    Args:
        encoder_cfg:   ``model.encoder`` config sub-dict.
        predictor_cfg: ``model.predictor`` config sub-dict.
        masking_cfg:   ``masking`` config sub-dict.
        ema_momentum:  Initial EMA momentum (overridden by scheduler).
    """

    def __init__(
        self,
        encoder_cfg: dict,
        predictor_cfg: dict,
        masking_cfg: dict,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Context encoder (trained with gradients)
        # ------------------------------------------------------------------
        self.context_encoder: VisionTransformer = build_vit(encoder_cfg)

        # ------------------------------------------------------------------
        # Target encoder (EMA, no gradient)
        # ------------------------------------------------------------------
        self.target_encoder: VisionTransformer = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Predictor
        # ------------------------------------------------------------------
        grid_h = grid_w = (
            encoder_cfg.get("image_size", 96) // encoder_cfg.get("patch_size", 8)
        )
        self.predictor: JEPAPredictor = build_predictor(
            predictor_cfg,
            encoder_embed_dim=self.context_encoder.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        self.masking_cfg  = masking_cfg
        self.ema_momentum = ema_momentum
        self.grid_h       = grid_h
        self.grid_w       = grid_w

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update target encoder weights via Exponential Moving Average.

        Args:
            momentum: EMA coefficient ``m``.  Target params are updated as
                      ``θ_t ← m · θ_t + (1 − m) · θ_c``.
        """
        for param_c, param_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_c.data)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _sample_masks(self, batch_size: int, device: torch.device):
        """Sample a fresh pair of context / target masks for one batch.

        Returns:
            context_masks: Boolean tensor ``(B, N)`` — True = context.
            target_masks:  Boolean tensor ``(B, N)`` — True = target.
        """
        mc = self.masking_cfg
        ctx_list, tgt_list = [], []
        for _ in range(batch_size):
            ctx_mask, tgt_mask = create_jepa_masks(
                num_patches_h      = self.grid_h,
                num_patches_w      = self.grid_w,
                context_scale      = tuple(mc.get("context_scale", [0.85, 1.0])),
                context_aspect_ratio = tuple(mc.get("context_aspect_ratio", [0.75, 1.5])),
                target_scale       = tuple(mc.get("target_scale", [0.15, 0.2])),
                target_aspect_ratio  = tuple(mc.get("target_aspect_ratio", [0.75, 1.5])),
                num_target_blocks  = mc.get("num_target_blocks", 4),
                allow_overlap      = mc.get("allow_overlap", False),
            )
            ctx_list.append(ctx_mask)
            tgt_list.append(tgt_mask)

        context_masks = torch.stack(ctx_list).to(device)
        target_masks  = torch.stack(tgt_list).to(device)

        # Clip each mask to the minimum True-count across the batch so that
        # all items have exactly the same number of tokens — required for the
        # batched boolean-indexing reshape in the encoder and predictor.
        context_masks = _clip_mask_to_min(context_masks)
        target_masks  = _clip_mask_to_min(target_masks)

        return context_masks, target_masks

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the I-JEPA self-supervised loss.

        Args:
            images: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Dictionary with keys:
              - ``"loss"``:          Scalar MSE loss.
              - ``"context_masks"``: Boolean ``(B, N)`` context mask.
              - ``"target_masks"``:  Boolean ``(B, N)`` target mask.
        """
        B, _, H, W = images.shape
        device = images.device

        # 1. Sample masks
        context_masks, target_masks = self._sample_masks(B, device)

        # 2. Encode context (gradient flows here)
        ctx_tokens, _ = self.context_encoder(images, mask=context_masks)

        # 3. Encode targets (no gradient)
        with torch.no_grad():
            tgt_tokens, _ = self.target_encoder(images, mask=target_masks)
            # Normalise target representations (standard in I-JEPA)
            tgt_tokens = F.layer_norm(tgt_tokens, tgt_tokens.shape[-1:])

        # 4. Predict target embeddings from context
        pred_tokens = self.predictor(ctx_tokens, context_masks, target_masks)

        # 5. MSE loss (smooth L1 as an alternative is also common)
        loss = F.mse_loss(pred_tokens, tgt_tokens)

        return {
            "loss":          loss,
            "context_masks": context_masks,
            "target_masks":  target_masks,
        }

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global features using the context encoder.

        Args:
            images: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, D)`` — mean of patch tokens.
        """
        return self.context_encoder.forward_features(images)


# ---------------------------------------------------------------------------
# EMA momentum scheduler
# ---------------------------------------------------------------------------

class EMAMomentumScheduler:
    """Cosine schedule for the EMA momentum coefficient.

    Linearly increases momentum from ``m_start`` to ``m_end`` following a
    cosine schedule over the course of training.

    Args:
        model:        The :class:`IJEPA` model.
        m_start:      Initial momentum value.
        m_end:        Final momentum value (typically 1.0).
        total_steps:  Total number of optimisation steps.
    """

    def __init__(
        self,
        model: IJEPA,
        m_start: float = 0.996,
        m_end: float = 1.0,
        total_steps: int = 100_000,
    ) -> None:
        self.model       = model
        self.m_start     = m_start
        self.m_end       = m_end
        self.total_steps = total_steps
        self._step       = 0

    def step(self) -> float:
        """Advance the schedule by one step and update the target encoder.

        Returns:
            Current momentum value.
        """
        t = self._step / max(1, self.total_steps)
        m = self.m_end - (self.m_end - self.m_start) * (math.cos(math.pi * t) + 1) / 2
        self.model.update_target_encoder(m)
        self._step += 1
        return m
