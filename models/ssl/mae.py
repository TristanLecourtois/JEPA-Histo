
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.vit import VisionTransformer, build_vit
from models.heads.projection_mlp import MAEDecoder
from utils.patching import patchify, create_mae_mask


class MAE(nn.Module):
    """Full MAE model: ViT encoder + lightweight Transformer decoder.

    Args:
        encoder_cfg:  ``model.encoder`` config sub-dict.
        decoder_cfg:  ``model.decoder`` config sub-dict.
        mae_cfg:      ``mae`` config sub-dict (mask_ratio, norm_pix_loss).
    """

    def __init__(
        self,
        encoder_cfg: dict,
        decoder_cfg: dict,
        mae_cfg: dict,
    ) -> None:
        super().__init__()

        self.mask_ratio    = mae_cfg.get("mask_ratio", 0.75)
        self.norm_pix_loss = mae_cfg.get("norm_pix_loss", True)

        patch_size = encoder_cfg.get("patch_size", 8)
        image_size = encoder_cfg.get("image_size", 96)
        self.patch_size = patch_size
        grid_h = grid_w = image_size // patch_size
        num_patches = grid_h * grid_w

        self.encoder: VisionTransformer = build_vit(encoder_cfg)

        self.decoder = MAEDecoder(
            num_patches       = num_patches,
            encoder_embed_dim = self.encoder.embed_dim,
            decoder_embed_dim = decoder_cfg.get("embed_dim", 512),
            decoder_depth     = decoder_cfg.get("depth", 8),
            decoder_num_heads = decoder_cfg.get("num_heads", 16),
            mlp_ratio         = decoder_cfg.get("mlp_ratio", 4.0),
            patch_size        = patch_size,
            grid_h            = grid_h,
            grid_w            = grid_w,
        )


    def _mask_and_encode(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches and encode visible tokens.

        Args:
            images: Float tensor ``(B, C, H, W)``.

        Returns:
            latent:      Encoder output ``(B, N_keep+1, D)`` (with [CLS]).
            mask:        Binary ``(B, N)`` — 1 = masked, 0 = kept.
            ids_restore: Permutation tensor ``(B, N)`` to restore order.
        """
        B = images.shape[0]
        N = self.encoder.num_patches

        # Sample masks per image (same ratio, different random positions)
        keep_ids_list, masked_ids_list = [], []
        for _ in range(B):
            k, m = create_mae_mask(N, self.mask_ratio)
            keep_ids_list.append(k)
            masked_ids_list.append(m)

        # Stack into batched tensors
        keep_ids    = torch.stack(keep_ids_list)      # (B, N_keep)
        masked_ids  = torch.stack(masked_ids_list)    # (B, N_mask)
        device      = images.device

        # Build ids_restore
        ids_shuffle = torch.cat([keep_ids, masked_ids], dim=1).to(device)  # (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Binary mask: 1 = masked
        mask = torch.ones(B, N, device=device)
        mask.scatter_(1, keep_ids.to(device), 0.0)

        # Tokenise and select visible patches
        x = self.encoder.patch_embed(images)           # (B, D, h, w)
        x = x.flatten(2).transpose(1, 2)               # (B, N, D)
        # Add positional embeddings (patch positions only)
        pos = self.encoder.pos_embed
        if self.encoder.use_cls_token:
            x = x + pos[:, 1:, :]
        else:
            x = x + pos

        # Keep only visible tokens
        keep_ids_exp = keep_ids.to(device).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x_vis = torch.gather(x, 1, keep_ids_exp)       # (B, N_keep, D)

        # Prepend [CLS] token
        cls_token = self.encoder.cls_token
        if cls_token is not None:
            cls = cls_token.expand(B, -1, -1)
            if self.encoder.use_cls_token:
                cls = cls + pos[:, :1, :]
            x_vis = torch.cat([cls, x_vis], dim=1)

        # Encode
        for blk in self.encoder.blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder.norm(x_vis)               # (B, N_keep+1, D)

        return x_vis, mask, ids_restore


    def _reconstruction_loss(
        self,
        images: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction loss over masked patches only.

        Args:
            images: Original images ``(B, C, H, W)``.
            pred:   Predicted pixel patches ``(B, N, p²C)``.
            mask:   Binary mask ``(B, N)`` — 1 = masked.

        Returns:
            Scalar mean reconstruction loss.
        """
        target = patchify(images, self.patch_size)  # (B, N, p²C)

        if self.norm_pix_loss:
            # Normalise each patch independently (variance normalisation)
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)         # (B, N)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MAE reconstruction loss.

        Args:
            images: Float tensor ``(B, C, H, W)``.

        Returns:
            Dictionary with keys:
              - ``"loss"``:        Scalar reconstruction loss.
              - ``"pred"``:        Reconstructed patch pixels ``(B, N, p²C)``.
              - ``"mask"``:        Binary mask ``(B, N)``.
        """
        latent, mask, ids_restore = self._mask_and_encode(images)
        pred  = self.decoder(latent, ids_restore)
        loss  = self._reconstruction_loss(images, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask}

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global features from the encoder (no masking at inference).

        Args:
            images: Float tensor ``(B, C, H, W)``.

        Returns:
            Feature tensor ``(B, D)``.
        """
        return self.encoder.forward_features(images)
