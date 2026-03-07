"""
Patch extraction and masking utilities for Whole-Slide Images (WSIs) and
tile-based datasets.

This module provides two levels of functionality:

1. **WSI-level**: Sliding-window patch extraction from OpenSlide-compatible
   WSI files (used by CAMELYON16 and TCGA preprocessing scripts).
2. **Token-level**: Differentiable masking helpers used inside I-JEPA and MAE
   to select context and target sets from a ViT token grid.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch



def extract_patches_from_wsi(
    slide_path: str,
    patch_size: int = 256,
    level: int = 0,
    overlap: int = 0,
    tissue_threshold: float = 0.5,
    otsu_downsample: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Extract non-overlapping tissue patches from a WSI.

    Uses Otsu thresholding on a downsampled thumbnail to identify tissue
    regions and discard background (glass / fat) patches.

    Args:
        slide_path:         Path to the WSI file (e.g. ``.svs``, ``.tif``).
        patch_size:         Patch side length in pixels at ``level``.
        level:              OpenSlide pyramid level to read at.
        overlap:            Pixel overlap between adjacent patches.
        tissue_threshold:   Minimum fraction of tissue pixels per patch.
        otsu_downsample:    Downsampling factor for the tissue mask.

    Returns:
        patches:    List of ``(patch_size, patch_size, 3)`` uint8 arrays.
        coords:     List of ``(x, y)`` top-left coordinates at ``level=0``.

    Raises:
        ImportError: If ``openslide-python`` is not installed.
    """
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "openslide-python is required for WSI processing.  "
            "Install via: pip install openslide-python"
        ) from exc

    import cv2

    slide  = openslide.OpenSlide(slide_path)
    w, h   = slide.level_dimensions[level]
    stride = patch_size - overlap

    # Build a low-resolution tissue mask using Otsu thresholding
    thumb_w = max(1, w // otsu_downsample)
    thumb_h = max(1, h // otsu_downsample)
    thumb   = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    patches, coords = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Check tissue fraction via the low-resolution mask
            mx = int(x / otsu_downsample)
            my = int(y / otsu_downsample)
            mpw = max(1, int(patch_size / otsu_downsample))
            mph = max(1, int(patch_size / otsu_downsample))
            roi = mask[my: my + mph, mx: mx + mpw]
            if roi.mean() / 255.0 < tissue_threshold:
                continue

            # Convert coordinates to level-0 space for OpenSlide
            ds  = slide.level_downsamples[level]
            x0  = int(x * ds)
            y0  = int(y * ds)
            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            patch  = np.array(region.convert("RGB"))
            patches.append(patch)
            coords.append((x0, y0))

    slide.close()
    return patches, coords


# Token-level masking (I-JEPA / MAE)

def sample_block_mask(
    num_patches_h: int,
    num_patches_w: int,
    scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
    num_blocks: int = 1,
    max_attempts: int = 100,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample a set of rectangular block masks on the token grid.

    Samples ``num_blocks`` non-overlapping rectangular masks.  Each block has
    an area uniformly drawn from ``scale`` (as a fraction of the total grid)
    and an aspect ratio drawn from ``aspect_ratio``.

    Args:
        num_patches_h:  Number of patch rows.
        num_patches_w:  Number of patch columns.
        scale:          ``(min, max)`` fraction of total tokens per block.
        aspect_ratio:   ``(min, max)`` height/width aspect ratio per block.
        num_blocks:     Number of non-overlapping blocks to sample.
        max_attempts:   Maximum random trials before giving up on overlap
                        avoidance.
        exclude:        Boolean mask of shape ``(H, W)`` indicating already-
                        occupied positions that must not be selected.

    Returns:
        Boolean mask of shape ``(num_patches_h, num_patches_w)`` where
        ``True`` marks selected (masked) positions.
    """
    H, W     = num_patches_h, num_patches_w
    total    = H * W
    occupied = np.zeros((H, W), dtype=bool)
    if exclude is not None:
        occupied |= exclude

    for _ in range(num_blocks):
        placed = False
        for _attempt in range(max_attempts):
            area = int(total * np.random.uniform(*scale))
            ar   = np.random.uniform(*aspect_ratio)
            bh   = max(1, int(round(math.sqrt(area * ar))))
            bw   = max(1, int(round(math.sqrt(area / ar))))
            bh   = min(bh, H)
            bw   = min(bw, W)
            r    = np.random.randint(0, H - bh + 1)
            c    = np.random.randint(0, W - bw + 1)
            if not occupied[r: r + bh, c: c + bw].any():
                occupied[r: r + bh, c: c + bw] = True
                placed = True
                break

        if not placed:
            # Fall back: place anywhere (may overlap with previous blocks)
            bh = max(1, int(round(math.sqrt(total * np.random.uniform(*scale)))))
            bw = max(1, int(round(math.sqrt(total * np.random.uniform(*scale)))))
            bh = min(bh, H)
            bw = min(bw, W)
            r  = np.random.randint(0, H - bh + 1)
            c  = np.random.randint(0, W - bw + 1)
            occupied[r: r + bh, c: c + bw] = True

    return occupied


def create_jepa_masks(
    num_patches_h: int,
    num_patches_w: int,
    context_scale: Tuple[float, float] = (0.85, 1.0),
    context_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    target_scale: Tuple[float, float] = (0.15, 0.2),
    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    num_target_blocks: int = 4,
    allow_overlap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate I-JEPA context and target token masks.

    Context tokens are obtained by sampling a large block and complementing
    it with the target regions (which are *removed* from the context) unless
    ``allow_overlap`` is True.

    Args:
        num_patches_h:        Number of patch rows.
        num_patches_w:        Number of patch columns.
        context_scale:        Fraction of tokens kept as context.
        context_aspect_ratio: Aspect-ratio range for the context block.
        target_scale:         Fraction of tokens per target block.
        target_aspect_ratio:  Aspect-ratio range per target block.
        num_target_blocks:    Number of target blocks to predict.
        allow_overlap:        Whether context and target may overlap.

    Returns:
        context_mask: Boolean tensor of shape ``(H*W,)``; True = keep.
        target_mask:  Boolean tensor of shape ``(H*W,)``; True = predict.
    """
    H, W = num_patches_h, num_patches_w

    # Sample target regions first
    target_grid = sample_block_mask(
        H, W, scale=target_scale, aspect_ratio=target_aspect_ratio,
        num_blocks=num_target_blocks,
    )

    # Sample context region (large), then remove target positions
    context_grid = sample_block_mask(
        H, W, scale=context_scale, aspect_ratio=context_aspect_ratio,
        num_blocks=1,
    )
    if not allow_overlap:
        context_grid &= ~target_grid

    context_mask = torch.from_numpy(context_grid.ravel())
    target_mask  = torch.from_numpy(target_grid.ravel())
    return context_mask, target_mask


def create_mae_mask(
    num_patches: int,
    mask_ratio: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random MAE token mask (uniform random, not block-structured).

    Args:
        num_patches: Total number of patch tokens (H*W or sequence length).
        mask_ratio:  Fraction of tokens to mask.

    Returns:
        keep_ids:   Indices of unmasked tokens (sorted), shape ``(N_keep,)``.
        masked_ids: Indices of masked tokens, shape ``(N_mask,)``.
    """
    num_masked = int(num_patches * mask_ratio)
    noise      = torch.rand(num_patches)
    ids_sorted = torch.argsort(noise)
    masked_ids = ids_sorted[:num_masked]
    keep_ids   = ids_sorted[num_masked:]
    return keep_ids.sort().values, masked_ids.sort().values


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Rearrange image tensor into non-overlapping patches.

    Args:
        imgs:       Float tensor of shape ``(B, C, H, W)``.
        patch_size: Side length of each square patch in pixels.

    Returns:
        Tensor of shape ``(B, N, patch_size**2 * C)`` where
        ``N = (H/p) * (W/p)``.
    """
    B, C, H, W = imgs.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, \
        f"Image size ({H}x{W}) must be divisible by patch_size ({p})."

    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)        # (B, h, w, p, p, C)
    x = x.reshape(B, h * w, p * p * C)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
    """Reconstruct image tensors from patch representations.

    Args:
        patches:    Float tensor of shape ``(B, N, patch_size**2 * C)``.
        patch_size: Side length of each square patch.
        img_size:   Spatial resolution of the reconstructed image.

    Returns:
        Tensor of shape ``(B, C, img_size, img_size)``.
    """
    p = patch_size
    h = w = img_size // p
    C = patches.shape[-1] // (p * p)
    B = patches.shape[0]

    x = patches.reshape(B, h, w, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4)        # (B, C, h, p, w, p)
    x = x.reshape(B, C, h * p, w * p)
    return x
