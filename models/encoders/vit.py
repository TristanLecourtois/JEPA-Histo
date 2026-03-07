

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    """Multi-head self-attention with optional qkv-bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer encoder block (Pre-LN)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1    = norm_layer(dim)
        self.attn     = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2    = norm_layer(dim)
        self.mlp      = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 2-D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be even).
        grid_h:    Number of patch rows.
        grid_w:    Number of patch columns.
        cls_token: Prepend a zero vector for the [CLS] token.

    Returns:
        Tensor of shape ``(grid_h*grid_w [+1], embed_dim)``.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([grid_y.ravel(), grid_x.ravel()], dim=0)  # (2, N)

    half = embed_dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)

    def _embed(pos: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("n,d->nd", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    emb = torch.cat([_embed(grid[0]), _embed(grid[1])], dim=-1)  # (N, embed_dim)

    if cls_token:
        emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0)
    return emb


class VisionTransformer(nn.Module):
    """Vision Transformer encoder.

    Args:
        image_size:      Spatial resolution of the input image (height = width).
        patch_size:      Side length of each square patch in pixels.
        in_channels:     Number of input image channels.
        embed_dim:       Token embedding dimension.
        depth:           Number of Transformer blocks.
        num_heads:       Number of attention heads.
        mlp_ratio:       Ratio of MLP hidden dim to embedding dim.
        qkv_bias:        Use bias in QKV projections.
        drop_rate:       Dropout rate for patch projections and MLP.
        attn_drop_rate:  Dropout rate for attention weights.
        drop_path_rate:  Maximum stochastic-depth drop probability (linearly
                         scaled across blocks).
        use_cls_token:   Prepend a learnable [CLS] token (required for DINO;
                         disabled for I-JEPA).
        norm_layer:      Normalisation layer constructor.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."

        self.embed_dim    = embed_dim
        self.patch_size   = patch_size
        self.image_size   = image_size
        self.use_cls_token = use_cls_token

        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        self.num_patches  = num_patches_h * num_patches_w
        self.grid_h       = num_patches_h
        self.grid_w       = num_patches_w

        # Patch embedding (tokenisation)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # [CLS] token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Positional embedding (fixed sinusoidal)
        pos_emb = get_2d_sincos_pos_embed(
            embed_dim, num_patches_h, num_patches_w, cls_token=use_cls_token
        )
        self.register_buffer("pos_embed", pos_emb.unsqueeze(0))  # (1, N[+1], D)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / truncated-normal weight initialisation following the ViT paper."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode an image (or a masked subset of its patches).

        Args:
            x:    Float tensor of shape ``(B, C, H, W)``.
            mask: Optional boolean tensor of shape ``(B, N)`` or ``(N,)``
                  where ``True`` marks tokens to keep.  If ``None``, all
                  tokens are processed.

        Returns:
            patch_tokens: Shape ``(B, N_kept, D)`` — patch-level embeddings.
            cls_token:    Shape ``(B, D)`` if ``use_cls_token`` else ``None``.
        """
        B = x.shape[0]

        # Tokenise
        x = self.patch_embed(x)          # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)

        # Add positional embedding to patch tokens (exclude CLS slot if present)
        pos = self.pos_embed
        if self.use_cls_token:
            x = x + pos[:, 1:, :]
        else:
            x = x + pos

        # Token masking (context selection for I-JEPA)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            x = x[mask].reshape(B, -1, x.shape[-1])  # keep selected tokens

        # Prepend [CLS] token
        cls_out = None
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            if self.use_cls_token:
                cls = cls + pos[:, :1, :]
            x = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.use_cls_token:
            cls_out      = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            patch_tokens = x

        return patch_tokens, cls_out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the global representation used for linear probing.

        For DINO (use_cls_token=True), returns the [CLS] token.
        For I-JEPA / MAE (use_cls_token=False), returns the mean of patch tokens.

        Args:
            x: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, D)``.
        """
        patch_tokens, cls_token = self.forward(x)
        if cls_token is not None:
            return cls_token
        return patch_tokens.mean(dim=1)

    @property
    def output_dim(self) -> int:
        return self.embed_dim


def vit_tiny(**kwargs) -> VisionTransformer:
    """ViT-Ti/8: embed_dim=192, depth=12, heads=3."""
    kwargs.setdefault("embed_dim", 192)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 3)
    return VisionTransformer(**kwargs)


def vit_small(**kwargs) -> VisionTransformer:
    """ViT-S/8: embed_dim=384, depth=12, heads=6."""
    kwargs.setdefault("embed_dim", 384)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 6)
    return VisionTransformer(**kwargs)


def vit_base(**kwargs) -> VisionTransformer:
    """ViT-B/8: embed_dim=768, depth=12, heads=12."""
    kwargs.setdefault("embed_dim", 768)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    return VisionTransformer(**kwargs)


def vit_large(**kwargs) -> VisionTransformer:
    """ViT-L/16: embed_dim=1024, depth=24, heads=16."""
    kwargs.setdefault("embed_dim", 1024)
    kwargs.setdefault("depth", 24)
    kwargs.setdefault("num_heads", 16)
    return VisionTransformer(**kwargs)


VIT_REGISTRY = {
    "vit_tiny":  vit_tiny,
    "vit_small": vit_small,
    "vit_base":  vit_base,
    "vit_large": vit_large,
}


def build_vit(cfg: dict) -> VisionTransformer:
    """Instantiate a ViT encoder from a config dictionary.

    Args:
        cfg: ``model.encoder`` sub-dict from the YAML config.

    Returns:
        Initialised :class:`VisionTransformer`.

    Raises:
        KeyError: If the requested architecture is not in :data:`VIT_REGISTRY`.
    """
    arch = cfg.get("arch", "vit_small")
    if arch not in VIT_REGISTRY:
        raise KeyError(
            f"Unknown ViT architecture: {arch!r}.  "
            f"Available: {list(VIT_REGISTRY)}"
        )
    constructor = VIT_REGISTRY[arch]
    return constructor(
        image_size      = cfg.get("image_size", 96),
        patch_size      = cfg.get("patch_size", 8),
        embed_dim       = cfg.get("embed_dim", 384),
        depth           = cfg.get("depth", 12),
        num_heads       = cfg.get("num_heads", 6),
        mlp_ratio       = cfg.get("mlp_ratio", 4.0),
        drop_path_rate  = cfg.get("drop_path_rate", 0.1),
        use_cls_token   = cfg.get("use_cls_token", False),
    )
