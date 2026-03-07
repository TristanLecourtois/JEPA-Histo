"""
Data augmentation pipelines for histopathology SSL pretraining.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class StainJitter:
    """Perturb H&E stain concentrations in the optical-density space.

    Applies a random multiplicative noise to the stain matrix, simulating
    scanner variability and different staining protocols.  Operates entirely
    on PIL Images and returns a PIL Image.

    Args:
        sigma1: Std-dev of log-normal perturbation on the first stain channel.
        sigma2: Std-dev of log-normal perturbation on the second stain channel.
        prob:   Probability of applying this transform.
    """

    # Ruifrok & Johnston H&E reference matrix (column vectors)
    _HE_REF = np.array(
        [[0.5626, 0.2159],
         [0.7201, 0.8012],
         [0.4062, 0.5581]],
        dtype=np.float32,
    )

    def __init__(
        self,
        sigma1: float = 0.2,
        sigma2: float = 0.2,
        prob: float = 0.5,
    ) -> None:
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.prob:
            return img

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.clip(img_np, 1e-6, 1.0)

        # Convert to optical density
        OD = -np.log(img_np)

        # Project onto H&E basis (least-squares)
        HE = self._HE_REF
        HE_pinv = np.linalg.pinv(HE)
        concentrations = HE_pinv @ OD.reshape(-1, 3).T  # (2, N)

        # Random multiplicative perturbation
        alpha = np.random.lognormal(0.0, self.sigma1)
        beta  = np.random.lognormal(0.0, self.sigma2)
        concentrations[0] *= alpha
        concentrations[1] *= beta

        # Reconstruct RGB
        OD_new = HE @ concentrations
        img_new = np.exp(-OD_new).T.reshape(img_np.shape)
        img_new = np.clip(img_new * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_new)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sigma1={self.sigma1}, "
            f"sigma2={self.sigma2}, prob={self.prob})"
        )



# Multi-crop transform (DINO)

class MultiCropTransform:
    """Produce multiple crops of different scales from a single image.

    Returns a list:  [global_crop_1, global_crop_2, local_1, ..., local_N]

    Args:
        global_size:        Spatial resolution of global crops.
        global_scale:       Scale range for global random-resized crops.
        local_size:         Spatial resolution of local crops.
        local_scale:        Scale range for local random-resized crops.
        num_local_crops:    Number of local crops.
        base_transform:     Colour / stain augmentations applied to every crop.
        normalize:          Final normalisation transform.
        global_blur_probs:  Per-view Gaussian-blur probability for the two
                            global crops (default ``[1.0, 0.1]``).
        local_blur_prob:    Gaussian-blur probability for local crops.
    """

    def __init__(
        self,
        global_size: int = 96,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_size: int = 32,
        local_scale: Tuple[float, float] = (0.05, 0.4),
        num_local_crops: int = 8,
        base_transform: T.Compose | None = None,
        normalize: T.Normalize | None = None,
        global_blur_probs: Sequence[float] = (1.0, 0.1),
        local_blur_prob: float = 0.5,
    ) -> None:
        self.num_local_crops = num_local_crops

        flip_and_color = base_transform or T.Compose([])
        norm = normalize or T.Compose([])
        to_tensor = T.ToTensor()

        def _gaussian_blur(p: float) -> T.RandomApply:
            return T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=p)

        def _global(blur_p: float) -> T.Compose:
            return T.Compose([
                T.RandomResizedCrop(global_size, scale=global_scale),
                flip_and_color,
                _gaussian_blur(blur_p),
                to_tensor,
                norm,
            ])

        def _local() -> T.Compose:
            return T.Compose([
                T.RandomResizedCrop(local_size, scale=local_scale),
                flip_and_color,
                _gaussian_blur(local_blur_prob),
                to_tensor,
                norm,
            ])

        self.global_transforms = [_global(p) for p in global_blur_probs]
        self.local_transform = _local()

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        crops = [t(img) for t in self.global_transforms]
        crops += [self.local_transform(img) for _ in range(self.num_local_crops)]
        return crops


def build_ssl_transform(cfg: dict, mode: str = "jepa") -> T.Compose:
    """Build the augmentation pipeline from a config dictionary.

    Args:
        cfg:  ``augmentation`` sub-dictionary from the YAML config.
        mode: One of ``"jepa"`` | ``"mae"`` (single-view) or ``"dino"``
              (multi-crop via :class:`MultiCropTransform`).

    Returns:
        A ``torchvision.transforms.Compose`` object (or
        :class:`MultiCropTransform` for DINO).
    """
    aug = cfg.get("augmentation", cfg)
    mean = aug.get("normalize", {}).get("mean", [0.7008, 0.5384, 0.6916])
    std  = aug.get("normalize", {}).get("std",  [0.2350, 0.2774, 0.2128])
    normalize = T.Normalize(mean=mean, std=std)

    color_cfg = aug.get("color_jitter", {})
    color_jitter = T.ColorJitter(
        brightness=color_cfg.get("brightness", 0.4),
        contrast=color_cfg.get("contrast", 0.4),
        saturation=color_cfg.get("saturation", 0.2),
        hue=color_cfg.get("hue", 0.1),
    )

    stain_cfg = aug.get("stain_jitter", {})
    stain_jitter = StainJitter(
        sigma1=stain_cfg.get("sigma1", 0.2),
        sigma2=stain_cfg.get("sigma2", 0.2),
        prob=stain_cfg.get("prob", 0.5),
    ) if stain_cfg.get("enabled", True) else None

    base_augs: list = []
    if aug.get("horizontal_flip", True):
        base_augs.append(T.RandomHorizontalFlip())
    if aug.get("vertical_flip", True):
        base_augs.append(T.RandomVerticalFlip())
    if stain_jitter is not None:
        base_augs.append(stain_jitter)
    if color_cfg.get("enabled", True):
        base_augs.append(T.RandomApply([color_jitter], p=color_cfg.get("prob", 0.8)))
    if aug.get("grayscale", {}).get("prob", 0.2) > 0:
        base_augs.append(T.RandomGrayscale(p=aug["grayscale"]["prob"]))

    image_size = cfg.get("data", cfg).get("image_size", 96)
    crop_cfg   = aug.get("random_resized_crop", {})
    scale      = tuple(crop_cfg.get("scale", [0.4, 1.0]))
    ratio      = tuple(crop_cfg.get("ratio", [0.75, 1.33]))

    if mode == "dino":
        local_size = aug.get("local_crop_size", 32)
        return MultiCropTransform(
            global_size=image_size,
            global_scale=tuple(aug.get("global_crops_scale", [0.4, 1.0])),
            local_size=local_size,
            local_scale=tuple(aug.get("local_crops_scale", [0.05, 0.4])),
            num_local_crops=aug.get("num_local_crops", 8),
            base_transform=T.Compose(base_augs),
            normalize=normalize,
            global_blur_probs=aug.get("gaussian_blur", {}).get("global_prob", [1.0, 0.1]),
            local_blur_prob=aug.get("gaussian_blur", {}).get("local_prob", 0.5),
        )

    # Single-view pipeline (JEPA / MAE)
    pipeline = [
        T.RandomResizedCrop(image_size, scale=scale, ratio=ratio),
        *base_augs,
        T.ToTensor(),
        normalize,
    ]
    return T.Compose(pipeline)


def build_eval_transform(image_size: int, mean: List[float], std: List[float]) -> T.Compose:
    """Minimal deterministic transform for evaluation / feature extraction.

    Args:
        image_size: Target spatial resolution.
        mean:       Per-channel normalisation mean.
        std:        Per-channel normalisation std.

    Returns:
        A ``torchvision.transforms.Compose`` object.
    """
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
