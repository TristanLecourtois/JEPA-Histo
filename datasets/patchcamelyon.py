"""
PatchCamelyon (PCam) dataset.

PCam is a binary patch-classification benchmark derived from the Camelyon16
whole-slide image challenge.  Each 96×96 pixel patch is labelled as positive
(contains tumour tissue in the central 32×32 region) or negative.

Dataset statistics:
  - Train:  262,144 patches
  - Val:     32,768 patches
  - Test:    32,768 patches
  - Classes: 0 = normal, 1 = tumour

Reference:
  Veeling et al., "Rotation Equivariant CNNs for Digital Pathology",
  MICCAI 2018.  https://github.com/basveeling/pcam

Expected directory layout::

    <root>/
        camelyonpatch_level_2_split_train_x.h5
        camelyonpatch_level_2_split_train_y.h5
        camelyonpatch_level_2_split_valid_x.h5
        camelyonpatch_level_2_split_valid_y.h5
        camelyonpatch_level_2_split_test_x.h5
        camelyonpatch_level_2_split_test_y.h5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Maps split name → prefix used in the actual filenames
_SPLIT_MAP = {
    "train": "train",
    "val":   "valid",
    "test":  "test",
}


class PatchCamelyon(Dataset):
    """PatchCamelyon benchmark dataset (HDF5 format).

    Reads patches lazily from the HDF5 archives so that only one
    batch of patches is loaded into memory at a time.

    Args:
        root:             Directory containing the PCam HDF5 files.
        split:            Dataset split: ``"train"`` | ``"val"`` | ``"test"``.
        transform:        Transform applied to each PIL image.
        target_transform: Transform applied to each label tensor.

    Raises:
        FileNotFoundError: If the expected HDF5 files are not found under
            ``root``.
        ImportError: If ``h5py`` is not installed.
    """

    CLASS_NAMES = ["normal", "tumour"]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for PatchCamelyon.  "
                "Install via: pip install h5py"
            ) from exc

        self.root             = Path(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.class_to_idx     = {c: i for i, c in enumerate(self.CLASS_NAMES)}

        if split not in _SPLIT_MAP:
            raise ValueError(f"split must be one of {list(_SPLIT_MAP)}; got {split!r}")

        prefix = _SPLIT_MAP[split]
        x_path = self.root / f"camelyonpatch_level_2_split_{prefix}_x.h5"
        y_path = self.root / f"camelyonpatch_level_2_split_{prefix}_y.h5"

        for p in (x_path, y_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"PCam file not found: {p}\n"
                    f"Download from: https://github.com/basveeling/pcam"
                )

        # Open file handles (kept open for lazy reading)
        self._h5x = h5py.File(x_path, "r")
        self._h5y = h5py.File(y_path, "r")
        self._x   = self._h5x["x"]  # (N, 96, 96, 3) uint8
        self._y   = self._h5y["y"]  # (N, 1, 1, 1)  uint8

        logger.info(
            "Loaded PatchCamelyon %s split: %d samples.", split, len(self._x)
        )

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        patch = self._x[index]          # (96, 96, 3) uint8
        label = int(self._y[index].flat[0])

        img = Image.fromarray(patch.astype(np.uint8), mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __del__(self) -> None:
        if hasattr(self, "_h5x"):
            self._h5x.close()
        if hasattr(self, "_h5y"):
            self._h5y.close()

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    def get_class_indices(self) -> dict:
        mapping: dict = {0: [], 1: []}
        labels = np.array(self._y).ravel()
        for idx, lbl in enumerate(labels):
            mapping[int(lbl)].append(idx)
        return mapping

    def label_fraction_subset(
        self,
        fraction: float,
        seed: int = 0,
    ) -> "_PCamSubset":
        """Return a :class:`_PCamSubset` with ``fraction`` of samples per class."""
        import random

        rng = random.Random(seed)
        class_indices = self.get_class_indices()
        selected = []
        for lbl, indices in sorted(class_indices.items()):
            k = max(1, int(len(indices) * fraction))
            chosen = rng.sample(indices, k)
            selected.extend((i, lbl) for i in chosen)

        return _PCamSubset(self, selected)

    def __repr__(self) -> str:
        return (
            f"PatchCamelyon(split={self.split}, n_samples={len(self)}, "
            f"root={self.root})"
        )


class _PCamSubset(Dataset):
    """Lightweight subset wrapper used by :meth:`PatchCamelyon.label_fraction_subset`."""

    def __init__(self, parent: PatchCamelyon, index_label_pairs: list) -> None:
        self._parent  = parent
        self._pairs   = index_label_pairs
        self.class_to_idx = parent.class_to_idx
        self.num_classes  = parent.num_classes

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int):
        src_idx, label = self._pairs[index]
        patch = self._parent._x[src_idx]
        img   = Image.fromarray(patch.astype(np.uint8), mode="RGB")
        if self._parent.transform is not None:
            img = self._parent.transform(img)
        return img, label
