"""
CAMELYON16 patch dataset.

CAMELYON16 consists of 400 Whole-Slide Images (WSIs) of sentinel lymph node
sections, annotated for metastatic breast cancer.  This module provides a
patch-level dataset derived from pre-extracted tiles.

Pre-processing expected (run once with ``scripts/preprocess_camelyon16.py``):
  1. Extract tissue patches (256×256 at 20×) from each WSI using the tissue
     mask or the provided XML annotations.
  2. Save patches as JPEG/PNG files under ``<root>/<split>/<label>/``.

Expected directory layout::

    <root>/
        train/
            normal/   *.jpg
            tumor/    *.jpg
        val/
            normal/   *.jpg
            tumor/    *.jpg
        test/
            normal/   *.jpg
            tumor/    *.jpg

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets.histo_dataset import HistoDataset, Sample

logger = logging.getLogger(__name__)


class CAMELYON16(HistoDataset):
    """Patch-level CAMELYON16 dataset loaded from a folder hierarchy.

    Args:
        root:             Root data directory (see module docstring for layout).
        split:            Dataset split: ``"train"`` | ``"val"`` | ``"test"``.
        transform:        Transform applied to each PIL image.
        target_transform: Transform applied to each label.
        patch_size:       Expected patch spatial resolution (used for logging).
    """

    CLASS_NAMES = ["normal", "tumor"]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        patch_size: int = 256,
    ) -> None:
        self.patch_size = patch_size
        super().__init__(root, split, transform, target_transform)

    def _load_samples(self) -> None:
        """Scan the folder hierarchy and collect ``(path, label)`` pairs."""
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASS_NAMES)}
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"CAMELYON16 split directory not found: {split_dir}\n"
                f"Expected layout: <root>/<split>/<class>/*.jpg"
            )

        self.samples: List[Sample] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s – skipping.", class_dir)
                continue
            paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            self.samples.extend((str(p), class_idx) for p in paths)

        logger.info(
            "CAMELYON16 %s: %d patches across %d classes.",
            self.split, len(self.samples), self.num_classes,
        )

    def __repr__(self) -> str:
        return (
            f"CAMELYON16(split={self.split}, n_samples={len(self)}, "
            f"patch_size={self.patch_size}, root={self.root})"
        )
