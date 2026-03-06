"""
TCGA (The Cancer Genome Atlas) patch dataset.

The TCGA dataset contains digitised H&E-stained WSIs for 33 cancer types.
This module loads pre-extracted patches stored in a flat folder-per-class
hierarchy and supports both coarse cancer-type classification and fine-grained
tissue-type subtyping.

Expected directory layout::

    <root>/
        train/
            LUAD/  *.jpg        # Lung Adenocarcinoma
            LUSC/  *.jpg        # Lung Squamous Cell Carcinoma
            BRCA/  *.jpg        # Breast Invasive Carcinoma
            ...
        val/
            ...
        test/
            ...

Reference:
  The Cancer Genome Atlas Research Network, "The Cancer Genome Atlas
  Pan-Cancer Analysis Project", Nature Genetics 2013.
  https://portal.gdc.cancer.gov/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from datasets.histo_dataset import HistoDataset, Sample

logger = logging.getLogger(__name__)

# Canonical 32 TCGA cancer-type abbreviations (excluding FFPE cohorts)
TCGA_CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD",
    "DLBC", "ESCA", "GBM",  "HNSC", "KICH", "KIRC",
    "KIRP", "LAML", "LGG",  "LIHC", "LUAD", "LUSC",
    "MESO", "OV",   "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM",
    "UCEC", "UCS",
]


class TCGADataset(HistoDataset):
    """Multi-cancer TCGA patch dataset.

    Scans each class sub-directory present under ``<root>/<split>/`` and
    builds a class index from the discovered folder names.  This allows the
    dataset to be used with an arbitrary subset of cancer types simply by
    populating the corresponding directories.

    Args:
        root:             Root data directory.
        split:            Dataset split: ``"train"`` | ``"val"`` | ``"test"``.
        transform:        Transform applied to each PIL image.
        target_transform: Transform applied to each label.
        cancer_types:     Restrict the dataset to a specific subset of cancer
                          types.  If ``None``, all discovered classes are used.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cancer_types: Optional[List[str]] = None,
    ) -> None:
        self._allowed_types = (
            set(cancer_types) if cancer_types is not None else None
        )
        super().__init__(root, split, transform, target_transform)

    def _load_samples(self) -> None:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"TCGA split directory not found: {split_dir}"
            )

        # Discover class directories
        class_dirs = sorted(
            d for d in split_dir.iterdir()
            if d.is_dir() and (
                self._allowed_types is None or d.name in self._allowed_types
            )
        )

        if not class_dirs:
            raise RuntimeError(
                f"No class directories found under {split_dir}.  "
                "Ensure patches are organised as <root>/<split>/<cancer_type>/*.jpg."
            )

        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.samples: List[Sample] = []

        for d in class_dirs:
            class_idx = self.class_to_idx[d.name]
            paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            self.samples.extend((str(p), class_idx) for p in paths)
            logger.debug("  %s: %d patches", d.name, len(paths))

        logger.info(
            "TCGA %s: %d patches across %d cancer types.",
            self.split, len(self.samples), self.num_classes,
        )

    def __repr__(self) -> str:
        classes = list(self.class_to_idx.keys())
        return (
            f"TCGADataset(split={self.split}, n_samples={len(self)}, "
            f"cancer_types={classes}, root={self.root})"
        )
