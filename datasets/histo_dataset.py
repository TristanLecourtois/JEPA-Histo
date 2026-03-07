from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


Sample = Tuple[str, int]   # (image_path, class_index)


class HistoDataset(Dataset, abc.ABC):
    """Abstract base for histopathology patch datasets.

    Subclasses must implement :meth:`_load_samples`, which should populate
    ``self.samples`` with ``(path, label)`` pairs and set
    ``self.class_to_idx``.

    Args:
        root:         Root data directory.
        split:        One of ``"train"`` | ``"val"`` | ``"test"``.
        transform:    Transform applied to each PIL image.
        target_transform: Transform applied to each integer label.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root             = Path(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.samples: List[Sample]           = []
        self.class_to_idx: Dict[str, int]    = {}
        self._load_samples()

    @abc.abstractmethod
    def _load_samples(self) -> None:
        """Populate ``self.samples`` and ``self.class_to_idx``.

        Must be implemented by every concrete subclass.
        """

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def get_class_indices(self) -> Dict[int, List[int]]:
        """Return a mapping from class index to sample indices.

        Used by few-shot samplers to draw balanced support sets.

        Returns:
            Dict mapping ``class_idx -> [sample_idx, ...]``.
        """
        mapping: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(self.samples):
            mapping.setdefault(label, []).append(idx)
        return mapping

    def sample_few_shot(
        self,
        n_per_class: int,
        seed: int = 0,
    ) -> "HistoDataset":
        """Return a new dataset with at most ``n_per_class`` samples per class.

        Args:
            n_per_class: Number of labelled examples per class to keep.
            seed:        Random seed for reproducibility.

        Returns:
            A shallow copy of this dataset with a reduced ``samples`` list.
        """
        import copy
        import random

        rng = random.Random(seed)
        class_indices = self.get_class_indices()
        selected: List[Sample] = []
        for label, indices in sorted(class_indices.items()):
            chosen = rng.sample(indices, min(n_per_class, len(indices)))
            selected.extend(self.samples[i] for i in chosen)

        new_ds        = copy.copy(self)
        new_ds.samples = selected
        return new_ds

    def label_fraction_subset(
        self,
        fraction: float,
        seed: int = 0,
    ) -> "HistoDataset":
        """Return a new dataset with ``fraction`` of labelled samples per class.

        Args:
            fraction: Float in ``(0, 1]``.
            seed:     Random seed.

        Returns:
            A shallow copy with a reduced ``samples`` list.
        """
        import copy
        import random

        rng = random.Random(seed)
        class_indices = self.get_class_indices()
        selected: List[Sample] = []
        for label, indices in sorted(class_indices.items()):
            k = max(1, int(len(indices) * fraction))
            chosen = rng.sample(indices, k)
            selected.extend(self.samples[i] for i in chosen)

        new_ds        = copy.copy(self)
        new_ds.samples = selected
        return new_ds


    @property
    def num_classes(self) -> int:
        """Number of distinct classes in this split."""
        return len(self.class_to_idx)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_idx.items()}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={self.root}, split={self.split}, "
            f"n_samples={len(self)}, n_classes={self.num_classes})"
        )
