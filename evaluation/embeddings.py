"""
Embedding visualisation and nearest-neighbour evaluation utilities.

Provides:
  - t-SNE / UMAP dimensionality reduction for qualitative inspection of the
    learned representation space.
  - k-NN accuracy (weighted voting) as a non-parametric evaluation proxy that
    requires no additional training (Zhai et al., 2019).
  - Embedding-space quality metrics (intra/inter-class distances).

These analyses help diagnose whether the encoder learns semantically
meaningful representations before committing to the linear probe protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.logger import Logger


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_dataset(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and optionally L2-normalise embeddings for an entire dataset.

    Args:
        encoder:   Frozen encoder implementing ``forward_features(x)``.
        loader:    DataLoader yielding ``(images, labels)`` batches.
        device:    Compute device.
        normalise: If True, L2-normalise embeddings (required for cosine kNN).

    Returns:
        embeddings: Float array of shape ``(N, D)``.
        labels:     Integer array of shape ``(N,)``.
    """
    encoder.eval()
    all_feats, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats  = encoder.forward_features(images)       # (B, D)
        if normalise:
            feats = F.normalize(feats, dim=-1, p=2)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# k-NN classifier
# ---------------------------------------------------------------------------

def knn_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Weighted k-NN accuracy in cosine-similarity space.

    Computes softmax-weighted votes from the k nearest training neighbours
    for each test sample (following He et al., 2020).

    Args:
        train_embeddings: Shape ``(N_train, D)`` (L2-normalised).
        train_labels:     Shape ``(N_train,)``.
        test_embeddings:  Shape ``(N_test, D)`` (L2-normalised).
        test_labels:      Shape ``(N_test,)``.
        k:                Number of nearest neighbours.
        temperature:      Softmax temperature for vote weighting.

    Returns:
        kNN accuracy as a float in ``[0, 1]``.
    """
    # Compute cosine similarity matrix
    sim = test_embeddings @ train_embeddings.T  # (N_test, N_train)

    num_classes   = int(train_labels.max()) + 1
    top_k_sim     = np.partition(-sim, k, axis=1)[:, :k]   # negated
    top_k_idx     = np.argpartition(-sim, k, axis=1)[:, :k]
    top_k_sim     = -top_k_sim  # restore signs

    # Softmax-weighted voting
    weights = np.exp(top_k_sim / temperature)                # (N_test, k)
    weights = weights / weights.sum(axis=1, keepdims=True)

    top_k_labels = train_labels[top_k_idx]                  # (N_test, k)
    vote_matrix  = np.zeros((len(test_labels), num_classes))
    for i in range(len(test_labels)):
        for j in range(k):
            vote_matrix[i, top_k_labels[i, j]] += weights[i, j]

    predicted = vote_matrix.argmax(axis=1)
    return float((predicted == test_labels).mean())


# ---------------------------------------------------------------------------
# Dimensionality reduction (t-SNE / UMAP)
# ---------------------------------------------------------------------------

def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2-D using t-SNE.

    Args:
        embeddings:   Float array ``(N, D)``.
        n_components: Target dimensionality.
        perplexity:   t-SNE perplexity parameter.
        n_iter:       Number of optimisation iterations.
        seed:         Random seed.

    Returns:
        Reduced embedding array of shape ``(N, n_components)``.

    Raises:
        ImportError: If ``scikit-learn`` is not installed.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn required for t-SNE.  pip install scikit-learn") from exc

    # Pre-reduce with PCA to speed up t-SNE
    if embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA
        embeddings = PCA(n_components=50, random_state=seed).fit_transform(embeddings)

    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
    ).fit_transform(embeddings)


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2-D using UMAP.

    Args:
        embeddings:   Float array ``(N, D)``.
        n_components: Target dimensionality.
        n_neighbors:  UMAP local neighbourhood size.
        min_dist:     Minimum distance between embedded points.
        seed:         Random seed.

    Returns:
        Reduced embedding array of shape ``(N, n_components)``.

    Raises:
        ImportError: If ``umap-learn`` is not installed.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn required.  pip install umap-learn") from exc

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Embedding quality metrics
# ---------------------------------------------------------------------------

def class_separation(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute intra- and inter-class cosine distances.

    The ratio ``inter / intra`` (higher is better) measures whether the
    representation space groups same-class samples together.

    Args:
        embeddings: L2-normalised float array ``(N, D)``.
        labels:     Integer array ``(N,)``.

    Returns:
        Dictionary with keys ``"intra_dist"``, ``"inter_dist"``,
        ``"separation_ratio"``.
    """
    classes   = np.unique(labels)
    centroids = np.array([embeddings[labels == c].mean(0) for c in classes])

    # Intra-class: average distance to centroid within each class
    intra = []
    for c, centroid in zip(classes, centroids):
        class_embs = embeddings[labels == c]
        dists = 1 - class_embs @ centroid  # cosine distance
        intra.append(dists.mean())
    intra_dist = float(np.mean(intra))

    # Inter-class: average pairwise centroid distance
    inter = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            inter.append(1 - centroids[i] @ centroids[j])
    inter_dist = float(np.mean(inter)) if inter else float("nan")

    ratio = inter_dist / (intra_dist + 1e-8)
    return {
        "intra_dist":       intra_dist,
        "inter_dist":       inter_dist,
        "separation_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Visualisation helper (saves a matplotlib figure)
# ---------------------------------------------------------------------------

def plot_embeddings(
    reduced: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding visualisation",
    save_path: Optional[str | Path] = None,
) -> None:
    """Scatter-plot 2-D reduced embeddings coloured by class.

    Args:
        reduced:      Float array ``(N, 2)``.
        labels:       Integer array ``(N,)``.
        class_names:  Optional list mapping class index → name.
        title:        Plot title.
        save_path:    If provided, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting.  pip install matplotlib")

    classes  = np.unique(labels)
    cmap     = plt.cm.get_cmap("tab10", len(classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in classes:
        mask  = labels == c
        label = class_names[c] if class_names and c < len(class_names) else str(c)
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=8, alpha=0.7, color=cmap(c), label=label,
        )
    ax.legend(loc="best", markerscale=3, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
