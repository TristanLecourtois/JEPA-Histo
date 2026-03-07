"""
Evaluation metrics for histopathology classification.

Implements:
  - Top-k accuracy (standard classification metric).
  - AUROC (area under the ROC curve) — the primary metric in medical imaging
    papers as it is threshold-independent and robust to class imbalance.
  - Average precision (area under the precision–recall curve).
  - Balanced accuracy for imbalanced test sets.
  - Confusion-matrix utilities.
  - Calibration error (ECE) for model reliability assessment.

All functions operate on raw logits / probabilities and integer labels and
are fully differentiable-free (no gradient tracking) for use in eval loops.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int = 1,
) -> float:
    """Compute top-k classification accuracy.

    Args:
        logits: Raw logit tensor of shape ``(N, C)``.
        labels: Integer label tensor of shape ``(N,)``.
        topk:   k for top-k accuracy.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    with torch.no_grad():
        _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
        return correct[:topk].any(dim=0).float().mean().item()


def balanced_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute balanced accuracy (mean per-class recall).

    Args:
        logits: Raw logit tensor of shape ``(N, C)``.
        labels: Integer label tensor of shape ``(N,)``.

    Returns:
        Balanced accuracy as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1).numpy()
    lbls  = labels.numpy()
    classes = np.unique(lbls)
    recalls = []
    for c in classes:
        mask = lbls == c
        recalls.append((preds[mask] == c).mean())
    return float(np.mean(recalls))



def auroc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-averaged AUROC.

    For binary classification (num_classes == 2), returns standard AUROC.
    For multi-class, uses the one-vs-rest macro average.

    Args:
        logits:      Raw logit tensor of shape ``(N, C)``.
        labels:      Integer label tensor of shape ``(N,)``.
        num_classes: Number of classes C.

    Returns:
        Macro-averaged AUROC as a float in ``[0, 1]``.
    """
    probs = F.softmax(logits, dim=-1).numpy()
    lbls  = labels.numpy()

    if num_classes == 2:
        return _binary_auroc(probs[:, 1], lbls)

    # One-vs-rest macro average
    aucs = []
    for c in range(num_classes):
        y_bin  = (lbls == c).astype(int)
        y_score = probs[:, c]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue  # skip degenerate classes
        aucs.append(_binary_auroc(y_score, y_bin))
    return float(np.mean(aucs)) if aucs else float("nan")


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUROC via the trapezoidal rule (no sklearn dependency)."""
    order = np.argsort(-scores)
    lbls_sorted = labels[order]
    npos = lbls_sorted.sum()
    nneg = len(lbls_sorted) - npos
    if npos == 0 or nneg == 0:
        return float("nan")

    tp = np.cumsum(lbls_sorted)
    fp = np.cumsum(1 - lbls_sorted)
    tpr = tp / npos
    fpr = fp / nneg

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    auc = float(np.trapz(tpr, fpr))
    return auc


def average_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-averaged average precision (AUPRC).

    Args:
        logits:      Raw logit tensor of shape ``(N, C)``.
        labels:      Integer label tensor of shape ``(N,)``.
        num_classes: Number of classes.

    Returns:
        Macro-averaged AP as a float in ``[0, 1]``.
    """
    probs = F.softmax(logits, dim=-1).numpy()
    lbls  = labels.numpy()
    aps   = []

    for c in range(num_classes):
        y_bin   = (lbls == c).astype(int)
        y_score = probs[:, c]
        if y_bin.sum() == 0:
            continue
        aps.append(_average_precision_binary(y_score, y_bin))

    return float(np.mean(aps)) if aps else float("nan")


def _average_precision_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AP via the step-function integration (sklearn-compatible)."""
    order    = np.argsort(-scores)
    labels_s = labels[order]
    tp       = np.cumsum(labels_s)
    prec     = tp / (np.arange(len(labels_s)) + 1)
    rec      = tp / labels_s.sum()
    # Step integral: sum of precision * Δrecall
    drecall = np.diff(rec, prepend=0)
    return float((prec * drecall).sum())


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the average gap between
    confidence and accuracy, weighted by bin size.

    Args:
        logits:  Raw logit tensor of shape ``(N, C)``.
        labels:  Integer label tensor of shape ``(N,)``.
        n_bins:  Number of confidence bins.

    Returns:
        ECE as a float in ``[0, 1]`` (lower is better).
    """
    probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels).float()

    confs   = confs.numpy()
    correct = correct.numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(labels))


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        logits:      Raw logit tensor of shape ``(N, C)``.
        labels:      Integer label tensor of shape ``(N,)``.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` where
        ``cm[i, j]`` is the number of samples with true class ``i`` predicted
        as class ``j``.
    """
    preds = logits.argmax(dim=1).numpy()
    lbls  = labels.numpy()
    cm    = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(lbls, preds):
        cm[t, p] += 1
    return cm


def compute_all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict:
    """Compute all classification metrics at once.

    Args:
        logits:      Raw logit tensor ``(N, C)``.
        labels:      Integer label tensor ``(N,)``.
        num_classes: Number of classes.

    Returns:
        Dictionary with keys: ``"acc"``, ``"balanced_acc"``, ``"auroc"``,
        ``"ap"``, ``"ece"``.
    """
    return {
        "acc":          accuracy(logits, labels, topk=1),
        "balanced_acc": balanced_accuracy(logits, labels),
        "auroc":        auroc(logits, labels, num_classes),
        "ap":           average_precision(logits, labels, num_classes),
        "ece":          expected_calibration_error(logits, labels),
    }
