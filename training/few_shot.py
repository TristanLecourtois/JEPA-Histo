"""
Few-shot learning evaluation for self-supervised histopathology representations.

Evaluates the quality of frozen encoder representations under label-scarce
conditions by training a linear probe with only a small fraction of the
available labels (1 %, 5 %, 10 %).  This directly models the realistic
clinical scenario where expert annotation is expensive and limited.

Protocol:
  1. Extract features from the frozen encoder for all splits.
  2. For each label fraction in ``{1%, 5%, 10%}``:
     a. Sample a stratified subset of training labels.
     b. Train a linear classifier on the subset features.
     c. Evaluate on the *full* test set (accuracy + AUROC).
  3. Report mean ± std over multiple random seeds for statistical validity.

This protocol follows:
  - Azizi et al., "Big Self-Supervised Models Advance Medical Image
    Classification", ICCV 2021.
  - Chen et al., "A Simple Framework for Contrastive Learning of Visual
    Representations", ICML 2020.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from evaluation.metrics import accuracy, auroc
from models.heads.linear_probe import LinearProbe
from training.linear_probe import extract_features, train_linear_head
from utils.logger import Logger


# ---------------------------------------------------------------------------
# Stratified subset sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    labels: torch.Tensor,
    fraction: float,
    seed: int = 0,
) -> torch.Tensor:
    """Return indices of a stratified random subset of size ``fraction * N``.

    Sampling is done per-class to maintain class balance.

    Args:
        labels:   Integer label tensor of shape ``(N,)``.
        fraction: Fraction in ``(0, 1]`` of samples to keep per class.
        seed:     Random seed.

    Returns:
        Index tensor of shape ``(M,)`` where ``M ≈ fraction * N``.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    classes = labels.unique().tolist()
    selected = []
    for c in classes:
        class_idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        k = max(1, int(len(class_idx) * fraction))
        chosen = rng.sample(class_idx, k)
        selected.extend(chosen)

    return torch.tensor(selected, dtype=torch.long)


# ---------------------------------------------------------------------------
# Single few-shot trial
# ---------------------------------------------------------------------------

def few_shot_trial(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    fraction: float,
    probe_cfg: dict,
    logger: Logger,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, float]:
    """Run one few-shot trial at a given label fraction.

    Args:
        train_features: Full training features ``(N_train, D)``.
        train_labels:   Full training labels ``(N_train,)``.
        test_features:  Test features ``(N_test, D)``.
        test_labels:    Test labels ``(N_test,)``.
        val_features:   Validation features ``(N_val, D)``.
        val_labels:     Validation labels ``(N_val,)``.
        fraction:       Label fraction to use (e.g. 0.01, 0.05, 0.10).
        probe_cfg:      Linear probe training config.
        logger:         Logger instance.
        device:         Compute device.
        seed:           Random seed for subset sampling.

    Returns:
        Dictionary with ``"acc"`` and ``"auroc"`` keys.
    """
    # Sample a stratified subset of training labels
    subset_idx    = stratified_sample(train_labels, fraction, seed=seed)
    sub_features  = train_features[subset_idx]
    sub_labels    = train_labels[subset_idx]

    num_classes = int(train_labels.max().item()) + 1
    in_dim      = train_features.shape[1]
    probe       = LinearProbe(in_dim, num_classes)

    train_linear_head(
        probe, sub_features, sub_labels,
        val_features, val_labels,
        probe_cfg, logger, device,
    )

    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_features.to(device)).cpu()

    acc   = accuracy(test_logits, test_labels, topk=1)
    auc   = auroc(test_logits, test_labels, num_classes=num_classes)
    return {"acc": acc, "auroc": auc}


# ---------------------------------------------------------------------------
# Full few-shot evaluation sweep
# ---------------------------------------------------------------------------

def run_few_shot_evaluation(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict,
    logger: Logger,
    device: torch.device,
    fractions: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Sweep over label fractions and seeds; report mean ± std.

    Args:
        encoder:      Pretrained encoder (will be frozen).
        train_loader: DataLoader for the full labelled training split.
        val_loader:   DataLoader for the validation split.
        test_loader:  DataLoader for the test split.
        cfg:          Full experiment config.
        logger:       Logger instance.
        device:       Compute device.
        fractions:    Label fractions to evaluate.  Defaults to
                      ``[0.01, 0.05, 0.10, 1.0]``.
        seeds:        Random seeds for repeated trials.  Defaults to
                      ``[0, 1, 2]``.

    Returns:
        Nested dict ``{fraction_key: {"acc_mean", "acc_std",
        "auroc_mean", "auroc_std"}}``.
    """
    fractions = fractions or [0.01, 0.05, 0.10, 1.0]
    seeds     = seeds     or [0, 1, 2]

    probe_cfg = cfg.get("few_shot", cfg.get("linear_probe", {
        "lr": 1e-3, "weight_decay": 0.0, "epochs": 100, "batch_size": 256
    }))

    # Freeze encoder and extract all features once
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    logger.info("Extracting features for few-shot evaluation …")
    train_feats, train_lbls = extract_features(encoder, train_loader, device)
    val_feats,   val_lbls   = extract_features(encoder, val_loader,   device)
    test_feats,  test_lbls  = extract_features(encoder, test_loader,  device)
    logger.info(
        "Done.  train=%d  val=%d  test=%d.",
        len(train_feats), len(val_feats), len(test_feats),
    )

    all_results: Dict[str, Dict[str, float]] = {}

    for frac in fractions:
        frac_key = f"{int(frac * 100)}pct"
        accs, aurocs = [], []

        for seed in seeds:
            logger.info(
                "Few-shot trial  fraction=%.2f%%  seed=%d …", frac * 100, seed
            )
            res = few_shot_trial(
                train_feats, train_lbls,
                test_feats,  test_lbls,
                val_feats,   val_lbls,
                fraction=frac,
                probe_cfg=probe_cfg,
                logger=logger,
                device=device,
                seed=seed,
            )
            accs.append(res["acc"])
            aurocs.append(res["auroc"])

        mean_acc   = float(np.mean(accs))
        std_acc    = float(np.std(accs))
        mean_auroc = float(np.mean(aurocs))
        std_auroc  = float(np.std(aurocs))

        all_results[frac_key] = {
            "acc_mean":   mean_acc,
            "acc_std":    std_acc,
            "auroc_mean": mean_auroc,
            "auroc_std":  std_auroc,
        }

        logger.info(
            "  [%s]  acc=%.2f±%.2f%%  auroc=%.4f±%.4f",
            frac_key,
            mean_acc * 100, std_acc * 100,
            mean_auroc, std_auroc,
        )
        logger.log_scalars(
            {
                f"fewshot/{frac_key}/acc_mean":   mean_acc,
                f"fewshot/{frac_key}/auroc_mean": mean_auroc,
            },
            step=0,
        )

    # Summary table
    logger.info("\n%s", _format_results_table(all_results))
    return all_results


def _format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format results as a Markdown table for logging."""
    header = "| Label fraction | Accuracy (%)         | AUROC                |"
    sep    = "|----------------|----------------------|----------------------|"
    rows   = [header, sep]
    for frac_key, v in results.items():
        rows.append(
            f"| {frac_key:<14} "
            f"| {v['acc_mean']*100:.2f} ± {v['acc_std']*100:.2f}       "
            f"| {v['auroc_mean']:.4f} ± {v['auroc_std']:.4f}     |"
        )
    return "\n".join(rows)
