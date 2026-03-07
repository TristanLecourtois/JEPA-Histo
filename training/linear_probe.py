from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from evaluation.metrics import accuracy, auroc
from models.heads.linear_probe import LinearProbe
from utils.logger import Logger


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache encoder features for the entire dataset.

    Args:
        encoder: Frozen encoder.  Must implement ``forward_features(x)``.
        loader:  DataLoader for the dataset to embed.
        device:  Compute device.

    Returns:
        features: Float tensor of shape ``(N, D)``.
        labels:   Long tensor of shape ``(N,)``.
    """
    encoder.eval()
    all_feats, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats  = encoder.forward_features(images)
        all_feats.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats), torch.cat(all_labels)



def train_linear_head(
    probe: LinearProbe,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    cfg: dict,
    logger: Logger,
    device: torch.device,
) -> float:
    """Train the linear classifier on pre-extracted features.

    Args:
        probe:          :class:`~models.heads.linear_probe.LinearProbe`.
        train_features: Shape ``(N_train, D)``.
        train_labels:   Shape ``(N_train,)``.
        val_features:   Shape ``(N_val, D)``.
        val_labels:     Shape ``(N_val,)``.
        cfg:            ``linear_probe`` config sub-dict.
        logger:         Logger instance.
        device:         Compute device.

    Returns:
        Best validation accuracy achieved.
    """
    lr      = cfg.get("lr", 1e-3)
    wd      = cfg.get("weight_decay", 0.0)
    epochs  = cfg.get("epochs", 100)
    batch_s = cfg.get("batch_size", 256)

    probe.to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
    train_ldr = DataLoader(train_ds, batch_size=batch_s, shuffle=True, drop_last=False)

    best_val_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0

        for feats, labels in train_ldr:
            feats, labels = feats.to(device), labels.to(device)
            logits = probe(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features.to(device))
        val_acc = accuracy(val_logits.cpu(), val_labels, topk=1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            logger.info(
                "  Linear probe epoch [%d/%d]  loss=%.4f  val_acc=%.2f%%",
                epoch + 1, epochs, epoch_loss / len(train_ldr), val_acc * 100,
            )
            logger.log_scalars(
                {"probe/val_acc": val_acc, "probe/loss": epoch_loss},
                step=epoch,
            )

    logger.info("Best val accuracy: %.2f%%", best_val_acc * 100)
    return best_val_acc


def run_linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict,
    logger: Logger,
    device: torch.device,
    label_fraction: Optional[float] = None,
) -> dict:
    """End-to-end linear probe evaluation.

    1. Freeze the encoder.
    2. Extract features for train / val / test sets.
    3. Train a linear classifier on the training features.
    4. Evaluate on the test set (accuracy + AUROC).

    Args:
        encoder:        Pretrained encoder.
        train_loader:   DataLoader for labelled training data.
        val_loader:     DataLoader for validation data.
        test_loader:    DataLoader for test data.
        cfg:            Full experiment config.
        logger:         Logger instance.
        device:         Compute device.
        label_fraction: If provided, restricts training labels to this
                        fraction (for the few-shot protocol).

    Returns:
        Dictionary with keys ``"acc"`` and ``"auroc"``.
    """
    probe_cfg = cfg.get("linear_probe", {
        "lr": 1e-3, "weight_decay": 0.0, "epochs": 100, "batch_size": 256
    })

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    logger.info("Extracting features …")
    train_feats, train_lbls = extract_features(encoder, train_loader, device)
    val_feats,   val_lbls   = extract_features(encoder, val_loader,   device)
    test_feats,  test_lbls  = extract_features(encoder, test_loader,  device)
    logger.info(
        "Features extracted  train=%d  val=%d  test=%d.",
        len(train_feats), len(val_feats), len(test_feats),
    )

    num_classes = int(train_lbls.max().item()) + 1
    in_dim      = train_feats.shape[1]
    probe       = LinearProbe(in_dim, num_classes)

    train_linear_head(
        probe, train_feats, train_lbls,
        val_feats, val_lbls,
        probe_cfg, logger, device,
    )

    # Test evaluation
    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_feats.to(device)).cpu()

    test_acc   = accuracy(test_logits, test_lbls, topk=1)
    test_auroc = auroc(test_logits, test_lbls, num_classes=num_classes)

    results = {"acc": test_acc, "auroc": test_auroc}

    tag = f"frac={label_fraction:.2f}" if label_fraction is not None else "full"
    logger.info(
        "Linear probe [%s]  test_acc=%.2f%%  test_auroc=%.4f",
        tag, test_acc * 100, test_auroc,
    )
    logger.log_scalars(
        {f"test/{tag}/acc": test_acc, f"test/{tag}/auroc": test_auroc},
        step=0,
    )
    return results
