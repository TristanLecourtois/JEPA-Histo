"""
Self-supervised pretraining loop.

Supports three SSL methods dispatched via ``cfg.experiment.method``:
  - ``"jepa"`` → :class:`~models.ssl.jepa.IJEPA`
  - ``"dino"`` → :class:`~models.ssl.dino.DINO`
  - ``"mae"``  → :class:`~models.ssl.mae.MAE`

Training features:
  - Mixed precision (AMP) via ``torch.cuda.amp``.
  - Cosine learning-rate schedule with linear warm-up.
  - Gradient clipping.
  - Periodic checkpoint saving with resumption support.
  - EMA momentum cosine scheduling (JEPA / DINO).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.logger import Logger



def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """Build AdamW optimiser, separating weight-decayed and non-decayed params.

    Following best practices (Loshchilov & Hutter, 2019), biases and
    LayerNorm parameters are excluded from weight decay.

    Args:
        model: The SSL model.
        cfg:   ``optimizer`` config sub-dict.

    Returns:
        Configured :class:`torch.optim.AdamW` instance.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay,    "weight_decay": cfg.get("weight_decay", 0.05)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        param_groups,
        lr=cfg.get("base_lr", 1.5e-4),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
    )


def cosine_lr_schedule(
    optimizer: AdamW,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Apply cosine LR decay with linear warm-up.

    Args:
        optimizer:    The optimiser whose LR is modified in-place.
        step:         Current global step (0-indexed).
        total_steps:  Total number of training steps.
        warmup_steps: Number of linear warm-up steps.
        base_lr:      Peak learning rate.
        min_lr:       Minimum learning rate at the end of cosine decay.

    Returns:
        Current learning rate value.
    """
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr



def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    cfg: dict,
    best: bool = False,
) -> None:
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "cfg":       cfg,
    }
    fname = "checkpoint_best.pth" if best else f"checkpoint_ep{epoch:04d}.pth"
    torch.save(state, output_dir / fname)
    # Always keep a 'latest' pointer for easy resumption
    torch.save(state, output_dir / "checkpoint_latest.pth")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scaler: Optional[GradScaler] = None,
) -> int:
    """Load a checkpoint and return the next epoch to resume from.

    Args:
        path:      Path to the checkpoint file.
        model:     The model whose state dict will be restored.
        optimizer: Optional optimiser to restore.
        scaler:    Optional AMP scaler to restore.

    Returns:
        Next epoch index (``checkpoint_epoch + 1``).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0) + 1


def _jepa_step(model, batch, device) -> torch.Tensor:
    images, _ = batch
    images = images.to(device, non_blocking=True)
    out = model(images)
    return out["loss"]


def _dino_step(model, batch, device, teacher_temp_scheduler) -> torch.Tensor:
    crops, _ = batch
    crops = [c.to(device, non_blocking=True) for c in crops]
    teacher_temp = teacher_temp_scheduler.step()
    out = model(crops, teacher_temp=teacher_temp)
    return out["loss"]


def _mae_step(model, batch, device) -> torch.Tensor:
    images, _ = batch
    images = images.to(device, non_blocking=True)
    out = model(images)
    return out["loss"]



def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict,
    logger: Logger,
    device: torch.device,
    resume_path: Optional[str] = None,
    teacher_temp_scheduler=None,
    ema_scheduler=None,
) -> None:
    """Run the full self-supervised pretraining loop.

    Args:
        model:                  The SSL model (JEPA / DINO / MAE).
        train_loader:           DataLoader for the unlabelled pretraining set.
        cfg:                    Full experiment config dictionary.
        logger:                 :class:`~utils.logger.Logger` instance.
        device:                 Compute device.
        resume_path:            Optional path to a checkpoint for warm-start.
        teacher_temp_scheduler: Required for DINO; ignored otherwise.
        ema_scheduler:          EMA scheduler for JEPA / DINO.
    """
    output_dir  = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    method      = cfg["experiment"].get("method", "jepa")
    opt_cfg     = cfg["optimizer"]
    sched_cfg   = cfg["scheduler"]
    train_cfg   = cfg["training"]

    epochs      = train_cfg["epochs"]
    batch_size  = train_cfg["batch_size"]
    fp16        = train_cfg.get("fp16", True)
    clip_grad   = opt_cfg.get("clip_grad", 1.0)
    log_every   = cfg["experiment"].get("log_every", 50)
    save_every  = cfg["experiment"].get("save_every", 10)

    steps_per_epoch = len(train_loader)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = sched_cfg.get("warmup_epochs", 15) * steps_per_epoch
    base_lr         = opt_cfg["base_lr"]
    min_lr          = sched_cfg.get("min_lr", 1e-6)

    optimizer = build_optimizer(model, opt_cfg)
    scaler    = GradScaler(enabled=fp16)

    start_epoch = 0
    if resume_path is not None:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scaler)
        logger.info("Resumed from %s (epoch %d).", resume_path, start_epoch)

    model.to(device)
    model.train()

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            lr = cosine_lr_schedule(
                optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=fp16):
                if method == "jepa":
                    loss = _jepa_step(model, batch, device)
                elif method == "dino":
                    loss = _dino_step(model, batch, device, teacher_temp_scheduler)
                elif method == "mae":
                    loss = _mae_step(model, batch, device)
                else:
                    raise ValueError(f"Unknown SSL method: {method!r}")

            scaler.scale(loss).backward()

            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if ema_scheduler is not None:
                ema_scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if step % log_every == 0:
                logger.log_scalars(
                    {"train/loss": loss.item(), "train/lr": lr},
                    step=global_step,
                )
                logger.info(
                    "Epoch [%d/%d] step [%d/%d]  loss=%.4f  lr=%.2e",
                    epoch + 1, epochs, step + 1, steps_per_epoch,
                    loss.item(), lr,
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / steps_per_epoch
        logger.info(
            "── Epoch %d done  avg_loss=%.4f  time=%.1fs", epoch + 1, avg_loss, elapsed
        )
        logger.log_scalar("train/epoch_loss", avg_loss, step=epoch)

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(output_dir, epoch, model, optimizer, scaler, cfg)
            logger.info("Checkpoint saved at epoch %d.", epoch + 1)

    logger.info("Pretraining complete.")
