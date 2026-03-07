from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """Unified experiment logger with optional TensorBoard / W&B backends.

    Args:
        log_dir: Directory where log files and TensorBoard events are stored.
        name: Root logger name (defaults to the experiment name).
        use_tensorboard: Attach a TensorBoard SummaryWriter.
        use_wandb: Attach a Weights & Biases run.
        wandb_config: Configuration dict forwarded to ``wandb.init``.
    """

    def __init__(
        self,
        log_dir: str | Path,
        name: str = "jepa_histo",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.log_dir / "experiment.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
                self.info("TensorBoard writer initialised.")
            except ImportError:
                self.warning("tensorboard not installed – skipping.")

        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(**(wandb_config or {}))
                self.info("W&B run initialised.")
            except ImportError:
                self.warning("wandb not installed – skipping.")


    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)


    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric to all active backends.

        Args:
            tag: Metric name (e.g. ``"train/loss"``).
            value: Scalar value.
            step: Global training step or epoch.
        """
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, global_step=step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once.

        Args:
            metrics: Dictionary mapping tag → value.
            step: Global training step or epoch.
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_image(self, tag: str, image_tensor: Any, step: int) -> None:
        """Log an image tensor (C, H, W) or (N, C, H, W) to TensorBoard."""
        if self._tb_writer is not None:
            self._tb_writer.add_images(tag, image_tensor, global_step=step)

    def close(self) -> None:
        """Flush and close all writer handles."""
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for use inside a module.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
