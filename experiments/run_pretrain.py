"""
Entry point for self-supervised pretraining.

Dispatches to the appropriate SSL model (JEPA / DINO / MAE) based on the
``experiment.method`` field in the YAML config.

Example usage::

    # I-JEPA pretraining on PatchCamelyon
    python experiments/run_pretrain.py --config configs/jepa.yaml

    # DINO pretraining with custom output dir
    python experiments/run_pretrain.py \\
        --config configs/dino.yaml \\
        --output_dir outputs/dino_pcam \\
        --resume outputs/dino_pcam/checkpoint_latest.pth

    # Multi-GPU (DDP) — 4 GPUs
    torchrun --nproc_per_node=4 experiments/run_pretrain.py \\
        --config configs/jepa.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.patchcamelyon import PatchCamelyon
from datasets.camelyon16 import CAMELYON16
from datasets.tcga import TCGADataset
from models.ssl.jepa import IJEPA, EMAMomentumScheduler
from models.ssl.dino import DINO, TeacherTempScheduler
from models.ssl.mae import MAE
from training.pretrain import pretrain
from utils.logger import Logger
from utils.seed import set_seed
from utils.transforms import build_ssl_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL pretraining for histopathology")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to the YAML config file.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override experiment output directory.")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--no_wandb",   action="store_true",
                        help="Disable Weights & Biases logging.")
    return parser.parse_args()


def build_dataset(cfg: dict, transform):
    """Instantiate the pretraining dataset from the config."""
    data_cfg = cfg["data"]
    name     = data_cfg["dataset"]
    root     = data_cfg["data_root"]

    if name == "patchcamelyon":
        return PatchCamelyon(root, split="train", transform=transform)
    elif name == "camelyon16":
        return CAMELYON16(root, split="train", transform=transform)
    elif name == "tcga":
        return TCGADataset(root, split="train", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


def build_model(cfg: dict):
    """Instantiate the SSL model and associated schedulers."""
    method = cfg["experiment"].get("method", "jepa")
    ema_cfg = cfg["model"].get("ema", {})

    if method == "jepa":
        model = IJEPA(
            encoder_cfg   = cfg["model"]["encoder"],
            predictor_cfg = cfg["model"]["predictor"],
            masking_cfg   = cfg["masking"],
            ema_momentum  = ema_cfg.get("momentum_start", 0.996),
        )
        return model, None

    elif method == "dino":
        model = DINO(
            encoder_cfg  = cfg["model"]["encoder"],
            head_cfg     = cfg["model"]["projection_head"],
            dino_cfg     = cfg["dino"],
            ema_momentum = ema_cfg.get("momentum_start", 0.996),
        )
        return model, None

    elif method == "mae":
        model = MAE(
            encoder_cfg = cfg["model"]["encoder"],
            decoder_cfg = cfg["model"]["decoder"],
            mae_cfg     = cfg["mae"],
        )
        return model, None

    else:
        raise ValueError(f"Unknown SSL method: {method!r}")

def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override output dir if requested
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir

    # Infer method from config filename if not set explicitly
    if "method" not in cfg["experiment"]:
        stem = Path(args.config).stem
        for m in ("jepa", "dino", "mae"):
            if m in stem:
                cfg["experiment"]["method"] = m
                break

    set_seed(cfg["experiment"].get("seed", 42))

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Logger
    logger = Logger(
        log_dir        = output_dir / "logs",
        name           = cfg["experiment"]["name"],
        use_tensorboard= True,
        use_wandb      = not args.no_wandb and cfg.get("use_wandb", False),
    )
    logger.info("Config loaded from %s", args.config)
    logger.info("Method: %s", cfg["experiment"].get("method", "jepa"))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Transforms
    method    = cfg["experiment"].get("method", "jepa")
    transform = build_ssl_transform(cfg, mode=method)

    # Dataset & DataLoader
    dataset = build_dataset(cfg, transform)
    logger.info("Dataset: %s", dataset)

    data_cfg = cfg["data"]
    loader   = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = data_cfg.get("num_workers", 8),
        pin_memory  = data_cfg.get("pin_memory", True),
        drop_last   = True,
    )

    # Model
    model, _ = build_model(cfg)
    logger.info(
        "Parameters: %.2fM total / %.2fM trainable",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )

    # Schedulers
    total_steps     = cfg["training"]["epochs"] * len(loader)
    ema_scheduler   = None
    temp_scheduler  = None

    if method in ("jepa",):
        ema_cfg = cfg["model"].get("ema", {})
        ema_scheduler = EMAMomentumScheduler(
            model,
            m_start    = ema_cfg.get("momentum_start", 0.996),
            m_end      = ema_cfg.get("momentum_end", 1.0),
            total_steps= total_steps,
        )

    if method == "dino":
        dino_cfg = cfg["dino"]
        ema_cfg  = cfg["model"].get("ema", {})
        ema_scheduler = type("_EMA", (), {
            "step": lambda self: model.update_teacher(
                ema_cfg.get("momentum_start", 0.996)
            )
        })()
        temp_scheduler = TeacherTempScheduler(
            temp_start      = dino_cfg.get("teacher_temp_start", 0.04),
            temp_end        = dino_cfg.get("teacher_temp_end", 0.07),
            warmup_epochs   = dino_cfg.get("teacher_temp_warmup_epochs", 30),
            total_epochs    = cfg["training"]["epochs"],
            steps_per_epoch = len(loader),
        )

    # Train
    pretrain(
        model                 = model,
        train_loader          = loader,
        cfg                   = cfg,
        logger                = logger,
        device                = device,
        resume_path           = args.resume,
        teacher_temp_scheduler= temp_scheduler,
        ema_scheduler         = ema_scheduler,
    )

    logger.close()


if __name__ == "__main__":
    main()
