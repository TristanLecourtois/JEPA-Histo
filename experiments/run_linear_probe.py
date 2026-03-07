"""
Entry point for linear probe evaluation of a pretrained SSL encoder.

Loads a pretrained encoder checkpoint, freezes it, and trains a linear
classifier on top.  Reports test accuracy and AUROC.

Example usage::

    python experiments/run_linear_probe.py \\
        --config configs/jepa.yaml \\
        --checkpoint outputs/jepa/checkpoint_best.pth \\
        --method jepa

    # Evaluate a DINO checkpoint on CAMELYON16
    python experiments/run_linear_probe.py \\
        --config configs/dino.yaml \\
        --checkpoint outputs/dino/checkpoint_latest.pth \\
        --dataset camelyon16 \\
        --data_root data/camelyon16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.patchcamelyon import PatchCamelyon
from datasets.camelyon16 import CAMELYON16
from datasets.tcga import TCGADataset
from training.linear_probe import run_linear_probe
from utils.logger import Logger
from utils.seed import set_seed
from utils.transforms import build_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe evaluation")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the SSL pretrained checkpoint.")
    parser.add_argument("--method",     type=str, default=None,
                        choices=["jepa", "dino", "mae"],
                        help="SSL method (inferred from config if not set).")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Override dataset name.")
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Override data root.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    return parser.parse_args()


def load_encoder(cfg: dict, checkpoint_path: str, method: str):
    """Load the context / student encoder from a pretrained checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"]

    if method == "jepa":
        from models.ssl.jepa import IJEPA
        model = IJEPA(
            encoder_cfg   = cfg["model"]["encoder"],
            predictor_cfg = cfg["model"]["predictor"],
            masking_cfg   = cfg.get("masking", {}),
        )
        model.load_state_dict(state)
        encoder = model.context_encoder

    elif method == "dino":
        from models.ssl.dino import DINO
        model = DINO(
            encoder_cfg  = cfg["model"]["encoder"],
            head_cfg     = cfg["model"]["projection_head"],
            dino_cfg     = cfg.get("dino", {}),
        )
        model.load_state_dict(state)
        encoder = model.teacher_encoder

    elif method == "mae":
        from models.ssl.mae import MAE
        model = MAE(
            encoder_cfg = cfg["model"]["encoder"],
            decoder_cfg = cfg["model"]["decoder"],
            mae_cfg     = cfg.get("mae", {}),
        )
        model.load_state_dict(state)
        encoder = model.encoder

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return encoder


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = args.method or cfg["experiment"].get("method", "jepa")
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir

    set_seed(cfg["experiment"].get("seed", 42))

    output_dir = Path(cfg["experiment"]["output_dir"]) / "linear_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir         = output_dir / "logs",
        name            = f"{cfg['experiment']['name']}_lp",
        use_tensorboard = True,
        use_wandb       = not args.no_wandb and cfg.get("use_wandb", False),
    )
    logger.info("Linear probe: method=%s  checkpoint=%s", method, args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build evaluation transform
    aug_cfg   = cfg.get("augmentation", {})
    norm_cfg  = aug_cfg.get("normalize", {})
    mean      = norm_cfg.get("mean", [0.7008, 0.5384, 0.6916])
    std       = norm_cfg.get("std",  [0.2350, 0.2774, 0.2128])
    image_size= cfg["data"]["image_size"]
    transform = build_eval_transform(image_size, mean, std)

    # Datasets
    data_cfg  = cfg["data"]
    root      = data_cfg["data_root"]
    ds_name   = data_cfg["dataset"]

    def _make_ds(split):
        if ds_name == "patchcamelyon":
            return PatchCamelyon(root, split=split, transform=transform)
        elif ds_name == "camelyon16":
            return CAMELYON16(root, split=split, transform=transform)
        elif ds_name == "tcga":
            return TCGADataset(root, split=split, transform=transform)
        raise ValueError(ds_name)

    train_ds = _make_ds("train")
    val_ds   = _make_ds("val")
    test_ds  = _make_ds("test")

    nw = data_cfg.get("num_workers", 8)
    pm = data_cfg.get("pin_memory", True)
    bs = cfg.get("linear_probe", {}).get("batch_size", 256)

    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    val_ldr   = torch.utils.data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    test_ldr  = torch.utils.data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # Load encoder
    encoder = load_encoder(cfg, args.checkpoint, method)
    logger.info("Encoder loaded from %s", args.checkpoint)

    # Run linear probe
    results = run_linear_probe(
        encoder, train_ldr, val_ldr, test_ldr, cfg, logger, device
    )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)
    logger.close()


if __name__ == "__main__":
    main()
