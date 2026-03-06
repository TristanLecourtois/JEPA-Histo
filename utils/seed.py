"""
Reproducibility utilities.

Centralises all random-state initialisation so that every experiment
can be reproduced exactly from its configuration seed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, force cuDNN deterministic mode (may reduce
            throughput on some architectures).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN auto-tuner for faster but non-deterministic training
        torch.backends.cudnn.benchmark = True
