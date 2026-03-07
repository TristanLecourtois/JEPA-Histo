# JEPA-Histo: Joint-Embedding Predictive Architecture for Histopathology Representation Learning

> **Self-Supervised Pretraining of Pathology Encoders via Latent Prediction — with Application to Few-Shot Tissue Classification**

---

## Abstract

Acquiring expert-annotated histopathology slides is expensive and time-consuming, making label-efficient learning a critical challenge in computational pathology.
We investigate **Image JEPA (I-JEPA)** — a self-supervised method that learns representations by predicting missing regions of an image *entirely in latent space*, without pixel-level reconstruction — as a pretraining strategy for H&E-stained whole-slide image analysis.
Unlike contrastive methods (DINO) that require view augmentation invariance, or generative methods (MAE) that encourage low-level texture encoding, I-JEPA's latent prediction objective naturally biases the encoder towards *semantic* patch representations.
We pretrain ViT-S/8 encoders on PatchCamelyon and evaluate on linear probing and few-shot classification at 1%, 5%, and 10% label fractions, reporting accuracy and AUROC.


---

## Motivation

Whole-slide image (WSI) analysis at scale requires representations that are:

1. **Semantically meaningful** — capturing tissue architecture rather than stain artefacts.
2. **Label-efficient** — performing well with only a handful of annotated cases.
3. **Domain-robust** — transferring across scanners, staining protocols, and cancer types.

Self-supervised pretraining on large unlabelled patch collections addresses (1) and (2) by learning from image structure alone.
The key question is *which pretraining objective* best satisfies all three desiderata for histopathology.

**I-JEPA** predicts the latent representation of masked image regions given an unmasked context, using a momentum-updated target encoder to provide stable prediction targets.
Because the prediction task is posed entirely in representation space, the model is not incentivised to encode pixel-level textures — an important property for stain-variable histopathology.

---

## Method

**Masking strategy.**
Four non-overlapping rectangular target blocks are sampled from the token grid (each covering 15–20% of patches, aspect ratio 0.75–1.5).
The context is a large block (85–100% of patches) from which the target positions are removed.
This structured masking forces the predictor to reason about spatial context over multiple semantic regions simultaneously.

**Target encoder.**
The target encoder is an exponential moving average (EMA) of the context encoder, with momentum annealed from 0.996 to 1.0 following a cosine schedule.
Using EMA targets prevents representation collapse without requiring negative pairs.

**Predictor.**
A *narrow* Transformer (half the hidden width of the encoder) takes context tokens as keys/values and target positional embeddings as queries.
The bottleneck architecture prevents the predictor from trivially copying context features and forces it to compose semantic information.

### Baselines

| Method | Objective | Representation | Key paper |
|--------|-----------|----------------|-----------|
| **I-JEPA** | Latent patch prediction | Patch tokens (mean-pooled) | Assran et al., CVPR 2023 |
| DINO | Self-distillation (CLS token) | [CLS] token | Caron et al., ICCV 2021 |
| MAE | Pixel reconstruction | [CLS] token (mean-pool) | He et al., CVPR 2022 |

---

## Repository Structure

```
JEPA-Histo/
├── configs/
│   ├── jepa.yaml              # I-JEPA hyperparameters
│   ├── dino.yaml              # DINO hyperparameters
│   └── mae.yaml               # MAE hyperparameters
│
├── datasets/
│   ├── histo_dataset.py       # Abstract base class
│   ├── patchcamelyon.py       # PCam HDF5 reader
│   ├── camelyon16.py          # CAMELYON16 folder reader
│   └── tcga.py                # TCGA multi-cancer dataset
│
├── models/
│   ├── encoders/
│   │   ├── vit.py             # Vision Transformer (ViT-Ti/S/B/L)
│   │   └── resnet.py          # ResNet baseline encoder
│   ├── ssl/
│   │   ├── jepa.py            # I-JEPA (context enc + EMA target + predictor)
│   │   ├── dino.py            # DINO (student/teacher + centering)
│   │   └── mae.py             # MAE (encoder + pixel-space decoder)
│   └── heads/
│       ├── predictor.py       # JEPA latent predictor (narrow ViT)
│       ├── projection_mlp.py  # DINO head, MAE decoder, generic MLP
│       └── linear_probe.py    # Linear / MLP evaluation heads
│
├── training/
│   ├── pretrain.py            # Main SSL training loop (AMP, cosine LR, ckpts)
│   ├── linear_probe.py        # Feature extraction + linear classifier training
│   └── few_shot.py            # Stratified label-fraction sweep
│
├── evaluation/
│   ├── metrics.py             # Accuracy, AUROC, AUPRC, ECE, confusion matrix
│   └── embeddings.py          # kNN probe, t-SNE/UMAP, class separation
│
├── utils/
│   ├── transforms.py          # H&E augmentations, StainJitter, MultiCropTransform
│   ├── patching.py            # Block masking (I-JEPA), MAE mask, WSI extraction
│   ├── seed.py                # Reproducibility
│   └── logger.py              # TensorBoard + W&B unified logger
│
├── experiments/
│   ├── run_pretrain.py        # CLI entry point for pretraining
│   ├── run_linear_probe.py    # CLI entry point for linear probe
│   └── run_fewshot.py         # CLI entry point for few-shot sweep
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/tristanlecourtois/JEPA-Histo.git
cd JEPA-Histo

# 2. Create a virtual environment (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install openslide for WSI preprocessing
#    macOS:   brew install openslide
#    Ubuntu:  sudo apt-get install openslide-tools
pip install openslide-python
```

---

## Datasets

### PatchCamelyon (PCam)

Binary patch classification (tumour vs. normal) from Camelyon16 WSIs.
96×96 pixels at 10× magnification.  262,144 training / 32,768 validation / 32,768 test patches.

```bash
# Download from the official repository
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5
# ... (see https://github.com/basveeling/pcam for all files)
mkdir -p data/patchcamelyon && mv *.h5 data/patchcamelyon/
```

### CAMELYON16

Sentinel lymph node WSIs with pixel-level tumour annotations.
Pre-extract 256×256 patches at 20× magnification using the tissue / annotation masks:

```bash
python scripts/preprocess_camelyon16.py \
    --wsi_dir /path/to/camelyon16/wsis \
    --output_dir data/camelyon16
```

### TCGA

```bash
# Download slides via GDC Data Transfer Tool, then extract patches:
python scripts/preprocess_tcga.py \
    --manifest gdc_manifest.txt \
    --output_dir data/tcga
```

Set the `data.data_root` field in the config or pass `--data_root` at runtime.

**Dataset normalisation statistics** (pre-computed on PCam training set, used for all datasets):

| Dataset       | Mean (R, G, B)              | Std (R, G, B)               |
|---------------|-----------------------------|-----------------------------|
| PatchCamelyon | 0.7008, 0.5384, 0.6916      | 0.2350, 0.2774, 0.2128      |

---

## Experiments

All experiments are fully reproducible via the YAML configurations and fixed random seeds.

### 1. Self-Supervised Pretraining

```bash
# I-JEPA — single GPU
python experiments/run_pretrain.py --config configs/jepa.yaml

# I-JEPA — multi-GPU (4× A100)
torchrun --nproc_per_node=4 experiments/run_pretrain.py \
    --config configs/jepa.yaml \
    --output_dir outputs/jepa

# DINO
python experiments/run_pretrain.py --config configs/dino.yaml

# MAE
python experiments/run_pretrain.py --config configs/mae.yaml

# Resume from checkpoint
python experiments/run_pretrain.py \
    --config configs/jepa.yaml \
    --resume outputs/jepa/checkpoint_latest.pth
```

Training artefacts are written to `outputs/<method>/`:
- `checkpoint_ep<epoch>.pth` — periodic snapshots
- `checkpoint_latest.pth`    — always updated (for resumption)
- `checkpoint_best.pth`      — best validation checkpoint
- `logs/experiment.log`      — text log
- `logs/tb/`                 — TensorBoard events

### 2. Linear Probe Evaluation

```bash
python experiments/run_linear_probe.py \
    --config configs/jepa.yaml \
    --checkpoint outputs/jepa/checkpoint_best.pth
```

Expected output:
```
Linear probe [full]  test_acc=88.41%  test_auroc=0.9523
```

### 3. Few-Shot Learning

```bash
# Default: 1%, 5%, 10%, 100% over 3 seeds
python experiments/run_fewshot.py \
    --config configs/jepa.yaml \
    --checkpoint outputs/jepa/checkpoint_best.pth \
    --fractions 0.01 0.05 0.10 1.0 \
    --seeds 0 1 2
```

Results are saved to `outputs/<method>/fewshot/fewshot_results.json`.

---

## Configuration Reference

All hyperparameters are controlled by YAML config files.  Key fields:

```yaml
experiment:
  name:       jepa_histo_pretrain
  method:     jepa                  # jepa | dino | mae
  output_dir: outputs/jepa
  seed:       42

data:
  dataset:    patchcamelyon         # patchcamelyon | camelyon16 | tcga
  data_root:  data/patchcamelyon
  image_size: 96

model:
  encoder:
    arch:          vit_small         # vit_tiny | vit_small | vit_base | vit_large
    patch_size:    8
    embed_dim:     384
    depth:         12
    num_heads:     6
    use_cls_token: false             # false for JEPA/MAE, true for DINO

optimizer:
  base_lr:       1.5e-4             # peak LR (cosine schedule)
  weight_decay:  0.05

training:
  epochs:     300
  batch_size: 512
  fp16:       true
```

See `configs/jepa.yaml`, `configs/dino.yaml`, `configs/mae.yaml` for full documentation.

---


## References

```
[1] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
    LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture. CVPR 2023.

[2] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P.,
    & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers.
    ICCV 2021.

[3] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022).
    Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.

[4] Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers
    for Image Recognition at Scale. ICLR 2021.

[5] Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018).
    Rotation Equivariant CNNs for Digital Pathology. MICCAI 2018.

[6] Tellez, D., Litjens, G., Bándi, P., Bulten, W., Bokhorst, J. M., Ciompi, F.,
    & van der Laak, J. (2019). Quantifying the effects of data augmentation and
    stain color normalization in convolutional neural networks for computational
    pathology. Medical Image Analysis, 58, 101544.

[7] Azizi, S., et al. (2021). Big Self-Supervised Models Advance Medical Image
    Classification. ICCV 2021.

[8] Litjens, G., et al. (2018). 1399 H&E-stained sentinel lymph node sections of
    breast cancer patients: the CAMELYON dataset. GigaScience, 7(6).
```

---

## License

This project is released under the **MIT License**.  See `LICENSE` for details.
