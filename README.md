# Fundus Image Classification (ResNet-18 / ResNet-34)

PyTorch pipeline for **stratified splitting**, **offline augmentation**, and **transfer learning** on a multi-class fundus dataset. The training script uses `torchvision` ResNet with optional asymmetric green-channel preprocessing, MixUp, AMP, and progressive unfreezing.

> **This repository** ships **code and docs only**. Raw images, large CSVs, and `.pth` checkpoints are **not** included (see `.gitignore`). Obtain data under your own license and run the pipeline locally.

## Features

- Normalize images to RGB JPG and fixed spatial size (`prepare_resnet_images.py`, `--size`).
- Per-class stratified split **80% / 10% / 10%** (`split_dataset.py`).
- Standard `ImageFolder` layout (`build_split_dirs.py`).
- Offline train augmentation: **3×** training images (original + left/right random rotation in `[10°, 20°]` on the **raw** image, then resize) (`build_augmented_trainset.py`).
- Training: pretrained ResNet-18/34, dropout on the head, label smoothing, optional Gaussian noise, **MixUp** (default `--mixup-alpha 0.2`), **ReduceLROnPlateau** or cosine schedule, early stopping, layer-wise LR, optional `--drop-classes`, optional SE blocks, optional A-preprocess.
- Optional **test-time augmentation** (`evaluate_tta.py`).

## Requirements

- **Python 3.10+** recommended.
- **NVIDIA GPU + CUDA** strongly recommended (training uses AMP by default).
- Install Python deps:

```bash
pip install -r requirements.txt
```

Install a **PyTorch** wheel that matches your CUDA version from [pytorch.org](https://pytorch.org) if `pip install torch` is not correct for your machine.

## Input data you must provide

### Directory layout (defaults)

Place the repo root as the working directory. Expected defaults:

| Item | Default path | Purpose |
|------|----------------|--------|
| Label CSV | `dataset.csv` | Master list of images + class names |
| Raw images | `Fundus Dataset/Fundus Dataset/<ClassName>/` | Original files (any extension PIL can read) |

Override with CLI flags on each script if your layout differs (`--csv`, `--images-root` in `prepare_resnet_images.py`, etc.).

### `dataset.csv` format

The preparation script expects a **header row** and at least these columns:

- `image_path` — any path string whose **last two segments** are `ClassName/filename` (e.g. Kaggle-style `.../Other/abc.jpg` works; only `Other/abc.jpg` is used to find `Fundus Dataset/.../Other/abc.*`).
- `class` — class name string (must match a subfolder under the raw images root).
- `label_encoded` — integer label (carried through to `dataset_resnet.csv`).

After step 1, `dataset_resnet.csv` (or your `--output-csv`) uses the same columns with updated `image_path` pointing at normalized JPGs.

### Classes (original project)

`AMD`, `Cataract`, `Diabetes`, `Glaucoma`, `Hypertension`, `Myopia`, `Normal`, `Other`.

The **documented best setup** trains on **6 classes** by dropping `Other` and `Diabetes` at runtime: `--drop-classes "Other,Diabetes"`. Use all 8 classes by omitting `--drop-classes` (expect different metrics).

---

## Reproduce training (end-to-end)

Run **from the repository root** (the folder that contains the `.py` files).

**Reproducibility:** splits and augmentation use **seed 42** by default (`split_dataset.py`, `build_augmented_trainset.py`, `train_resnet18.py --seed 42`). Keep the same seed for comparable numbers.

### A) Full pipeline at **320×320** (matches reported experiments)

Use separate folders for 320 so you do not overwrite 224 outputs:

```bash
# 1) Normalize to JPG 320×320
python prepare_resnet_images.py --size 320 --output-dir resnet_images_jpg_320 --output-csv dataset_resnet_320.csv

# 2) Stratified split
python split_dataset.py --input-csv dataset_resnet_320.csv --output-dir splits_320

# 3) ImageFolder layout (train/val/test)
python build_split_dirs.py --images-dir resnet_images_jpg_320 --splits-dir splits_320 --output-dir resnet_data_320

# 4) Offline 3× train augmentation (reads raw images + train.csv; copies val/test)
python build_augmented_trainset.py --size 320 --splits-dir splits_320 --base-data-dir resnet_data_320 --output-dir resnet_data_aug_320

# 5) Train (6-class, A-preprocess, MixUp 0.2, plateau)
python train_resnet18.py --model-name resnet18 --seed 42 --epochs 30 --batch-size 24 --data-root resnet_data_aug_320 --img-size 320 --noise-prob 0.3 --noise-std 0.02 --num-workers 4 --prefetch-factor 1 --freeze-epochs 4 --layer4-only-epochs 3 --mixup-alpha 0.2 --full-head-lr-scale 0.5 --full-backbone-lr-scale 0.5 --drop-classes "Other,Diabetes" --use-a-preprocess --a-green-alpha 0.35 --a-green-gain 1.15 --scheduler plateau
```

**Windows:** `--num-workers 4 --prefetch-factor 1` avoids common worker/RAM issues. On Linux with enough RAM, you may raise `num_workers` (script default is `8`).

**VRAM:** If CUDA OOM, lower `--batch-size` (e.g. 16).

**Slightly gentler full-unfreeze LR** (alternative from the log):  
`--full-head-lr-scale 0.45 --full-backbone-lr-scale 0.45`

**Disable early stopping** (sanity check): append `--early-stop-patience 999`

### B) Default **224×224** pipeline (smaller disk / faster)

```bash
python prepare_resnet_images.py
python split_dataset.py
python build_split_dirs.py
python build_augmented_trainset.py
python train_resnet18.py --data-root resnet_data_aug --img-size 224 --epochs 30 --batch-size 32
```

Tune regularization and `--drop-classes` as needed; defaults are documented below.

---

## Training outputs

Under `checkpoints/` (or `--output-dir`):

| File | Description |
|------|-------------|
| `best_<model_name>.pth` | Best weights by **validation accuracy** (e.g. `best_resnet18.pth`) |
| `train_meta.json` | Classes, hyperparameters, `best_val_acc`, `test_acc`, etc. |
| `history.csv` | Per-epoch metrics |
| `training_curves.png` | Loss/accuracy curves |
| `test_confusion_matrix.png` | Test-set confusion matrix |

---

## Optional: test-time augmentation

Requires a finished training run so `train_meta.json` and the checkpoint exist. The script reads **`img_size`**, **A-preprocess flags**, and **`drop_classes`** from `train_meta.json` so evaluation matches training.

```bash
python evaluate_tta.py --data-root resnet_data_aug_320 --checkpoint checkpoints/best_resnet18.pth --meta checkpoints/train_meta.json --tta-mode 5fold
```

Earlier project experiments often found **no TTA** better than TTA for this setup; see `EXPERIMENT_LOG.md`.

---

## Default hyperparameters (`train_resnet18.py`)

| Topic | Default |
|--------|---------|
| MixUp | `--mixup-alpha 0.2` |
| Freeze schedule | `--freeze-epochs 4`, `--layer4-only-epochs 3` |
| Regularization | `--dropout 0.4`, `--label-smoothing 0.1`, `--weight-decay 1e-3` |
| Scheduler | `--scheduler plateau` |
| Early stopping | `--early-stop-patience 4`, `--early-stop-min-delta 1e-3` |
| Base LR groups | `--head-lr 5e-5`, `--backbone-lr 1e-5` |
| Full-unfreeze scale | `--full-head-lr-scale 0.5`, `--full-backbone-lr-scale 0.5` |
| Noise | `--noise-std 0.02`, `--noise-prob 0.3` |
| Seed | `--seed 42` |
| AMP | on; `--disable-amp` to turn off |

Use `python train_resnet18.py --help` for the full list (SE blocks, focal loss, cosine scheduler, etc.).

---

## Experiment history

See **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** for round-by-round results, negative results (TTA, SE, focal loss, `mixup_alpha=0.1`), and a summary table.

---

## License / data

This repository contains **code only**. Fundus images and label files are **not** redistributed here. Follow your data provider’s terms and your institution’s policy when publishing.
