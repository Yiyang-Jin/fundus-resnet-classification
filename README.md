# Fundus Image Classification (ResNet-18 / ResNet-34)

PyTorch pipeline for **stratified splitting**, **offline augmentation**, and **transfer learning** on a multi-class fundus dataset. The training script targets `torchvision` ResNet models with optional asymmetric green-channel preprocessing, MixUp, AMP, and progressive unfreezing.

> **This folder** is the GitHub-ready copy: scripts, `requirements.txt`, `.gitignore`, and documentation. **Do not commit** raw images, large CSVs, or `.pth` weights (see `.gitignore`).

## Features

- Normalize images to RGB JPG and fixed spatial size (`prepare_resnet_images.py`, configurable `--size`).
- Per-class stratified split **80% / 10% / 10%** (`split_dataset.py`).
- Standard `ImageFolder` layout (`build_split_dirs.py`).
- Offline train augmentation: **3×** samples per training image (original + left/right random rotation 10–20° before resize) (`build_augmented_trainset.py`).
- Training: pretrained ResNet-18 or ResNet-34, dropout head, label smoothing, optional noise, **MixUp** (`--mixup-alpha`, default **`0.2`**), **ReduceLROnPlateau** or cosine schedule, early stopping, layer-wise LR, optional class drops, optional SE blocks, optional A-preprocess.
- Optional **test-time augmentation** evaluation (`evaluate_tta.py`).

## Requirements

- Python 3.10+ recommended  
- NVIDIA GPU with CUDA (optional but expected for reasonable training time)  
- Install:

```bash
pip install -r requirements.txt
```

PyTorch: install a build that matches your CUDA version from [pytorch.org](https://pytorch.org) if the default wheel is not suitable.

## Expected data layout

Place your repository root **next to** the scripts (this folder), or pass paths via CLI flags.

**Raw data (example):**

- `Fundus Dataset/Fundus Dataset/<ClassName>/*.{jpg,png,...}`
- Label file: `dataset.csv` (paths and class labels as produced by your dataset)

**Classes in the original project:** `AMD`, `Cataract`, `Diabetes`, `Glaucoma`, `Hypertension`, `Myopia`, `Normal`, `Other`.

The **recommended production setting** for this codebase’s experiments uses **6 classes** by dropping `Other` and `Diabetes` at train time (`--drop-classes "Other,Diabetes"`). Adjust if your task requires all 8 classes.

## Pipeline (run from this directory)

### 1) Normalize images for ResNet

Default output: `224×224` JPG + `dataset_resnet.csv`.

```bash
python prepare_resnet_images.py
```

For the **320×320** pipeline used in the best reported runs:

```bash
python prepare_resnet_images.py --size 320 --out-dir resnet_images_jpg_320 --out-csv dataset_resnet_320.csv
```

(Adjust `--raw-root` / `--csv` if your paths differ.)

### 2) Stratified split

```bash
python split_dataset.py
```

Use `--input-csv` / `--out-dir` if you used non-default names for the 320 pipeline.

### 3) Build folder layout

```bash
python build_split_dirs.py
```

### 4) Offline-augmented training set

```bash
python build_augmented_trainset.py
```

For 320-based trees, pass matching `--size`, `--splits-dir`, `--base-data-dir`, and `--output-dir` (e.g. `resnet_data_aug_320`).

### 5) Train

**Windows:** if you see worker or memory errors, use `--num-workers 4` and `--prefetch-factor 1` (as in the commands below).

**Strongest observed configuration (6 classes, A-preprocess, MixUp 0.2, plateau, LR scales 0.5):**

```bash
python train_resnet18.py --model-name resnet18 --epochs 30 --batch-size 24 --data-root "resnet_data_aug_320" --img-size 320 --noise-prob 0.3 --noise-std 0.02 --num-workers 4 --prefetch-factor 1 --freeze-epochs 4 --layer4-only-epochs 3 --mixup-alpha 0.2 --full-head-lr-scale 0.5 --full-backbone-lr-scale 0.5 --drop-classes "Other,Diabetes" --use-a-preprocess --a-green-alpha 0.35 --a-green-gain 1.15 --scheduler plateau
```

**Alternative fine-tune (slightly gentler full-unfreeze LR):** use `--full-head-lr-scale 0.45 --full-backbone-lr-scale 0.45` (see `EXPERIMENT_LOG.md`).

**Train without early stopping** (e.g. sanity check):

```bash
python train_resnet18.py ... --early-stop-patience 999
```

### Outputs

Written under `checkpoints/` by default:

- `best_resnet18.pth` — best by validation accuracy  
- `train_meta.json` — classes, hyperparameters, metrics  
- `history.csv`, `training_curves.png`, `test_confusion_matrix.png`

### Optional: TTA evaluation

```bash
python evaluate_tta.py --tta-mode 5fold
```

(Previous experiments on this project often favored **no TTA**; see log.)

## Default hyperparameters (script defaults)

Key defaults in `train_resnet18.py` include:

- `--mixup-alpha 0.2`  
- `--freeze-epochs 4`, `--layer4-only-epochs 3`  
- `--dropout 0.4`, `--label-smoothing 0.1`, `--weight-decay 1e-3`  
- `--scheduler plateau`  
- `--early-stop-patience 4`  
- Base LRs: `--head-lr 5e-5`, `--backbone-lr 1e-5`; after full unfreeze, scales `--full-head-lr-scale 0.5`, `--full-backbone-lr-scale 0.5`  

Override any of these via CLI.

## Experiment history

See **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** for round-by-round results, failed trials (TTA, SE, focal loss, `mixup_alpha=0.1`), and the consolidated summary table.

## License / data

This repository contains **code only**. Fundus images and labels are **not** included. Use your institution’s data policy and licenses when publishing.

## 中文说明（简要）

本目录为可上传 **GitHub** 的独立副本：含全部脚本、`requirements.txt`、`.gitignore` 与英文文档。训练默认已恢复 **`mixup_alpha=0.2`**。原始数据与权重请勿提交；完整实验记录见 `EXPERIMENT_LOG.md`。
