import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.3) -> None:
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class FundusAsymmetricPreprocess:
    """
    A-strategy preprocessing:
    - enhance local contrast mainly on green channel
    - slightly increase green-channel contribution
    """

    def __init__(self, green_alpha: float = 0.35, green_gain: float = 1.15) -> None:
        self.green_alpha = green_alpha
        self.green_gain = green_gain

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        r, g, b = image.split()
        g_eq = ImageOps.equalize(g)
        g_mix = Image.blend(g, g_eq, self.green_alpha)
        g_mix = g_mix.point(lambda px: max(0, min(255, int(px * self.green_gain))))
        return Image.merge("RGB", (r, g_mix, b))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet model on fundus dataset.")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--data-root", type=Path, default=Path("resnet_data_aug"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--layer4-only-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--full-head-lr-scale", type=float, default=0.5)
    parser.add_argument("--full-backbone-lr-scale", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--glaucoma-weight", type=float, default=1.0)
    parser.add_argument("--use-se-layer4", action="store_true")
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine"],
        help="Learning-rate scheduler type.",
    )
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--noise-prob", type=float, default=0.3)
    parser.add_argument("--use-a-preprocess", action="store_true")
    parser.add_argument("--a-green-alpha", type=float, default=0.35)
    parser.add_argument("--a-green-gain", type=float, default=1.15)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Online random rotation degrees. Set 0 when using offline augmented dataset.",
    )
    parser.add_argument("--use-hflip", action="store_true")
    parser.add_argument("--drop-other", action="store_true")
    parser.add_argument(
        "--drop-classes",
        type=str,
        default="",
        help="Comma-separated class names to exclude, e.g. 'Other,Diabetes'.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(
    rotation_deg: float,
    use_hflip: bool,
    noise_std: float,
    noise_prob: float,
    img_size: int,
    use_a_preprocess: bool,
    a_green_alpha: float,
    a_green_gain: float,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = []
    eval_tf_ops = []
    if use_a_preprocess:
        preprocess = FundusAsymmetricPreprocess(
            green_alpha=a_green_alpha, green_gain=a_green_gain
        )
        train_tf.append(preprocess)
        eval_tf_ops.append(preprocess)

    train_tf.extend(
        [
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=rotation_deg),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ]
    )
    if use_hflip:
        train_tf.append(transforms.RandomHorizontalFlip(p=0.5))
    train_tf.extend(
        [
            transforms.ToTensor(),
            AddGaussianNoise(std=noise_std, p=noise_prob),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_tf_ops.extend(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(eval_tf_ops)
    return transforms.Compose(train_tf), eval_tf


def drop_class_from_imagefolder(dataset: datasets.ImageFolder, class_name: str) -> int:
    if class_name not in dataset.class_to_idx:
        return 0

    drop_idx = dataset.class_to_idx[class_name]
    old_classes = list(dataset.classes)
    old_samples = list(dataset.samples)

    old_to_new = {}
    new_classes = []
    for old_idx, cls in enumerate(old_classes):
        if old_idx == drop_idx:
            continue
        old_to_new[old_idx] = len(new_classes)
        new_classes.append(cls)

    new_samples = []
    removed = 0
    for path, old_target in old_samples:
        if old_target == drop_idx:
            removed += 1
            continue
        new_samples.append((path, old_to_new[old_target]))

    dataset.classes = new_classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}
    dataset.samples = new_samples
    dataset.imgs = new_samples
    dataset.targets = [target for _, target in new_samples]
    return removed


def drop_classes_from_imagefolder(dataset: datasets.ImageFolder, class_names: list[str]) -> dict[str, int]:
    removed_stats: dict[str, int] = {}
    for name in class_names:
        removed_stats[name] = drop_class_from_imagefolder(dataset, name)
    return removed_stats


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_with_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: GradScaler,
    use_amp: bool,
    mixup_alpha: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        do_mixup = mixup_alpha > 0
        if do_mixup:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            index = torch.randperm(images.size(0), device=device)
            mixed_images = lam * images + (1.0 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
        else:
            mixed_images = images
            labels_a, labels_b, lam = labels, labels, 1.0

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed_images)
            if do_mixup:
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        if do_mixup:
            # MixUp does not have a strict hard-label accuracy; this weighted score is a useful proxy.
            correct += (
                lam * (preds == labels_a).sum().item()
                + (1.0 - lam) * (preds == labels_b).sum().item()
            )
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def set_trainable_params(model: nn.Module, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if mode in {"layer4_head", "full"}:
        for param in model.layer4.parameters():
            param.requires_grad = True

    if mode == "full":
        for name, param in model.named_parameters():
            if not name.startswith("fc.") and not name.startswith("layer4."):
                param.requires_grad = True


def build_optimizer(args: argparse.Namespace, model: nn.Module, mode: str) -> optim.Optimizer:
    if mode == "head_only":
        params = [{"params": model.fc.parameters(), "lr": args.head_lr}]
    elif mode == "layer4_head":
        params = [
            {"params": model.layer4.parameters(), "lr": args.backbone_lr},
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ]
    else:
        full_backbone_lr = args.backbone_lr * args.full_backbone_lr_scale
        full_head_lr = args.head_lr * args.full_head_lr_scale
        backbone_params = []
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                backbone_params.append(param)
        params = [
            {"params": backbone_params, "lr": full_backbone_lr},
            {"params": model.fc.parameters(), "lr": full_head_lr},
        ]

    return optim.AdamW(params, weight_decay=args.weight_decay)


def attach_se_to_layer4(model: nn.Module, reduction: int) -> None:
    wrapped = []
    for block in model.layer4:
        out_channels = block.conv2.out_channels
        wrapped.append(nn.Sequential(block, SEBlock(out_channels, reduction=reduction)))
    model.layer4 = nn.Sequential(*wrapped)


def save_history_csv(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_backbone", "lr_head"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_curves(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, marker="o", label="Train Acc")
    plt.plot(epochs, val_acc, marker="o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(
    y_true: list[int], y_pred: list[int], class_names: list[str], path: Path
) -> None:
    n = len(class_names)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = list(range(n))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    max_v = max(max(row) for row in cm) if cm else 0
    thresh = max_v / 2 if max_v > 0 else 0
    for i in range(n):
        for j in range(n):
            plt.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color="white" if cm[i][j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(
        args.rotation_deg,
        args.use_hflip,
        args.noise_std,
        args.noise_prob,
        args.img_size,
        args.use_a_preprocess,
        args.a_green_alpha,
        args.a_green_gain,
    )

    train_set = datasets.ImageFolder(args.data_root / "train", transform=train_tf)
    val_set = datasets.ImageFolder(args.data_root / "val", transform=eval_tf)
    test_set = datasets.ImageFolder(args.data_root / "test", transform=eval_tf)

    classes_to_drop: list[str] = []
    if args.drop_other:
        classes_to_drop.append("Other")
    if args.drop_classes.strip():
        classes_to_drop.extend([c.strip() for c in args.drop_classes.split(",") if c.strip()])
    classes_to_drop = list(dict.fromkeys(classes_to_drop))

    if classes_to_drop:
        train_stats = drop_classes_from_imagefolder(train_set, classes_to_drop)
        val_stats = drop_classes_from_imagefolder(val_set, classes_to_drop)
        test_stats = drop_classes_from_imagefolder(test_set, classes_to_drop)
        print(
            f"Dropped classes {classes_to_drop} -> "
            f"train={train_stats}, val={val_stats}, test={test_stats}",
            flush=True,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if args.model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if args.use_se_layer4:
        attach_se_to_layer4(model, reduction=args.se_reduction)
    num_classes = len(train_set.classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=args.dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)

    current_mode = "head_only"
    set_trainable_params(model, current_mode)
    optimizer = build_optimizer(args, model, current_mode)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
    scaler = GradScaler(device=device.type, enabled=use_amp)
    class_weights = torch.ones(len(train_set.classes), dtype=torch.float32)
    if "Glaucoma" in train_set.class_to_idx and args.glaucoma_weight != 1.0:
        class_weights[train_set.class_to_idx["Glaucoma"]] = args.glaucoma_weight
        print(f"Applied class weight for Glaucoma: {args.glaucoma_weight}", flush=True)
    if args.loss_type == "focal":
        train_criterion = FocalLoss(gamma=args.focal_gamma, class_weights=class_weights)
    else:
        train_criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
            weight=class_weights.to(device),
        )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    no_improve_count = 0
    layer4_unfrozen = False
    backbone_unfrozen = False

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Classes ({num_classes}): {train_set.classes}")

    for epoch in range(1, args.epochs + 1):
        if (not layer4_unfrozen) and epoch > args.freeze_epochs:
            current_mode = "layer4_head"
            set_trainable_params(model, current_mode)
            optimizer = build_optimizer(args, model, current_mode)
            if args.scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - epoch + 1)
                )
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
                )
            layer4_unfrozen = True
            print("Layer4 unfrozen. Fine-tuning layer4 + head.", flush=True)

        if (not backbone_unfrozen) and epoch > (args.freeze_epochs + args.layer4_only_epochs):
            current_mode = "full"
            set_trainable_params(model, current_mode)
            optimizer = build_optimizer(args, model, current_mode)
            if args.scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - epoch + 1)
                )
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
                )
            backbone_unfrozen = True
            print("Full backbone unfrozen. Fine-tuning all layers.", flush=True)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_criterion,
            scaler,
            use_amp,
            args.mixup_alpha,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        , flush=True)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr_backbone": round(optimizer.param_groups[0]["lr"], 10),
                "lr_head": round(optimizer.param_groups[-1]["lr"], 10),
            }
        )
        save_history_csv(history, args.output_dir / "history.csv")
        save_curves(history, args.output_dir / "training_curves.png")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if val_loss + args.early_stop_min_delta < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= args.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_loss improvement for {args.early_stop_patience} epochs).",
                    flush=True,
                )
                break

    best_ckpt_path = args.output_dir / f"best_{args.model_name}.pth"
    torch.save(best_state, best_ckpt_path)
    print(f"Saved best checkpoint to: {best_ckpt_path.resolve()}")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds, test_labels = evaluate_with_predictions(
        model, test_loader, device, use_amp
    )
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    history_csv_path = args.output_dir / "history.csv"
    curves_path = args.output_dir / "training_curves.png"
    cm_path = args.output_dir / "test_confusion_matrix.png"
    save_history_csv(history, history_csv_path)
    save_curves(history, curves_path)
    save_confusion_matrix(test_labels, test_preds, train_set.classes, cm_path)
    print(f"Saved training history to: {history_csv_path.resolve()}")
    print(f"Saved training curves to: {curves_path.resolve()}")
    print(f"Saved test confusion matrix to: {cm_path.resolve()}")

    meta = {
        "model_name": args.model_name,
        "classes": train_set.classes,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "full_head_lr_scale": args.full_head_lr_scale,
        "full_backbone_lr_scale": args.full_backbone_lr_scale,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "freeze_epochs": args.freeze_epochs,
        "layer4_only_epochs": args.layer4_only_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "rotation_deg": args.rotation_deg,
        "use_hflip": args.use_hflip,
        "drop_other": args.drop_other,
        "drop_classes": classes_to_drop,
        "noise_std": args.noise_std,
        "noise_prob": args.noise_prob,
        "use_a_preprocess": args.use_a_preprocess,
        "a_green_alpha": args.a_green_alpha,
        "a_green_gain": args.a_green_gain,
        "mixup_alpha": args.mixup_alpha,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "glaucoma_weight": args.glaucoma_weight,
        "use_se_layer4": args.use_se_layer4,
        "se_reduction": args.se_reduction,
        "scheduler": args.scheduler,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "amp_enabled": use_amp,
    }
    meta_path = args.output_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata to: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
