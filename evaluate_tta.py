import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FundusAsymmetricPreprocess:
    def __init__(self, green_alpha: float = 0.35, green_gain: float = 1.15) -> None:
        self.green_alpha = green_alpha
        self.green_gain = green_gain

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        r, g, b = image.split()
        g_eq = TF.equalize(g)
        g_mix = Image.blend(g, g_eq, self.green_alpha)
        g_mix = g_mix.point(lambda px: max(0, min(255, int(px * self.green_gain))))
        return Image.merge("RGB", (r, g_mix, b))


class TTADataset(Dataset):
    def __init__(
        self,
        base: datasets.ImageFolder,
        img_size: int,
        use_a: bool,
        a_alpha: float,
        a_gain: float,
        tta_mode: str,
    ):
        self.samples = base.samples
        self.targets = [t for _, t in self.samples]
        self.img_size = img_size
        self.use_a = use_a
        self.a_proc = FundusAsymmetricPreprocess(a_alpha, a_gain)
        self.tta_mode = tta_mode
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare(self, img: Image.Image) -> Image.Image:
        if self.use_a:
            img = self.a_proc(img)
        return img

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self._prepare(img)

        if self.tta_mode == "2fold":
            # lightweight: original + horizontal flip
            views = [img, TF.hflip(img)]
        elif self.tta_mode == "3fold":
            # lightweight+: original + horizontal flip + small rotation
            views = [img, TF.hflip(img), TF.rotate(img, 10)]
        else:
            # full 5-fold
            views = [
                img,
                TF.hflip(img),
                TF.vflip(img),
                TF.vflip(TF.hflip(img)),
                TF.rotate(img, 15),
            ]
        tta_tensors = torch.stack([self.to_tensor(v) for v in views], dim=0)
        return tta_tensors, target


def drop_classes(base: datasets.ImageFolder, class_names: list[str]) -> None:
    for class_name in class_names:
        if class_name not in base.class_to_idx:
            continue
        drop_idx = base.class_to_idx[class_name]
        old_classes = list(base.classes)
        old_samples = list(base.samples)

        old_to_new = {}
        new_classes = []
        for old_idx, cls in enumerate(old_classes):
            if old_idx == drop_idx:
                continue
            old_to_new[old_idx] = len(new_classes)
            new_classes.append(cls)

        new_samples = []
        for path, old_target in old_samples:
            if old_target == drop_idx:
                continue
            new_samples.append((path, old_to_new[old_target]))

        base.classes = new_classes
        base.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}
        base.samples = new_samples
        base.imgs = new_samples
        base.targets = [target for _, target in new_samples]


def build_model(meta: dict, num_classes: int) -> nn.Module:
    if meta["model_name"] == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=meta["dropout"]),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tta_imgs, labels in loader:
            # tta_imgs: [B, 5, C, H, W]
            b, n, c, h, w = tta_imgs.shape
            tta_imgs = tta_imgs.view(b * n, c, h, w).to(device)
            labels = labels.to(device)
            logits = model(tta_imgs).view(b, n, -1)
            probs = torch.softmax(logits, dim=-1).mean(dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with 5-fold TTA.")
    parser.add_argument("--data-root", type=Path, default=Path("resnet_data_aug_320"))
    parser.add_argument("--meta", type=Path, default=Path("checkpoints") / "train_meta.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "best_resnet18.pth")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="5fold",
        choices=["2fold", "3fold", "5fold"],
        help="Choose lightweight or full TTA configuration.",
    )
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    img_size = int(meta.get("img_size", 320))
    use_a = bool(meta.get("use_a_preprocess", False))
    a_alpha = float(meta.get("a_green_alpha", 0.35))
    a_gain = float(meta.get("a_green_gain", 1.15))
    drop_list = list(meta.get("drop_classes", []))

    base = datasets.ImageFolder(args.data_root / "test")
    if drop_list:
        drop_classes(base, drop_list)

    dataset = TTADataset(
        base,
        img_size=img_size,
        use_a=use_a,
        a_alpha=a_alpha,
        a_gain=a_gain,
        tta_mode=args.tta_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta, num_classes=len(base.classes)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    acc = evaluate(model, loader, device)
    print(f"TTA ({args.tta_mode}) test_acc={acc:.4f}")
    print(f"Classes={base.classes}")
    print(f"Dropped classes={drop_list}")


if __name__ == "__main__":
    main()
