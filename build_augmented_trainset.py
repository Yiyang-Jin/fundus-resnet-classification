import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline augmented dataset with 3x train images."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--raw-images-root",
        type=Path,
        default=Path("Fundus Dataset") / "Fundus Dataset",
        help="Raw source images root by class folders.",
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--base-data-dir", type=Path, default=Path("resnet_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("resnet_data_aug"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-deg", type=float, default=10.0)
    parser.add_argument("--max-deg", type=float, default=20.0)
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_raw_image(raw_root: Path, class_name: str, filename_from_csv: str) -> Path:
    cls_dir = raw_root / class_name
    stem = Path(filename_from_csv).stem

    exact = cls_dir / filename_from_csv
    if exact.exists():
        return exact

    candidates = sorted(cls_dir.glob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"Raw image not found for class={class_name}, stem={stem}")
    return candidates[0]


def save_processed(img: Image.Image, out_path: Path, size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    img.save(out_path, format="JPEG", quality=95)


def rotate_then_resize(img: Image.Image, angle: float, size: int) -> Image.Image:
    # Keep source resolution during rotation, then resize to target.
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    rotated = rotated.convert("RGB")
    rotated = rotated.resize((size, size), Image.BILINEAR)
    return rotated


def build_train(args: argparse.Namespace, rng: random.Random) -> tuple[int, int]:
    train_rows = read_rows(args.root / args.splits_dir / "train.csv")
    written = 0
    missing = 0

    for row in train_rows:
        class_name = row["class"]
        csv_rel = Path(row["image_path"])
        filename = csv_rel.name

        try:
            raw_path = find_raw_image(args.root / args.raw_images_root, class_name, filename)
        except FileNotFoundError:
            missing += 1
            continue

        with Image.open(raw_path) as src:
            base_name = Path(filename).stem
            out_cls = args.root / args.output_dir / "train" / class_name

            # 1) original
            original_out = out_cls / f"{base_name}_orig.jpg"
            save_processed(src, original_out, args.size)
            written += 1

            # 2) left random rotation
            left_deg = -rng.uniform(args.min_deg, args.max_deg)
            left = rotate_then_resize(src, left_deg, args.size)
            left_out = out_cls / f"{base_name}_left{abs(left_deg):.1f}.jpg"
            left.save(left_out, format="JPEG", quality=95)
            written += 1

            # 3) right random rotation
            right_deg = rng.uniform(args.min_deg, args.max_deg)
            right = rotate_then_resize(src, right_deg, args.size)
            right_out = out_cls / f"{base_name}_right{right_deg:.1f}.jpg"
            right.save(right_out, format="JPEG", quality=95)
            written += 1

    return written, missing


def copy_eval_split(args: argparse.Namespace, split: str) -> int:
    src_root = args.root / args.base_data_dir / split
    dst_root = args.root / args.output_dir / split
    count = 0

    for class_dir in src_root.iterdir():
        if not class_dir.is_dir():
            continue
        dst_class = dst_root / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)
        for img_path in class_dir.iterdir():
            if not img_path.is_file():
                continue
            shutil.copy2(img_path, dst_class / img_path.name)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ensure_clean_dir(args.root / args.output_dir)

    train_written, train_missing = build_train(args, rng)
    val_copied = copy_eval_split(args, "val")
    test_copied = copy_eval_split(args, "test")

    print("Augmented dataset build complete:")
    print(f"- train written: {train_written} (expected ~ 3x original)")
    print(f"- train missing raw sources: {train_missing}")
    print(f"- val copied: {val_copied}")
    print(f"- test copied: {test_copied}")
    print(f"Output dir: {(args.root / args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
