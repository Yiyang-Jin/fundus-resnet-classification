import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test directory structure from split CSV files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing resnet_images_jpg and splits.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("resnet_images_jpg"),
        help="Directory of normalized source images.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("splits"),
        help="Directory containing train.csv/val.csv/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resnet_data"),
        help="Output directory for normalized split folders.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="File operation mode. copy keeps source images; move relocates them.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def build_one_split(
    split_name: str, rows: list[dict], root: Path, images_dir: Path, output_dir: Path, mode: str
) -> tuple[int, int]:
    copied = 0
    missing = 0
    for row in rows:
        rel = Path(row["image_path"])
        src = root / rel
        if not src.exists():
            src = root / images_dir / Path(row["class"]) / Path(rel).name
        if not src.exists():
            missing += 1
            continue

        dst = root / output_dir / split_name / row["class"] / Path(rel).name
        transfer_file(src, dst, mode)
        copied += 1
    return copied, missing


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    splits_dir = root / args.splits_dir

    summary = {}
    for split_name in ["train", "val", "test"]:
        rows = load_rows(splits_dir / f"{split_name}.csv")
        copied, missing = build_one_split(
            split_name=split_name,
            rows=rows,
            root=root,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mode=args.mode,
        )
        summary[split_name] = (copied, missing)

    print("Directory normalization complete:")
    for split_name in ["train", "val", "test"]:
        copied, missing = summary[split_name]
        print(f"- {split_name}: files={copied}, missing={missing}")
    print(f"Output: {(root / args.output_dir)}")


if __name__ == "__main__":
    main()
