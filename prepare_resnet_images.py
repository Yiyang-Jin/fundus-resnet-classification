import argparse
import csv
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize fundus images for ResNet training."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset.csv"),
        help="Input CSV path (default: dataset.csv).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("Fundus Dataset") / "Fundus Dataset",
        help="Root folder containing class subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resnet_images_jpg"),
        help="Output folder for normalized images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset_resnet.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target width/height for ResNet input (default: 224).",
    )
    return parser.parse_args()


def local_relative_path(raw_path: str) -> Path:
    path_obj = Path(raw_path.replace("\\", "/"))
    # In CSV the path uses Kaggle prefix, e.g. .../Other/abc.jpg.
    # We only need <class>/<filename> to locate local files.
    parts = path_obj.parts
    if len(parts) < 2:
        raise ValueError(f"Invalid image_path: {raw_path}")
    return Path(parts[-2]) / parts[-1]


def process_image(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        image = image.resize((size, size), Image.BILINEAR)
        image.save(dst, format="JPEG", quality=95)


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")

    converted = 0
    skipped = 0

    with args.csv.open("r", newline="", encoding="utf-8") as in_f, args.output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=["image_path", "class", "label_encoded"])
        writer.writeheader()

        for row in reader:
            rel = local_relative_path(row["image_path"])
            src = args.images_root / rel

            new_name = Path(rel.name).stem + ".jpg"
            dst_rel = Path(row["class"]) / new_name
            dst = args.output_dir / dst_rel

            if not src.exists():
                skipped += 1
                continue

            process_image(src, dst, args.size)
            converted += 1

            writer.writerow(
                {
                    "image_path": str(dst.as_posix()),
                    "class": row["class"],
                    "label_encoded": row["label_encoded"],
                }
            )

    print(f"Done. Converted: {converted}, Skipped (missing source): {skipped}")
    print(f"Output images: {args.output_dir.resolve()}")
    print(f"Output csv: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
