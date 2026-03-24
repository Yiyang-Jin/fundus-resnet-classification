import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset CSV into stratified train/val/test sets."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("dataset_resnet.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Directory to save split CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def split_one_class(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    rng.shuffle(rows)
    n = len(rows)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val : n_train + n_val + n_test]
    return train_rows, val_rows, test_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    with args.input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["image_path", "class", "label_encoded"]
        by_class = defaultdict(list)
        for row in reader:
            by_class[row["class"]].append(row)

    train_all: list[dict] = []
    val_all: list[dict] = []
    test_all: list[dict] = []

    print("Per-class split summary:")
    for class_name in sorted(by_class.keys()):
        rows = by_class[class_name]
        train_rows, val_rows, test_rows = split_one_class(rows, rng)
        train_all.extend(train_rows)
        val_all.extend(val_rows)
        test_all.extend(test_rows)
        print(
            f"- {class_name}: total={len(rows)}, "
            f"train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}"
        )

    # Shuffle each split so classes are mixed for training loops.
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    write_csv(args.output_dir / "train.csv", train_all, fieldnames)
    write_csv(args.output_dir / "val.csv", val_all, fieldnames)
    write_csv(args.output_dir / "test.csv", test_all, fieldnames)

    print("\nOverall:")
    print(f"train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
