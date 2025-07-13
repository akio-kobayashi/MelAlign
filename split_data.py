import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional

HEADER = ["key", "source", "target"]

def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not set(HEADER).issubset(rows[0].keys()):
        raise ValueError(f"CSV must contain columns: {HEADER}")
    return rows

def save_csv(rows: List[Dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")

def split_csv(
    src: Path,
    out_dir: Path,
    exclude_keys: Optional[List[str]],
    include_keys: Optional[List[str]],
    sample_n: Optional[int],
    train_ratio: float,
    seed: int = 42,
):
    """Filter by key substrings (include first, then exclude), optionally subsample, then split."""
    random.seed(seed)
    rows = load_csv(src)

    # --- include only rows whose key contains any include substring ---
    if include_keys:
        before_count = len(rows)
        rows = [r for r in rows if any(substr in r['key'] for substr in include_keys)]
        after_count = len(rows)
        print(f"Included {after_count} / {before_count} rows containing substrings {include_keys}")

    # --- filter out rows whose key contains any exclude substring ---
    if exclude_keys:
        before_count = len(rows)
        rows = [
            r for r in rows
            if not any(substr in r['key'] for substr in exclude_keys)
        ]
        after_count = len(rows)
        print(f"Excluded {before_count - after_count} rows containing substrings {exclude_keys}")

    # --- optional subsampling ---
    if sample_n is not None and sample_n < len(rows):
        original = len(rows)
        rows = random.sample(rows, sample_n)
        print(f"Sub-sampled {sample_n} / {original} rows from CSV")

    random.shuffle(rows)
    n_train = int(len(rows) * train_ratio)
    train_rows, val_rows = rows[:n_train], rows[n_train:]

    save_csv(train_rows, out_dir / "train.csv")
    save_csv(val_rows,   out_dir / "val.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Filter rows by key, optionally subsample and split CSV into train/val"
    )
    ap.add_argument(
        "--src", type=Path, required=True, help="input CSV file"
    )
    ap.add_argument(
        "--out_dir", type=Path, default="./", help="output directory"
    )
    ap.add_argument(
        "--exclude", "-e",
        type=str,
        nargs='+',
        default=[],
        help="list of substrings in key to exclude (e.g. --exclude sub1 sub2)"
    )
    ap.add_argument(
        "--include", "-i",
        type=str,
        nargs='+',
        default=[],
        help="list of substrings in key to include"
    )
    ap.add_argument(
        "--num", type=int, default=None,
        help="number of rows to sample before split"
    )
    ap.add_argument(
        "--ratio", type=float, default=0.9,
        help="train split ratio"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    args = ap.parse_args()

    split_csv(
        args.src,
        args.out_dir,
        args.exclude,
        args.include,
        args.num,
        args.ratio,
        args.seed,
    )
