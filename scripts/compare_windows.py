"""Check that two window folders made by make_windows.py hold the same windows.

Usage:
    uv run python scripts/compare_windows.py <windows folder A> <windows folder B>

Compares, for each split, the window metadata (`{split}.csv`) and the arrays (`{split}.npz`):
labels and masks must be equal, and values must be equal with NaN counted as equal to NaN.
Exits with code 1 and names the first difference otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def compare(a: Path, b: Path) -> list[str]:
    """Return the differences between two window folders, empty if they match."""
    problems = []
    for split in SPLITS:
        meta_a, meta_b = pd.read_csv(a / f"{split}.csv"), pd.read_csv(b / f"{split}.csv")
        if not meta_a.equals(meta_b):
            diff = meta_a.compare(meta_b) if meta_a.shape == meta_b.shape else f"shapes {meta_a.shape} vs {meta_b.shape}"
            problems.append(f"{split}.csv differs:\n{diff}")
            continue
        npz_a, npz_b = np.load(a / f"{split}.npz"), np.load(b / f"{split}.npz")
        for key in ("y", "mask"):
            if not np.array_equal(npz_a[key], npz_b[key]):
                problems.append(f"{split}.npz: {key} differs")
        x_a, x_b = npz_a["X"], npz_b["X"]
        if x_a.shape != x_b.shape:
            problems.append(f"{split}.npz: X shapes {x_a.shape} vs {x_b.shape}")
        elif not np.array_equal(x_a, x_b, equal_nan=True):
            bad = ~((x_a == x_b) | (np.isnan(x_a) & np.isnan(x_b)))
            window = int(np.argwhere(bad)[0][0])
            problems.append(f"{split}.npz: X differs in {int(bad.sum())} values, first in window {meta_a['window_id'][window]}")
        print(f"{split}: {len(meta_a)} windows compared")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    problems = compare(args.a, args.b)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("The two window folders are identical.")


if __name__ == "__main__":
    main()
