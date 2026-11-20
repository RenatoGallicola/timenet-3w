"""How much does a careless split inflate the scores? (issue #4)

Usage:
    uv run python scripts/split_leakage.py <windows folder> <output folder>

Runs the XGBoost `shape` baseline of scripts/baselines.py with three 5-fold cross-validations over
the same windows, from the most to the least leaky:
- **random**: windows are assigned to folds at random, so overlapping windows of one instance land
  on both sides;
- **by instance**: each instance stays in one fold, but other instances of the same well do not;
- **by well**: each well stays in one fold, as in the project's evaluation.

Only the train and validation windows are used: the test wells are frozen (docs/DECISIONS.md).
Reports macro-F1 over the 6 classes for each scheme and writes a chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402

KIND = "shape"
N_FOLDS = 5
SCHEMES = {"random": None, "by instance": "instance", "by well": "well"}


def run(windows_dir: Path) -> dict:
    labels = [int(c) for c in json.loads((windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    parts = [baselines.load(windows_dir, split) for split in ("train", "val")]
    X = np.concatenate([p[0] for p in parts])
    mask = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    meta = pd.concat([p[3] for p in parts], ignore_index=True)

    results = {}
    for name, group in SCHEMES.items():
        if group is None:
            folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X, y)
        else:
            folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X, y, meta[group])
        scores = []
        for train, test in folds:
            model = baselines.fit(X[train], mask[train], y[train], KIND, labels)
            pred = baselines.predict(model, X[test], mask[test], KIND, labels)
            scores.append(float(f1_score(y[test], pred, labels=labels, average="macro", zero_division=0)))
        results[name] = {
            "macro_f1_mean": round(float(np.mean(scores)), 4),
            "macro_f1_std": round(float(np.std(scores)), 4),
            "folds": [round(s, 4) for s in scores],
        }
        print(f"{name:12s} macro-F1 {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    return {
        "model": f"xgboost_{KIND}",
        "windows": int(len(y)),
        "wells": int(meta["well"].nunique()),
        "instances": int(meta["instance"].nunique()),
        "note": "Train and validation windows only; macro-F1 over the 6 classes, 5 folds per scheme.",
        "schemes": results,
    }


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["schemes"])
    means = [summary["schemes"][n]["macro_f1_mean"] for n in names]
    stds = [summary["schemes"][n]["macro_f1_std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=["#c44e52", "#dd8452", "#4c72b0"])
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.2f}", ha="center")
    ax.set_ylabel("macro-F1 over 6 classes (5-fold CV)")
    ax.set_ylim(0, 1)
    ax.set_title("Same model, same windows: how the split changes the score")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    summary = run(args.windows_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "split_leakage.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(summary, args.out_dir / "split_leakage.png")


if __name__ == "__main__":
    main()
