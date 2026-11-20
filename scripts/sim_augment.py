"""Do the simulated 3W instances help on real wells? (issue #5)

Usage:
    uv run python scripts/sim_augment.py <TimeNet registry> <windows folder> <output folder>

Cuts windows from the simulated instances of the main classes with the same rules as the real
windows (scripts/make_windows.py). 3W has simulated instances only for classes 2, 8 and 9. Then the
XGBoost `shape` baseline is trained twice, on real training windows only and on real plus simulated
windows, and both are scored **on real wells only**:
- on the validation wells;
- with a 5-fold cross-validation grouped by well over the real train and validation windows, where
  the simulated windows are added to the training folds only.

The test wells are frozen (docs/DECISIONS.md) and not used. Simulated instances never record
P-ANULAR, so a model could learn "P-ANULAR missing means a simulated event class"; the sensor
dropout of the baselines softens this, and the scores on real wells show whether it matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402
import make_windows as mw  # noqa: E402

KIND = "shape"
N_FOLDS = 5


def simulated_windows(registry: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of the simulated instances of the main classes."""
    windows = []
    for instance, event, values, labels in mw.iter_timenet(registry, "simulated"):
        if event in mw.CLASSES:
            windows.extend(mw.windows_of_instance(instance, event, values, labels))
    X = np.stack([w["x"] for w in windows])
    mask = np.stack([w["mask"] for w in windows])
    y = np.array([w["label"] for w in windows])
    return X, mask, y


def scores(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict:
    per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class_f1": {str(c): round(float(f), 4) for c, f in zip(labels, per_class)},
    }


def train_and_score(train: tuple, test: tuple, extra: tuple | None, labels: list[int]) -> dict:
    X, mask, y = train
    if extra is not None:
        X, mask, y = np.concatenate([X, extra[0]]), np.concatenate([mask, extra[1]]), np.concatenate([y, extra[2]])
    model = baselines.fit(X, mask, y, KIND, labels)
    return scores(test[2], baselines.predict(model, test[0], test[1], KIND, labels), labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry", type=Path)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    labels = [int(c) for c in json.loads((args.windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    sim = simulated_windows(args.registry)
    sim_counts = {str(c): int((sim[2] == c).sum()) for c in labels}
    print("simulated windows by class:", sim_counts)

    tr = baselines.load(args.windows_dir, "train")
    va = baselines.load(args.windows_dir, "val")
    result = {"simulated_windows_by_class": sim_counts, "validation": {}, "cv_by_well": {}}
    for name, extra in (("real only", None), ("real + simulated", sim)):
        result["validation"][name] = train_and_score(tr[:3], va[:3], extra, labels)
        print(f"validation {name:17s} macro-F1 {result['validation'][name]['macro_f1']:.3f}  {result['validation'][name]['per_class_f1']}")

    X = np.concatenate([tr[0], va[0]])
    mask = np.concatenate([tr[1], va[1]])
    y = np.concatenate([tr[2], va[2]])
    wells = pd.concat([tr[3], va[3]], ignore_index=True)["well"]
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X, y, wells))
    for name, extra in (("real only", None), ("real + simulated", sim)):
        runs = [train_and_score((X[a], mask[a], y[a]), (X[b], mask[b], y[b]), extra, labels) for a, b in folds]
        macro = [r["macro_f1"] for r in runs]
        result["cv_by_well"][name] = {
            "macro_f1_mean": round(float(np.mean(macro)), 4),
            "macro_f1_std": round(float(np.std(macro)), 4),
            "per_class_f1_mean": {str(c): round(float(np.mean([r["per_class_f1"][str(c)] for r in runs])), 4) for c in labels},
        }
        cv = result["cv_by_well"][name]
        print(f"CV by well {name:17s} macro-F1 {cv['macro_f1_mean']:.3f} +/- {cv['macro_f1_std']:.3f}  {cv['per_class_f1_mean']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sim_augment.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
