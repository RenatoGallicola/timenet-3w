"""Cut the train and validation windows into 3 folds by well, for cross-validation (issue #25).

Usage:
    uv run python scripts/make_folds.py <windows folder> <output folder> [--xgboost]

The test wells are frozen and left out. The train and validation wells of the connector split are
pooled and assigned to 3 folds, each well to one fold. Among fixed seeds of a stratified group
k-fold, the script keeps the first assignment where every event class has windows in every
held-out fold, preferring the one whose per-class window shares are closest to 1/3.

For each fold k it writes `<output>/fold<k>/` in the format of docs/DATA_CONTRACT.md: `train` holds
the windows of the other two folds, `val` the held-out windows of fold k (there is no `test`), plus
`meta.json`. The well-to-fold assignment is saved in `splits/folds_by_well.json`.

With `--xgboost`, the XGBoost `shape` baseline is trained and scored on the same folds, as the
reference for OpenTSLM: per-fold and mean macro-F1 over the 6 classes, event recall and false
alarms.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402

N_FOLDS = 3
N_SEEDS = 500
SPLITS_FILE = Path(__file__).resolve().parent.parent / "splits" / "folds_by_well.json"


def pooled(windows_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    parts = [baselines.load(windows_dir, split) for split in ("train", "val")]
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
        pd.concat([p[3] for p in parts], ignore_index=True),
    )


def search_folds(y: np.ndarray, wells: pd.Series, classes: list[int]) -> tuple[int, dict[str, int]]:
    """Return the seed and well-to-fold assignment with every class in every held-out fold."""
    best, best_score = None, np.inf
    totals = np.array([(y == c).sum() for c in classes])
    for seed in range(N_SEEDS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        assignment: dict[str, int] = {}
        score = 0.0
        for k, (_, held) in enumerate(cv.split(np.zeros(len(y)), y, wells)):
            counts = np.array([(y[held] == c).sum() for c in classes])
            if (counts == 0).any():
                score = np.inf
                break
            score += float(np.abs(counts / totals - 1 / N_FOLDS).sum())
            assignment.update({w: k for w in wells.iloc[held].unique()})
        if score < best_score:
            best, best_score = (seed, assignment), score
    if best is None:
        raise SystemExit("No fold assignment puts every class in every held-out fold")
    return best


def write_fold(folder: Path, windows_dir: Path, parts: dict[str, np.ndarray], X, mask, y, meta) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for split, idx in parts.items():
        np.savez_compressed(folder / f"{split}.npz", X=X[idx], mask=mask[idx], y=y[idx])
        meta.iloc[idx].reset_index(drop=True).to_csv(folder / f"{split}.csv", index=False)
    shutil.copy(windows_dir / "meta.json", folder / "meta.json")


def xgboost_reference(X, mask, y, folds: list[dict[str, np.ndarray]], labels: list[int]) -> dict:
    per_fold = []
    for k, parts in enumerate(folds):
        tr, va = parts["train"], parts["val"]
        model = baselines.fit(X[tr], mask[tr], y[tr], "shape", labels)
        pred = baselines.predict(model, X[va], mask[va], "shape", labels)
        yv = y[va]
        per_fold.append(
            {
                "fold": k,
                "macro_f1": round(float(f1_score(yv, pred, labels=labels, average="macro", zero_division=0)), 4),
                "event_recall": round(float((pred[yv != 0] != 0).mean()), 4),
                "false_alarm_rate": round(float((pred[yv == 0] != 0).mean()), 4),
            }
        )
        print(f"XGBoost shape, fold {k}: {per_fold[-1]}")
    macro = [f["macro_f1"] for f in per_fold]
    return {"macro_f1_mean": round(float(np.mean(macro)), 4), "macro_f1_std": round(float(np.std(macro)), 4), "folds": per_fold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--xgboost", action="store_true", help="also score the XGBoost shape baseline on the folds")
    args = parser.parse_args()

    labels = [int(c) for c in json.loads((args.windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    X, mask, y, meta = pooled(args.windows_dir)
    seed, assignment = search_folds(y, meta["well"], labels)
    fold_of = meta["well"].map(assignment).to_numpy()

    folds, summary = [], {}
    for k in range(N_FOLDS):
        parts = {"train": np.flatnonzero(fold_of != k), "val": np.flatnonzero(fold_of == k)}
        folds.append(parts)
        write_fold(args.out_dir / f"fold{k}", args.windows_dir, parts, X, mask, y, meta)
        held = meta.iloc[parts["val"]]
        summary[f"fold{k}"] = {
            "train_windows": int(len(parts["train"])),
            "held_out_windows": int(len(parts["val"])),
            "held_out_wells": sorted(held["well"].unique()),
            "held_out_by_class": {
                str(c): {"windows": int((held["label"] == c).sum()), "wells": int(held.loc[held["label"] == c, "well"].nunique())}
                for c in labels
            },
        }
        print(f"fold{k}: {summary[f'fold{k}']['held_out_by_class']}")

    SPLITS_FILE.write_text(
        json.dumps({"seed": seed, "n_folds": N_FOLDS, "note": "train and validation wells of the connector split; test wells excluded", "wells": dict(sorted(assignment.items())), "folds": summary}, indent=1),
        encoding="utf-8",
    )
    print(f"seed {seed}; assignment saved to {SPLITS_FILE}")
    if args.xgboost:
        reference = xgboost_reference(X, mask, y, folds, labels)
        out = SPLITS_FILE.parent.parent / "results" / "folds"
        out.mkdir(parents=True, exist_ok=True)
        (out / "xgboost_shape_folds.json").write_text(json.dumps(reference, indent=1), encoding="utf-8")
        print(f"XGBoost shape over the folds: {reference['macro_f1_mean']:.3f} +/- {reference['macro_f1_std']:.3f}")


if __name__ == "__main__":
    main()
