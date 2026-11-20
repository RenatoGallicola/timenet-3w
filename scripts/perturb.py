"""Perturbation tests: does a model read the signal or only the summary statistics? (issue #3)

Usage:
    uv run python scripts/perturb.py make <windows folder> <output folder> [--split val]
    uv run python scripts/perturb.py xgboost <windows folder> <output folder> [--split val]

`make` writes one window folder per condition under <output folder>/<condition>/, in the format of
docs/DATA_CONTRACT.md, plus the statistics a TSLM prompt must show: `stats_mean` and `stats_std`
(float32 [N, 6], NaN = do not show). Any model can then be run on every condition and scored with
scripts/evaluate.py. OpenTSLM writes each series' mean and standard deviation into its prompt and
feeds the series normalized, so the signal and the statistics are two separate channels:

| Condition | Series | Statistics in the prompt | Question |
|---|---|---|---|
| original | own | own | reference |
| time_shuffle | own, time steps shuffled (one permutation for all sensors) | own | does the model use the temporal order? |
| noise_signal | Gaussian noise with the window's own mean and std | own | how far do the statistics alone go? |
| signal_from_other | series of a window of another class | own | conflict: signal vs statistics |
| stats_from_other | own | statistics of a window of another class | conflict: statistics vs signal |
| no_stats | own | none | how far does the series alone go? |

In the two conflict conditions the CSV holds `donor_label`, the class of the window whose part was
borrowed, so one can measure whether the answer follows the signal or the statistics.

`xgboost` runs the two baselines on every condition (they see only the series, so the statistics
conditions do not change them) as a check of the tool. Use the validation split while developing;
the test split once at the end (test wells are frozen, docs/DECISIONS.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402

CONDITIONS = ("original", "time_shuffle", "noise_signal", "signal_from_other", "stats_from_other", "no_stats")
SEED = 0


def stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window, per-sensor mean and std over the present points; NaN for absent sensors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, std = np.nanmean(X, axis=2), np.nanstd(X, axis=2)
    mean[~mask], std[~mask] = np.nan, np.nan
    return mean.astype(np.float32), std.astype(np.float32)


def donors(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each window, the index of a random window of another class."""
    out = np.empty(len(y), dtype=np.int64)
    for i, label in enumerate(y):
        candidates = np.flatnonzero(y != label)
        out[i] = rng.choice(candidates)
    return out


def make(windows_dir: Path, out_dir: Path, split: str) -> None:
    data = np.load(windows_dir / f"{split}.npz")
    X, mask, y = data["X"], data["mask"], data["y"]
    meta = pd.read_csv(windows_dir / f"{split}.csv")
    mean, std = stats(X, mask)
    rng = np.random.default_rng(SEED)
    donor = donors(y, rng)

    for condition in CONDITIONS:
        Xc, mc, mean_c, std_c = X.copy(), mask.copy(), mean.copy(), std.copy()
        rows = meta.copy()
        if condition == "time_shuffle":
            for i in range(len(Xc)):
                Xc[i] = Xc[i][:, rng.permutation(Xc.shape[2])]
        elif condition == "noise_signal":
            noise = rng.standard_normal(Xc.shape).astype(np.float32)
            Xc = np.where(np.isnan(X), np.nan, mean[..., None] + std[..., None] * noise).astype(np.float32)
        elif condition == "signal_from_other":
            Xc, mc = X[donor].copy(), mask[donor].copy()
            rows["donor_label"] = y[donor]
        elif condition == "stats_from_other":
            mean_c, std_c = mean[donor].copy(), std[donor].copy()
            # Show a statistic only for a sensor the window itself has.
            mean_c[~mask], std_c[~mask] = np.nan, np.nan
            rows["donor_label"] = y[donor]
        elif condition == "no_stats":
            mean_c[:], std_c[:] = np.nan, np.nan

        folder = out_dir / condition
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{split}.npz", X=Xc, mask=mc, y=y, stats_mean=mean_c, stats_std=std_c)
        rows.to_csv(folder / f"{split}.csv", index=False)
        shutil.copy(windows_dir / "meta.json", folder / "meta.json")
        print(f"{condition}: {len(y)} windows")


def follow_rates(pred: np.ndarray, y: np.ndarray, donor_label: np.ndarray) -> dict:
    return {
        "follows_original_label": round(float((pred == y).mean()), 4),
        "follows_donor_label": round(float((pred == donor_label).mean()), 4),
    }


def xgboost(windows_dir: Path, out_dir: Path, split: str) -> None:
    labels = [int(c) for c in json.loads((windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    X, mask, y, _ = baselines.load(windows_dir, "train")
    results = {}
    for kind in ("shape", "mean_std"):
        model = baselines.fit(X, mask, y, kind, labels)
        results[kind] = {}
        for condition in CONDITIONS:
            folder = out_dir / condition
            data = np.load(folder / f"{split}.npz")
            rows = pd.read_csv(folder / f"{split}.csv")
            pred = baselines.predict(model, data["X"], data["mask"], kind, labels)
            r = {"macro_f1": round(float(f1_score(data["y"], pred, labels=labels, average="macro", zero_division=0)), 4)}
            if "donor_label" in rows:
                r.update(follow_rates(pred, data["y"], rows["donor_label"].to_numpy()))
            results[kind][condition] = r
            print(f"{kind:9s} {condition:18s} {r}")
    (out_dir / f"xgboost_{split}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("make", "xgboost"))
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args()
    if args.command == "make":
        make(args.windows_dir, args.out_dir, args.split)
    else:
        xgboost(args.windows_dir, args.out_dir, args.split)


if __name__ == "__main__":
    main()
