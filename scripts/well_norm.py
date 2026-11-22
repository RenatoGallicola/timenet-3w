"""Per-well normalization of the window statistics (issue #37).

Usage:
    uv run python scripts/well_norm.py <folds folder> <output json>

The mean and std of a window are absolute levels (bar, degC) that differ from well to well, so a
model can learn the levels of its training wells instead of what an event does. Here each window's
statistics are expressed relative to its own well:

- level:  (window mean - median of the well's window means) / (MAD of the well's window means)
- spread: log10(window std / median of the well's window stds)

The reference of a well uses **all its windows, without labels** (transductive: in deployment the
recent sensor history of a well is available). Each well lies in one split only, so the reference of a
held-out well never uses training wells, and no label of a held-out window is used.

XGBoost (the settings of scripts/baselines.py: sensor dropout copies, class weights, 300 trees) is
trained on the 3 folds by well (scripts/make_folds.py) with four feature sets:

| Kind | Features |
|---|---|
| mean_std | absolute mean and std per sensor (the numbers in the OpenTSLM prompt) |
| mean_std_well | per-well level and spread per sensor |
| shape | scale-free shape features, pooled over pressures and temperatures (the reference) |
| shape_well | shape + per-well level and spread |
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402

KINDS = ("mean_std", "mean_std_well", "shape", "shape_well")
EPS = 1e-6


def window_stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = X.copy()
    X[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(X, 2), np.nanstd(X, 2)


def well_reference(X: np.ndarray, mask: np.ndarray, wells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window, the reference of its well: median mean, MAD of means, median std (each [N, V])."""
    mean, std = window_stats(X, mask)
    med_mean, mad_mean, med_std = (np.full_like(mean, np.nan) for _ in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for w in np.unique(wells):
            rows = wells == w
            m = np.nanmedian(mean[rows], 0)
            med_mean[rows] = m
            mad_mean[rows] = np.nanmedian(np.abs(mean[rows] - m), 0)
            med_std[rows] = np.nanmedian(std[rows], 0)
    return med_mean, mad_mean, med_std


def well_features(X: np.ndarray, mask: np.ndarray, ref: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = window_stats(X, mask)
    med_mean, mad_mean, med_std = ref
    level = (mean - med_mean) / (mad_mean + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.log10((std + EPS) / (med_std + EPS))
    return np.concatenate([np.clip(level, -1e3, 1e3), spread], 1)


def features(X: np.ndarray, mask: np.ndarray, kind: str, ref) -> np.ndarray:
    if kind == "mean_std":
        return baselines.features(X, mask, "mean_std")
    if kind == "mean_std_well":
        return well_features(X, mask, ref)
    if kind == "shape":
        return baselines.features(X, mask, "shape")
    return np.concatenate([baselines.features(X, mask, "shape"), well_features(X, mask, ref)], 1)


def fit(X, mask, y, kind, labels, ref) -> XGBClassifier:
    rng = np.random.default_rng(0)
    masks = [mask] + [baselines.sensor_dropout(mask, rng) for _ in range(baselines.N_DROPOUT_COPIES)]
    F = np.concatenate([features(X, m, kind, ref) for m in masks])
    yy = np.tile(np.array([labels.index(c) for c in y]), len(masks))
    model = XGBClassifier(
        n_estimators=baselines.N_TREES, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(labels), tree_method="hist", random_state=0,
    )
    model.fit(F, yy, sample_weight=baselines.class_weights(yy))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folds_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    folds = sorted(p for p in args.folds_dir.iterdir() if p.is_dir() and p.name.startswith("fold"))
    labels = [int(c) for c in json.loads((folds[0] / "meta.json").read_text(encoding="utf-8"))["classes"]]
    results = {k: {"folds": []} for k in KINDS}
    for k, folder in enumerate(folds):
        data = {s: baselines.load(folder, s) for s in ("train", "val")}
        refs = {s: well_reference(d[0], d[1], d[3]["well"].to_numpy()) for s, d in data.items()}
        Xt, mt, yt, _ = data["train"]
        Xv, mv, yv, _ = data["val"]
        for kind in KINDS:
            model = fit(Xt, mt, yt, kind, labels, refs["train"])
            pred = np.array(labels)[model.predict(features(Xv, mv, kind, refs["val"]))]
            r = {
                "fold": k,
                "macro_f1": round(float(f1_score(yv, pred, labels=labels, average="macro", zero_division=0)), 4),
                "event_recall": round(float((pred[yv != 0] != 0).mean()), 4),
                "false_alarm_rate": round(float((pred[yv == 0] != 0).mean()), 4),
                "per_class_f1": {
                    str(c): round(float(s), 4)
                    for c, s in zip(labels, f1_score(yv, pred, labels=labels, average=None, zero_division=0))
                },
            }
            results[kind]["folds"].append(r)
            print(f"fold {k} {kind:14s} macro-F1 {r['macro_f1']:.3f}  recall {r['event_recall']:.2f}  "
                  f"false alarms {r['false_alarm_rate']:.2f}", flush=True)
    for kind, r in results.items():
        for key in ("macro_f1", "event_recall", "false_alarm_rate"):
            vals = [f[key] for f in r["folds"]]
            r[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        r["macro_f1_std"] = round(float(np.std([f["macro_f1"] for f in r["folds"]])), 4)
        print(f"{kind:14s} macro-F1 {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f}  recall "
              f"{r['event_recall_mean']:.2f}  false alarms {r['false_alarm_rate_mean']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
