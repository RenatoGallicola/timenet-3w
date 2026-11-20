"""Classical baselines on the 3W windows (see docs/DATA_CONTRACT.md).

Usage:
    uv run python scripts/baselines.py <windows folder> <output json>

Two XGBoost classifiers:
- `shape`: scale-free features of each signal (trend, change, roughness, autocorrelation, ...),
  pooled over the pressure and the temperature sensors that are present. Absolute levels are left
  out: they differ from well to well and do not transfer to unseen wells;
- `mean_std`: only the mean and standard deviation of each signal, the same numbers OpenTSLM
  writes into its prompt.

Each well has its own set of sensors, so a model can learn which sensors are present instead of
what the signals do. Training therefore also sees copies of each window with random present sensors
hidden (keeping at least 2). Missing values stay NaN: XGBoost handles them natively.

Reported: the split the windows were cut with (fit on train, report val and test; the team uses the
connector's split, splits/wells_connector.json), and a 5-fold cross-validation grouped by well over
all windows, which is less sensitive to which wells end up in test.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

PRESSURES = [0, 2, 3, 4]  # P-TPT, P-ANULAR, P-MON-CKP, P-JUS-CKGL
TEMPERATURES = [1, 5]  # T-TPT, T-JUS-CKP
N_DROPOUT_COPIES = 4
N_TREES = 300
N_FOLDS = 5


def load(folder: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    data = np.load(folder / f"{split}.npz")
    return data["X"], data["mask"], data["y"], pd.read_csv(folder / f"{split}.csv")


def mean_crossings(Z: np.ndarray) -> np.ndarray:
    """Share of consecutive point pairs, both present, where the standardized signal changes sign.

    A missing point is skipped, not read as 0, so gaps add no false crossings.
    """
    sign = np.sign(Z)
    valid = ~np.isnan(Z[..., 1:]) & ~np.isnan(Z[..., :-1])
    crossings = (sign[..., 1:] * sign[..., :-1] < 0) & valid
    with np.errstate(invalid="ignore", divide="ignore"):
        return crossings.sum(2) / valid.sum(2)


def shape_features(X: np.ndarray) -> np.ndarray:
    """X [N, V, T] -> [N, F, V]: scale-free features of each signal."""
    mean = np.nanmean(X, 2, keepdims=True)
    std = np.nanstd(X, 2, keepdims=True)
    Z = (X - mean) / (std + 1e-9)
    T = X.shape[2]
    t = np.arange(T) - (T - 1) / 2
    q = T // 4

    def autocorr(k: int) -> np.ndarray:
        return np.nanmean(Z[..., k:] * Z[..., :-k], 2)

    feats = [
        np.nansum(Z * t, 2) / np.sum(t**2) * T,  # trend
        np.nanmean(Z[..., -q:], 2) - np.nanmean(Z[..., :q], 2),  # change, first vs last quarter
        np.nanstd(np.diff(Z, axis=2), 2),  # roughness
        autocorr(1), autocorr(6), autocorr(18),  # lags of 10 s, 1 min, 3 min
        mean_crossings(Z),
        np.nanmean(Z**3, 2), np.nanmean(Z**4, 2), np.nanmax(np.abs(Z), 2),  # skew, kurtosis, peak
        np.log10(std[..., 0] / (np.abs(mean[..., 0]) + 1e-9) + 1e-12),  # relative spread
        (np.nanmean(X[..., -q:], 2) - np.nanmean(X[..., :q], 2)) / (np.abs(mean[..., 0]) + 1e-9),  # relative change
        np.log10(np.nanstd(X[..., T // 2 :], 2) / (np.nanstd(X[..., : T // 2], 2) + 1e-9) + 1e-12),  # spread growth
    ]
    return np.stack(feats, 1)


def features(X: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    """X [N, V, T], mask [N, V] -> [N, F]. Variables outside the mask count as missing."""
    X = X.copy()
    X[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if kind == "mean_std":
            return np.concatenate([np.nanmean(X, axis=2), np.nanstd(X, axis=2)], axis=1)
        f = shape_features(X)
        pooled = []
        for group in (PRESSURES, TEMPERATURES):
            sub = f[..., group]
            pooled += [np.nanmean(sub, 2), np.nanmax(sub, 2), np.nanmin(sub, 2)]
        return np.concatenate(pooled, 1)


def sensor_dropout(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hide a random subset of the present sensors of each window, keeping at least 2."""
    m = mask.copy()
    for i in range(len(m)):
        present = np.flatnonzero(m[i])
        if len(present) > 2:
            keep = rng.integers(2, len(present) + 1)
            m[i, rng.choice(present, len(present) - keep, replace=False)] = False
    return m


def class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[c] for c in y])


def metrics(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict:
    if len(y) == 0:
        return {"n": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "n": int(len(y)),
            "macro_f1": round(float(f1_score(y, pred, labels=sorted(set(y)), average="macro", zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "accuracy": round(float((y == pred).mean()), 4),
            "per_class_f1": {
                str(c): round(float(s), 4)
                for c, s in zip(labels, f1_score(y, pred, labels=labels, average=None, zero_division=0))
                if (y == c).any()
            },
            "confusion": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
        }


def fit(X: np.ndarray, mask: np.ndarray, y: np.ndarray, kind: str, labels: list[int]) -> XGBClassifier:
    rng = np.random.default_rng(0)
    masks = [mask] + [sensor_dropout(mask, rng) for _ in range(N_DROPOUT_COPIES)]
    F = np.concatenate([features(X, m, kind) for m in masks])
    yy = np.tile(np.array([labels.index(c) for c in y]), len(masks))
    model = XGBClassifier(
        n_estimators=N_TREES, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(labels), tree_method="hist", random_state=0,
    )
    model.fit(F, yy, sample_weight=class_weights(yy))
    return model


def predict(model: XGBClassifier, X: np.ndarray, mask: np.ndarray, kind: str, labels: list[int]) -> np.ndarray:
    return np.array(labels)[model.predict(features(X, mask, kind))]


def report(y: np.ndarray, pred: np.ndarray, phase: np.ndarray, labels: list[int]) -> dict:
    """Macro-F1 is averaged over the classes present in `y`."""
    transient = phase == "transient"
    events = y != 0
    return {
        "all": metrics(y, pred, labels),
        "transient_only": metrics(y[transient], pred[transient], labels),
        "event_detection": {
            "description": "event vs normal, any event class",
            "recall_events": round(float((pred[events] != 0).mean()), 4) if events.any() else None,
            "false_alarm_rate": round(float((pred[~events] != 0).mean()), 4) if (~events).any() else None,
        },
    }


def run(folder: Path, kind: str, labels: list[int]) -> dict:
    data = {s: load(folder, s) for s in ("train", "val", "test")}
    X, mask, y, _ = data["train"]
    model = fit(X, mask, y, kind, labels)
    out = {}
    for split in ("val", "test"):
        X, mask, y, meta = data[split]
        out[split] = report(y, predict(model, X, mask, kind, labels), meta["phase"].to_numpy(), labels)

    X = np.concatenate([d[0] for d in data.values()])
    mask = np.concatenate([d[1] for d in data.values()])
    y = np.concatenate([d[2] for d in data.values()])
    meta = pd.concat([d[3] for d in data.values()], ignore_index=True)
    folds = []
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(cv.split(X, y, meta["well"])):
        model = fit(X[tr], mask[tr], y[tr], kind, labels)
        r = report(y[te], predict(model, X[te], mask[te], kind, labels), meta["phase"].to_numpy()[te], labels)
        folds.append({"fold": fold, "test_wells": sorted(meta["well"].iloc[te].unique()), **r})
    f1_all = np.array([f["all"]["macro_f1"] for f in folds])
    f1_tr = np.array([f["transient_only"]["macro_f1"] for f in folds])
    out["cv_by_well"] = {
        "macro_f1_mean": round(float(f1_all.mean()), 4),
        "macro_f1_std": round(float(f1_all.std()), 4),
        "transient_macro_f1_mean": round(float(f1_tr.mean()), 4),
        "transient_macro_f1_std": round(float(f1_tr.std()), 4),
        "folds": folds,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_json", type=Path)
    args = parser.parse_args()
    labels = [int(c) for c in json.loads((args.windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    results = {kind: run(args.windows_dir, kind, labels) for kind in ("shape", "mean_std")}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for kind, r in results.items():
        t, cv = r["test"], r["cv_by_well"]
        print(
            f"{kind:9s} test macro-F1 {t['all']['macro_f1']:.3f} (transient {t['transient_only']['macro_f1']:.3f})  "
            f"val {r['val']['all']['macro_f1']:.3f}  "
            f"CV by well {cv['macro_f1_mean']:.3f} +/- {cv['macro_f1_std']:.3f} "
            f"(transient {cv['transient_macro_f1_mean']:.3f})  per-class test {t['all']['per_class_f1']}"
        )


if __name__ == "__main__":
    main()
