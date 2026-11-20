"""Score model predictions on the 3W windows, the same way for every model.

Usage:
    uv run python scripts/evaluate.py <windows folder> <predictions.csv> [--split test] [--out report.json]
    uv run python scripts/evaluate.py <windows folder> <A.csv> --compare <B.csv> [--split test]

A predictions file is a CSV with one row per window: `window_id` and `prediction`. The prediction
is a 3W class code (0, 2, 4, 7, 8, 9), a class name ("Flow instability"), or a model's raw answer
("... Answer: Flow instability"). An answer that maps to no class, and a window with no prediction,
count as **errors**: they are scored as an "invalid" class and their share is reported.

Reported, following issue #2 and docs/DECISIONS.md:
- macro-F1 over the same 6 classes whatever the split holds, balanced accuracy and accuracy;
- per-class precision, recall and F1, with the number of windows, instances and wells behind each
  class: in the connector test split some classes come from a single well;
- event-vs-normal detection: share of event windows flagged as any event, false-alarm rate on
  normal windows;
- the same metrics on transient-phase windows (not an early-warning measure: see issue #15);
- with `--compare`, a paired comparison of two models on the same windows: a paired bootstrap over
  instances of the macro-F1 difference, and an exact McNemar test on window correctness.

There is no bootstrap over wells: with one well behind some classes it would be degenerate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

CLASSES = [0, 2, 4, 7, 8, 9]
INVALID = -1
# A split may lack some predicted classes; the fixed 6-class denominator is intended.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
N_BOOTSTRAP = 2000


def class_names(windows_dir: Path) -> dict[int, str]:
    """Return the class names stored in meta.json by make_windows.py."""
    meta = json.loads((windows_dir / "meta.json").read_text(encoding="utf-8"))
    return {int(code): name for code, name in meta["classes"].items()}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_prediction(value: object, names: dict[int, str]) -> int:
    """Map one prediction to a class code, or INVALID if it names no class.

    Accepts a class code, a class name, or a raw answer; in a raw answer the text after the last
    "Answer:" is used when present. Matching ignores case and punctuation.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return INVALID
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", text):
        code = int(float(text))
        return code if code in names else INVALID
    answer = re.split(r"answer\s*:", text, flags=re.IGNORECASE)[-1]
    target = _normalize(answer)
    by_name = {_normalize(name): code for code, name in names.items()}
    if target in by_name:
        return by_name[target]
    found = [code for name, code in by_name.items() if re.search(rf"\b{re.escape(name)}\b", target)]
    return found[0] if len(found) == 1 else INVALID


def load(windows_dir: Path, predictions: Path, split: str, names: dict[int, str]) -> pd.DataFrame:
    """Join the windows of a split with a predictions file; missing predictions are invalid."""
    meta = pd.read_csv(windows_dir / f"{split}.csv")
    pred = pd.read_csv(predictions, dtype={"window_id": str})
    if pred["window_id"].duplicated().any():
        raise SystemExit(f"{predictions}: duplicated window_id")
    unknown = set(pred["window_id"]) - set(meta["window_id"])
    if unknown:
        raise SystemExit(f"{predictions}: {len(unknown)} window_id not in the {split} split, e.g. {sorted(unknown)[:3]}")
    data = meta.merge(pred[["window_id", "prediction"]], on="window_id", how="left")
    data["pred"] = [parse_prediction(v, names) for v in data["prediction"]]
    return data


def metrics(data: pd.DataFrame, names: dict[int, str]) -> dict:
    """Compute the report of one model on a set of windows."""
    y, pred = data["label"].to_numpy(), data["pred"].to_numpy()
    if len(y) == 0:
        return {"windows": 0}
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    per_class = {}
    for i, code in enumerate(CLASSES):
        rows = data[data["label"] == code]
        per_class[str(code)] = {
            "name": names[code],
            "windows": int(len(rows)),
            "instances": int(rows["instance"].nunique()),
            "wells": int(rows["well"].nunique()),
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
        }
    events, normal = y != 0, y == 0
    return {
        "windows": int(len(y)),
        "instances": int(data["instance"].nunique()),
        "wells": int(data["well"].nunique()),
        "macro_f1_6_classes": round(float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "accuracy": round(float((y == pred).mean()), 4),
        "invalid_share": round(float((pred == INVALID).mean()), 4),
        "event_detection": {
            "event_recall": round(float((pred[events] != 0).mean()), 4) if events.any() else None,
            "false_alarm_rate": round(float((pred[normal] != 0).mean()), 4) if normal.any() else None,
        },
        "per_class": per_class,
    }


def report(data: pd.DataFrame, names: dict[int, str]) -> dict:
    """Metrics on all windows and on transient-phase windows."""
    return {
        "all_windows": metrics(data, names),
        "transient_phase_windows": metrics(data[data["phase"] == "transient"], names),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": round(p, 6),
        "note": "Treats windows as independent; overlapping windows of one instance are not, so read the p-value as optimistic.",
    }


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> dict:
    """Bootstrap over instances of macro-F1(A) - macro-F1(B) on the same windows."""
    y = a["label"].to_numpy()
    pa, pb = a["pred"].to_numpy(), b["pred"].to_numpy()
    groups = a.groupby("instance").indices
    keys = list(groups)
    rng = np.random.default_rng(seed)

    def f1(idx: np.ndarray, pred: np.ndarray) -> float:
        return f1_score(y[idx], pred[idx], labels=CLASSES, average="macro", zero_division=0)

    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        diffs.append(f1(idx, pa) - f1(idx, pb))
    diffs = np.array(diffs)
    full = np.arange(len(y))
    return {
        "macro_f1_difference": round(float(f1(full, pa) - f1(full, pb)), 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "share_of_resamples_a_better": round(float((diffs > 0).mean()), 4),
        "note": "Resamples instances; windows of one well stay correlated, so read the interval as optimistic.",
    }


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired comparison of two models scored on the same windows."""
    correct_a = (a["pred"] == a["label"]).to_numpy()
    correct_b = (b["pred"] == b["label"]).to_numpy()
    return {"paired_bootstrap_over_instances": paired_bootstrap(a, b), "mcnemar_exact": mcnemar_exact(correct_a, correct_b)}


def summary_line(name: str, r: dict) -> str:
    m = r["all_windows"]
    per = " ".join(f"{c}:{v['f1']:.2f}({v['wells']}w)" for c, v in m["per_class"].items())
    det = m["event_detection"]
    return (
        f"{name}: macro-F1 {m['macro_f1_6_classes']:.3f}  bal.acc {m['balanced_accuracy']:.3f}  "
        f"invalid {m['invalid_share']:.1%}  event recall {det['event_recall']:.2f}  false alarms "
        f"{det['false_alarm_rate']:.2f}  | per class F1 (wells) {per}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path, help="a second predictions file for a paired comparison")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    names = class_names(args.windows_dir)
    a = load(args.windows_dir, args.predictions, args.split, names)
    result = {"split": args.split, "predictions": str(args.predictions), "report": report(a, names)}
    print(summary_line(args.predictions.stem, result["report"]))
    if args.compare:
        b = load(args.windows_dir, args.compare, args.split, names)
        result["compared_with"] = {"predictions": str(args.compare), "report": report(b, names)}
        result["paired_comparison"] = compare(a, b)
        print(summary_line(args.compare.stem, result["compared_with"]["report"]))
        print("paired:", json.dumps(result["paired_comparison"]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
