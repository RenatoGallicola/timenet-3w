"""Lead-time analysis: how early is an event detected around its onset? (issue #15)

Usage:
    uv run python scripts/lead_time.py <TimeNet registry> <windows folder> <output folder> [--split val]

The training windows need at least 50% event seconds, so the onset of an event is never evaluated.
This script slides 30-minute windows across the onset instead. For each real event instance of the
chosen split, the onset is the first second labelled with the event (transient or steady). A window
whose end lies `d` minutes after the onset is scored for d = -30 ... +120 minutes, every 5 minutes;
`d <= 0` means the window ends before the event starts. The model is the XGBoost `shape` baseline,
retrained on the training windows exactly as in scripts/baselines.py.

For each offset the output gives the share of instances flagged as an event (any event class) and the
share given the right event class, next to the false-alarm rate on the normal windows of the same
split. Use the validation split while developing; the test split once, at the end (test wells are
frozen, see docs/DECISIONS.md).

Outputs: `lead_time_<split>.csv` (one row per instance and offset), `lead_time_<split>.json`
(aggregates), `lead_time_<split>.png` (chart).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402
import make_windows as mw  # noqa: E402

OFFSETS_MIN = list(range(-30, 121, 5))
KIND = "shape"


def onset_windows(registry: Path, wells: set[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Cut one window per (event instance, offset) for the real event instances of the given wells."""
    xs, masks, rows = [], [], []
    for instance, event, values, labels in mw.iter_timenet(registry):
        well = instance.split("_")[0]
        if well not in wells or event == 0 or event not in mw.CLASSES:
            continue
        in_event = np.flatnonzero((labels == event) | (labels == 100 + event))
        if len(in_event) == 0:
            continue
        onset = int(in_event[0])
        for d in OFFSETS_MIN:
            end = onset + d * 60
            start = end - mw.WINDOW_S
            if start < 0 or end > labels.shape[0]:
                continue
            x = mw.resample(values[:, start:end])
            mask = (~np.isnan(x)).mean(axis=1) >= mw.VAR_PRESENT_MIN_FRAC
            if mask.sum() < mw.MIN_PRESENT_VARS:
                continue
            xs.append(x)
            masks.append(mask)
            rows.append({"instance": instance, "well": well, "event": event, "offset_min": d})
    return np.stack(xs), np.stack(masks), pd.DataFrame(rows)


def _rates(group: pd.DataFrame) -> dict:
    return {
        "instances": int(group["instance"].nunique()),
        "wells": int(group["well"].nunique()),
        "detected": round(float((group["pred"] != 0).mean()), 4),
        "right_class": round(float((group["pred"] == group["event"]).mean()), 4),
    }


def aggregate(scored: pd.DataFrame, false_alarm: float) -> dict:
    """Detection rates at each offset, per event class and averaged over the classes.

    The class average gives each event type the same weight, like the macro-F1, so a class with
    many instances from a single well does not dominate the curve.
    """
    per_event = {
        str(int(event)): [{"offset_min": int(d), **_rates(g)} for d, g in group.groupby("offset_min")]
        for event, group in scored.groupby("event")
    }
    class_average = []
    for d, group in scored.groupby("offset_min"):
        rates = [_rates(g) for _, g in group.groupby("event")]
        class_average.append(
            {
                "offset_min": int(d),
                "classes": len(rates),
                "detected": round(float(np.mean([r["detected"] for r in rates])), 4),
                "right_class": round(float(np.mean([r["right_class"] for r in rates])), 4),
            }
        )
    return {
        "false_alarm_rate_on_normal_windows": round(false_alarm, 4),
        "class_average": class_average,
        "per_event": per_event,
        "note": "d <= 0: the window ends before the onset, so ideally the rate equals the false-alarm rate.",
    }


def plot(summary: dict, split: str, path: Path) -> None:
    """Detection and right-class rates against minutes since onset, with the false-alarm rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(summary["class_average"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rows["offset_min"], rows["detected"], marker="o", linewidth=2, label="flagged as an event (class average)")
    ax.plot(rows["offset_min"], rows["right_class"], marker="o", linewidth=2, label="right event class (class average)")
    for event, curve in summary["per_event"].items():
        c = pd.DataFrame(curve)
        ax.plot(c["offset_min"], c["detected"], linewidth=0.8, alpha=0.4, label=f"flagged, class {event}")
    ax.axhline(summary["false_alarm_rate_on_normal_windows"], color="grey", linestyle="--", label="false alarms on normal windows")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("minutes from event onset to the end of the 30-minute window")
    ax.set_ylabel("share of event instances")
    ax.set_ylim(0, 1)
    ax.set_title(f"XGBoost shape baseline: detection around event onset ({split} wells)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry", type=Path)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args()

    labels = [int(c) for c in json.loads((args.windows_dir / "meta.json").read_text(encoding="utf-8"))["classes"]]
    X, mask, y, _ = baselines.load(args.windows_dir, "train")
    model = baselines.fit(X, mask, y, KIND, labels)

    Xs, masks, ys, meta = baselines.load(args.windows_dir, args.split)
    normal_pred = baselines.predict(model, Xs[ys == 0], masks[ys == 0], KIND, labels)
    false_alarm = float((normal_pred != 0).mean())

    wells = set(meta["well"])
    Xo, mo, rows = onset_windows(args.registry, wells)
    proba = model.predict_proba(baselines.features(Xo, mo, KIND))
    rows["pred"] = np.array(labels)[proba.argmax(1)]
    rows["p_event"] = 1 - proba[:, labels.index(0)]

    summary = {"split": args.split, "model": f"xgboost_{KIND}", **aggregate(rows, false_alarm)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / f"lead_time_{args.split}.csv", index=False)
    (args.out_dir / f"lead_time_{args.split}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(summary, args.split, args.out_dir / f"lead_time_{args.split}.png")
    print(f"false alarms on normal windows: {false_alarm:.3f}")
    for r in summary["class_average"]:
        if r["offset_min"] % 15 == 0:
            print(f"{r['offset_min']:+4d} min: detected {r['detected']:.2f}  right class {r['right_class']:.2f}  (class average over {r['classes']})")
    for event, curve in summary["per_event"].items():
        first = [c for c in curve if 1 <= c["offset_min"] <= 30]
        before = [c for c in curve if c["offset_min"] <= 0]
        if first and before:
            print(
                f"class {event}: before onset {np.mean([c['detected'] for c in before]):.2f}, first 30 min "
                f"{np.mean([c['detected'] for c in first]):.2f} ({first[0]['instances']} instances, {first[0]['wells']} wells)"
            )

if __name__ == "__main__":
    main()
