"""Export the data of the static demo (issue #10) into demo/data.js.

Usage:
    uv run python scripts/export_demo.py <windows folder> [--per-class 4]

Picks test windows for the explorer, a fixed number per class from different instances (seeded),
mixing windows that OpenTSLM got right and wrong, and joins them with the saved predictions of
OpenTSLM run 1 and the XGBoost shape baseline. It also copies the result summaries the page shows
(model comparison, split leakage, lead time, perturbation check). Everything comes from files in
results/; nothing is computed on the test split here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 0
UNITS = {"P-TPT": "bar", "T-TPT": "°C", "P-ANULAR": "bar", "P-MON-CKP": "bar", "P-JUS-CKGL": "bar", "T-JUS-CKP": "°C"}
DESCRIPTIONS = {
    "P-TPT": "Pressure at the subsea Xmas tree (TPT)",
    "T-TPT": "Temperature at the subsea Xmas tree (TPT)",
    "P-ANULAR": "Pressure in the well annulus",
    "P-MON-CKP": "Pressure upstream of the production choke",
    "P-JUS-CKGL": "Pressure downstream of the gas lift choke",
    "T-JUS-CKP": "Temperature downstream of the production choke",
}


def predictions(name: str) -> dict[str, str]:
    frame = pd.read_csv(RESULTS / "predictions" / f"{name}.csv", dtype=str)
    return dict(zip(frame["window_id"], frame["prediction"]))


def to_class(value: str, names: dict[int, str]) -> int | None:
    if value is None:
        return None
    if value.lstrip("-").isdigit():
        return int(value)
    lookup = {n.lower(): c for c, n in names.items()}
    return lookup.get(value.strip().lower())


def series(x: np.ndarray, mask: np.ndarray, variables: list[str]) -> list[dict]:
    out = []
    for i, var in enumerate(variables):
        if not mask[i]:
            continue
        scale = 1e-5 if UNITS[var] == "bar" else 1.0
        values = [None if math.isnan(v) else round(float(v) * scale, 3) for v in x[i]]
        out.append({"name": var, "description": DESCRIPTIONS[var], "unit": UNITS[var], "values": values})
    return out


def pick_examples(meta: pd.DataFrame, X, mask, names, preds_tslm, preds_xgb, per_class: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    examples = []
    for code in sorted(names):
        rows = meta.index[meta["label"] == code].to_numpy().copy()
        rng.shuffle(rows)
        chosen, seen = [], set()
        # Alternate right and wrong OpenTSLM answers when both exist, one window per instance.
        right = [r for r in rows if to_class(preds_tslm.get(meta.at[r, "window_id"]), names) == code]
        wrong = [r for r in rows if r not in set(right)]
        for pool in (right, wrong) * per_class:
            for r in pool:
                if meta.at[r, "instance"] not in seen:
                    chosen.append(r)
                    seen.add(meta.at[r, "instance"])
                    break
            if len(chosen) == per_class:
                break
        for r in chosen:
            wid = meta.at[r, "window_id"]
            examples.append(
                {
                    "window_id": wid,
                    "instance": meta.at[r, "instance"],
                    "well": meta.at[r, "well"],
                    "label": int(code),
                    "phase": meta.at[r, "phase"],
                    "start_min": int(meta.at[r, "start_s"]) // 60,
                    "opentslm": to_class(preds_tslm.get(wid), names),
                    "xgboost": to_class(preds_xgb.get(wid), names),
                    "series": series(X[r], mask[r], list(UNITS)),
                }
            )
    return examples


def model_summary(report: dict) -> dict:
    m = report["all_windows"]
    return {
        "macro_f1": m["macro_f1_6_classes"],
        "balanced_accuracy": m["balanced_accuracy"],
        "event_recall": m["event_detection"]["event_recall"],
        "false_alarm_rate": m["event_detection"]["false_alarm_rate"],
        "per_class": {c: {"f1": v["f1"], "wells": v["wells"], "instances": v["instances"]} for c, v in m["per_class"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()

    meta_json = json.loads((args.windows_dir / "meta.json").read_text(encoding="utf-8"))
    names = {int(c): n for c, n in meta_json["classes"].items()}
    data = np.load(args.windows_dir / "test.npz")
    meta = pd.read_csv(args.windows_dir / "test.csv")
    examples = pick_examples(
        meta, data["X"], data["mask"], names,
        predictions("opentslm_sp_run1_test"), predictions("xgboost_shape_test"), args.per_class,
    )

    comparison = json.loads((RESULTS / "eval" / "opentslm_sp_run1_vs_xgboost_shape_test.json").read_text(encoding="utf-8"))
    baselines = json.loads((RESULTS / "baselines.json").read_text(encoding="utf-8"))
    lead_time = json.loads((RESULTS / "lead_time" / "lead_time_val.json").read_text(encoding="utf-8"))
    payload = {
        "classes": {str(c): n for c, n in names.items()},
        "window": {"minutes": meta_json["window_s"] // 60, "points": meta_json["n_points"], "step_s": meta_json["sample_period_s"]},
        "examples": examples,
        "results": {
            "test": {
                "opentslm_run1": model_summary(comparison["report"]),
                "xgboost_shape": model_summary(comparison["compared_with"]["report"]),
                "xgboost_mean_std_macro_f1": baselines["mean_std"]["test"]["all"]["macro_f1"],
                "paired": comparison["paired_comparison"]["paired_bootstrap_over_instances"],
            },
            "split_leakage": json.loads((RESULTS / "split_leakage" / "split_leakage.json").read_text(encoding="utf-8"))["schemes"],
            "lead_time_val": lead_time["class_average"],
            "lead_time_false_alarm_rate": lead_time["false_alarm_rate_on_normal_windows"],
            "perturbation_xgboost_val": json.loads((RESULTS / "perturbation" / "xgboost_val.json").read_text(encoding="utf-8")),
        },
    }
    out = ROOT / "demo" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DEMO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{len(examples)} examples, {out.stat().st_size / 1024:.0f} KB written to {out}")


if __name__ == "__main__":
    main()
