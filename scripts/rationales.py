"""Rationales computed from the window data, and an automatic check of every number they cite (issue #9).

A rationale describes what the shown sensors do in a 30-minute window, in short template sentences
that never mention the class:

    P-TPT drops by 25.3 bar within one minute around minute 12 (from 110.2 to 84.9 bar).
    P-MON-CKP rises by 3.21 bar (from 45.10 to 48.31 bar).
    T-JUS-CKP oscillates with 7 swings of about ±0.84 °C.
    The other sensors are stable.

It is built only from the sensors the model is shown (after sensor dropout), so it never cites a
sensor the model cannot see. Training on "<rationale> Answer: <state>" teaches the model to describe
the signal before it names the state; `check_rationale` then parses a rationale written by the model
and recomputes every claim from the window, so the explanations can be scored for faithfulness:

- a claim about a sensor that is not shown is a hallucination;
- a trend or step claim is right when its direction matches the data and its numbers fall within
  the tolerance below; an oscillation claim when the sensor oscillates by the same rule as here.

Values are in display units (bar, °C), as in the prompt. Only numpy is needed, so the module runs
locally and in the Modal image.

    uv run python scripts/rationales.py ..\\data\\windows --split train --show 8
    uv run python scripts/rationales.py ..\\data\\windows --split val --check-file <predictions.csv>
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np

SENSORS = ["P-TPT", "T-TPT", "P-ANULAR", "P-MON-CKP", "P-JUS-CKGL", "T-JUS-CKP"]
UNITS = {"P-TPT": "bar", "T-TPT": "°C", "P-ANULAR": "bar", "P-MON-CKP": "bar", "P-JUS-CKGL": "bar",
         "T-JUS-CKP": "°C"}
PA_PER_BAR = 1e5
EDGE = 30          # points in the first and last 5 minutes, whose means are "from" and "to"
STEP_LAG = 6       # points in one minute
POINTS_PER_MIN = 6

# A sensor change is cited when it is large against the sensor's typical noise. SCALE is the median
# within-window std of each sensor over the normal training windows (display units), from
# `python scripts/rationales.py ..\data\windows --split train --scales`; fixed here so rationales do
# not depend on the split they are computed on. P-JUS-CKGL is quantized (median std 0.0018 bar, most
# normal windows flat), so its scale is floored at 0.05 bar.
SCALE = {"P-TPT": 0.104, "T-TPT": 0.0211, "P-ANULAR": 0.101, "P-MON-CKP": 0.165, "P-JUS-CKGL": 0.05,
         "T-JUS-CKP": 0.0533}
MIN_SALIENCE = 5.0    # change / SCALE to call a trend or step
STEP_SHARE = 0.6      # a step: at least this share of the change happens within one minute
MIN_SWINGS = 6        # oscillation: sign changes of the detrended, smoothed series
OSC_AMPLITUDE = 5.0   # oscillation: amplitude / SCALE
MAX_CLAIMS = 3

# Tolerance of the check: a cited change or level is right within this share of the true change,
# or within this many window stds, whichever is larger.
TOL_SHARE = 0.25
TOL_STD = 0.5


def to_display(x: np.ndarray, sensor: str) -> np.ndarray:
    return x / PA_PER_BAR if UNITS[sensor] == "bar" else x


def _fmt(v: float, sensor: str) -> str:
    # Pressures in bar with 2 decimals (as in the prompt means); temperatures too.
    return f"{v:.2f}"


def _fmt_delta(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 10 else f"{v:.1f}"


def sensor_features(x: np.ndarray, sensor: str) -> dict | None:
    """Features of one sensor series (raw units in, display units out); None if too few points."""
    x = to_display(x.astype(np.float64), sensor)
    present = ~np.isnan(x)
    if present.sum() < 2 * EDGE:
        return None
    idx = np.arange(len(x))
    xi = np.interp(idx, idx[present], x[present])
    start, end = float(xi[:EDGE].mean()), float(xi[-EDGE:].mean())
    delta = end - start
    jumps = xi[STEP_LAG:] - xi[:-STEP_LAG]
    k = int(np.argmax(np.abs(jumps)))
    jump = float(jumps[k])
    # Oscillation: detrend linearly, smooth over 30 s, count sign changes.
    resid = xi - np.polyval(np.polyfit(idx, xi, 1), idx)
    smooth = np.convolve(resid, np.ones(3) / 3, mode="same")
    swings = int((np.diff(np.sign(smooth[1:-1])) != 0).sum())
    amplitude = float((np.percentile(resid, 95) - np.percentile(resid, 5)) / 2)
    scale = SCALE[sensor]
    return {
        "start": start, "end": end, "delta": delta, "std": float(xi.std()),
        "jump": jump, "jump_minute": round((k + STEP_LAG / 2) / POINTS_PER_MIN),
        "swings": swings, "amplitude": amplitude,
        "trend_salience": abs(delta) / scale,
        "osc_salience": amplitude / scale if swings >= MIN_SWINGS else 0.0,
    }


def is_step(f: dict) -> bool:
    return abs(f["jump"]) >= STEP_SHARE * abs(f["delta"]) and np.sign(f["jump"]) == np.sign(f["delta"])


def claims_for(X: np.ndarray, shown: np.ndarray) -> list[tuple[float, str, dict]]:
    """Candidate claims (salience, sentence, facts) for the shown sensors of one window."""
    out = []
    for j in np.flatnonzero(shown):
        s = SENSORS[j]
        f = sensor_features(X[j], s)
        if f is None:
            continue
        unit = UNITS[s]
        # One claim per sensor: an oscillation wins over a trend when its swings span at least
        # the net change (flow instability often drifts while it oscillates).
        oscillating = f["osc_salience"] >= OSC_AMPLITUDE and 2 * f["amplitude"] >= abs(f["delta"])
        if f["trend_salience"] >= MIN_SALIENCE and not oscillating:
            verb = "rises" if f["delta"] > 0 else "drops"
            when = f" within one minute around minute {f['jump_minute']}" if is_step(f) else ""
            text = (f"{s} {verb} by {_fmt_delta(abs(f['delta']))} {unit}{when} "
                    f"(from {_fmt(f['start'], s)} to {_fmt(f['end'], s)} {unit}).")
            out.append((f["trend_salience"], text, {"sensor": s, "kind": "trend"}))
        elif oscillating:
            text = f"{s} oscillates with {f['swings']} swings of about ±{_fmt_delta(f['amplitude'])} {unit}."
            out.append((f["osc_salience"], text, {"sensor": s, "kind": "oscillation"}))
    return sorted(out, key=lambda c: -c[0])


def rationale(X: np.ndarray, shown: np.ndarray) -> str:
    """Rationale for one window [6, T] given the mask of the sensors shown to the model."""
    claims = claims_for(X, shown)[:MAX_CLAIMS]
    n_shown = int(shown.sum())
    if not claims:
        return "All shown sensors are stable."
    text = " ".join(c[1] for c in claims)
    if len(claims) < n_shown:
        text += " The other sensors are stable."
    return text


# ------------------------------------------------------------------------------------------------
# Check of a rationale written by the model
# ------------------------------------------------------------------------------------------------

NUM = r"(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
SENSOR_RE = "(" + "|".join(re.escape(s) for s in SENSORS) + ")"
TREND_RE = re.compile(
    SENSOR_RE + r" (rises|drops) by " + NUM + r" (?:bar|°C)"
    r"(?: within one minute around minute (\d+))?"
    r"(?: \(from " + NUM + r" to " + NUM + r" (?:bar|°C)\))?"
)
OSC_RE = re.compile(SENSOR_RE + r" oscillates with (\d+) swings of about ±" + NUM)
STABLE_RE = re.compile(r"(All shown sensors are stable|The other sensors are stable)")


def _close(claimed: float, true: float, ref_delta: float, std: float) -> bool:
    return abs(claimed - true) <= max(TOL_SHARE * abs(ref_delta), TOL_STD * std, 0.01)


def check_rationale(text: str, X: np.ndarray, shown: np.ndarray) -> dict:
    """Recompute every claim of a rationale from the window. Returns counts per outcome."""
    text = text.split("Answer:")[0]
    result = {"claims": 0, "correct": 0, "hallucinated_sensor": 0, "wrong_direction": 0,
              "wrong_number": 0, "not_oscillating": 0, "missed_salient": 0, "details": []}
    cited = set()
    for m in TREND_RE.finditer(text):
        s, verb, d, minute, a, b = m.groups()
        result["claims"] += 1
        cited.add(s)
        j = SENSORS.index(s)
        f = sensor_features(X[j], s) if shown[j] else None
        if f is None:
            result["hallucinated_sensor"] += 1
            result["details"].append((m.group(0), "hallucinated_sensor"))
            continue
        if (verb == "rises") != (f["delta"] > 0):
            outcome = "wrong_direction"
        else:
            ok = _close(float(d), abs(f["delta"]), f["delta"], f["std"])
            if a is not None:
                ok &= _close(float(a), f["start"], f["delta"], f["std"])
                ok &= _close(float(b), f["end"], f["delta"], f["std"])
            if minute is not None:
                ok &= abs(int(minute) - f["jump_minute"]) <= 2
            outcome = "correct" if ok else "wrong_number"
        result[outcome] += 1
        result["details"].append((m.group(0), outcome))
    for m in OSC_RE.finditer(text):
        s, _swings, amp = m.groups()
        result["claims"] += 1
        cited.add(s)
        j = SENSORS.index(s)
        f = sensor_features(X[j], s) if shown[j] else None
        if f is None:
            outcome = "hallucinated_sensor"
        elif f["osc_salience"] < OSC_AMPLITUDE:
            outcome = "not_oscillating"
        else:
            outcome = "correct" if _close(float(amp), f["amplitude"], f["amplitude"], f["std"]) else "wrong_number"
        result[outcome] += 1
        result["details"].append((m.group(0), outcome))
    # Salient changes the rationale leaves out (only counted when it claims stability).
    if STABLE_RE.search(text):
        top = [c for c in claims_for(X, shown)[:MAX_CLAIMS] if c[2]["sensor"] not in cited]
        result["missed_salient"] = len(top)
    return result


def check_file(windows_dir: Path, split: str, predictions: Path) -> dict:
    """Check the rationales of a predictions file (window_id, prediction) against the windows."""
    import pandas as pd

    data = np.load(windows_dir / f"{split}.npz")
    X, mask = data["X"], data["mask"].astype(bool)
    rows = pd.read_csv(windows_dir / f"{split}.csv", dtype={"window_id": str})
    pos = {w: i for i, w in enumerate(rows.window_id)}
    preds = pd.read_csv(predictions, dtype={"window_id": str}, keep_default_na=False)
    keys = ["claims", "correct", "hallucinated_sensor", "wrong_direction", "wrong_number",
            "not_oscillating", "missed_salient"]
    total = dict.fromkeys(keys, 0)
    n, no_claim = 0, 0
    for w, p in zip(preds.window_id, preds.prediction):
        i = pos[w]
        r = check_rationale(str(p), X[i], mask[i])
        n += 1
        no_claim += r["claims"] == 0 and not STABLE_RE.search(str(p))
        for k in keys:
            total[k] += r[k]
    c = max(total["claims"], 1)
    return {
        "windows": n, **total,
        "claim_accuracy": round(total["correct"] / c, 4),
        "hallucinated_share": round(total["hallucinated_sensor"] / c, 4),
        "wrong_direction_share": round(total["wrong_direction"] / c, 4),
        "windows_without_rationale": int(no_claim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--show", type=int, default=0, help="print rationales of N windows per class")
    parser.add_argument("--scales", action="store_true", help="print the SCALE values from normal windows")
    parser.add_argument("--self-check", action="store_true", help="check the generated rationales themselves")
    parser.add_argument("--check-file", type=Path, help="check the rationales of a predictions file")
    args = parser.parse_args()

    data = np.load(args.windows_dir / f"{args.split}.npz")
    X, mask, y = data["X"], data["mask"].astype(bool), data["y"]

    if args.scales:
        normal = y == 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for j, s in enumerate(SENSORS):
                std = np.nanstd(to_display(X[normal, j].astype(np.float64), s), axis=-1)
                std = std[mask[normal, j] & np.isfinite(std)]
                print(f"{s}: median std {np.median(std):.3g} over {len(std)} windows")
        return

    if args.check_file:
        print(json.dumps(check_file(args.windows_dir, args.split, args.check_file), indent=1))
        return

    rng = np.random.default_rng(0)
    if args.show:
        for c in np.unique(y):
            for i in rng.choice(np.flatnonzero(y == c), min(args.show, int((y == c).sum())), replace=False):
                print(f"[{c}] {rationale(X[i], mask[i])}")
            print()

    texts = [rationale(X[i], mask[i]) for i in range(len(X))]
    stable = np.array([t.startswith("All shown") for t in texts])
    print("share of 'all stable' rationales per class:",
          {int(c): round(float(stable[y == c].mean()), 3) for c in np.unique(y)})
    lengths = np.array([len(t.split()) for t in texts])
    print(f"words per rationale: median {np.median(lengths):.0f}, max {lengths.max()}")
    if args.self_check:
        bad = 0
        for i, t in enumerate(texts):
            r = check_rationale(t, X[i], mask[i])
            bad += r["claims"] != r["correct"]
        print(f"self-check: {bad} of {len(texts)} rationales have a claim that fails its own check")


if __name__ == "__main__":
    main()
