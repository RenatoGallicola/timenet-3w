"""Bridge from the 3W windows (docs/DATA_CONTRACT.md) to OpenTSLM samples (issue #7).

Each window becomes a dict in the format of OpenTSLM's `QADataset._format_sample`:
`pre_prompt`, `time_series` (one 1-D float32 array per sensor), `time_series_text` (one line per
sensor), `post_prompt` and `answer`, plus `window_id` and `label`. OpenTSLM's collate function
(`extend_time_series_to_match_patch_size_and_aggregate`) and its models take these dicts as they
are.

- Only the present sensors enter the prompt, each with its name, unit, mean and standard deviation.
  The prompt statistics come from `stats_mean`/`stats_std` when the window file has them (perturbed
  windows of issue #3); otherwise they are computed from `X`. A NaN statistic is left out.
- The series given to the encoder is always normalized with the statistics of `X` itself, so a
  perturbation of the prompt statistics does not change the signal. OpenTSLM rejects NaN: gaps
  inside a present sensor are linearly interpolated in the encoder input only (the window files
  keep their NaN, as the data contract requires). The prompt says how many points were missing.
- In training, `sensor_dropout=True` hides a random subset of the present sensors at every
  access (at least 2 stay), as in `scripts/baselines.py`, so the model cannot learn which sensors a
  well has instead of what the signals do.
- `balanced_weights` gives per-window weights for `torch.utils.data.WeightedRandomSampler`, so the
  rare classes are seen as often as the common ones.

Only numpy and pandas are needed, so the module runs locally and in the Modal image.

    uv run python scripts/opentslm_bridge.py ..\\data\\windows --split val --show 2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Name, description and display unit of each sensor, in the order of the data contract.
SENSORS = {
    "P-TPT": ("pressure at the subsea Christmas tree (TPT)", "bar"),
    "T-TPT": ("temperature at the subsea Christmas tree (TPT)", "°C"),
    "P-ANULAR": ("pressure in the well annulus", "bar"),
    "P-MON-CKP": ("pressure upstream of the production choke (PCK)", "bar"),
    "P-JUS-CKGL": ("pressure downstream of the gas lift choke (CKGL)", "bar"),
    "T-JUS-CKP": ("temperature downstream of the production choke (PCK)", "°C"),
}
PA_PER_BAR = 1e5
MIN_STD = 1e-6
MIN_SENSORS = 2

PRE_PROMPT = """You are given sensor data from an offshore oil well, recorded over 30 minutes (180 points, one every 10 seconds).
Each sensor is shown normalized to zero mean and unit variance; its real mean and standard deviation are given in its description.
Not every well has every sensor: only the available sensors are shown.

Your task is to classify the state of the well in this window.
Possible answers:
{labels}."""

POST_PROMPT_LABEL = "Answer:"
POST_PROMPT_RATIONALE = 'Rationale (end with "Answer: <state>"):'


def class_names(windows_dir: Path) -> dict[int, str]:
    meta = json.loads((Path(windows_dir) / "meta.json").read_text(encoding="utf-8"))
    return {int(code): name for code, name in meta["classes"].items()}


def window_stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the present points of each sensor; NaN for absent sensors.

    Same definition as `scripts/perturb.py` (population std, `np.nanstd`).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, std = np.nanmean(X, axis=-1), np.nanstd(X, axis=-1)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    mean[~mask], std[~mask] = np.nan, np.nan
    return mean, std


def encoder_series(x: np.ndarray) -> np.ndarray:
    """Normalize one sensor series and interpolate its gaps, for the encoder input only."""
    present = ~np.isnan(x)
    mean, std = x[present].mean(), x[present].std()
    z = (x - mean) / max(float(std), MIN_STD)
    if not present.all():
        idx = np.arange(len(x))
        z = np.interp(idx, idx[present], z[present])
    return z.astype(np.float32)


def sensor_dropout(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hide a random subset of the present sensors of one window, keeping at least 2."""
    m = mask.copy()
    present = np.flatnonzero(m)
    if len(present) > MIN_SENSORS:
        keep = rng.integers(MIN_SENSORS, len(present) + 1)
        m[rng.choice(present, len(present) - keep, replace=False)] = False
    return m


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-window sampling weights that give every class the same total weight."""
    classes, counts = np.unique(y, return_counts=True)
    per_class = dict(zip(classes.tolist(), (1.0 / counts).tolist()))
    return np.array([per_class[int(c)] for c in y], dtype=np.float64)


def _format_value(value: float, unit: str, digits: str) -> str:
    if unit == "bar":
        value = value / PA_PER_BAR
    return f"{value:{digits}} {unit}"


def sensor_text(name: str, mean: float, std: float, n_missing: int, n_points: int) -> str:
    description, unit = SENSORS[name]
    text = f"The following is {name}, the {description}"
    stats = []
    if not np.isnan(mean):
        stats.append(f"mean {_format_value(mean, unit, '.2f')}")
    if not np.isnan(std):
        # Significant digits: many stds are far below 0.01 bar.
        stats.append(f"std {_format_value(std, unit, '.3g')}")
    if stats:
        text += ", with " + " and ".join(stats)
    if n_missing:
        text += f"; {n_missing} of {n_points} points were missing and are interpolated"
    return text + ":"


class ThreeWQADataset:
    """Map-style dataset of OpenTSLM samples for one split of a windows folder.

    Works with `torch.utils.data.DataLoader` (no torch import needed here). With
    `sensor_dropout=True` every access draws a new random subset of sensors.

    `rationales` maps `window_id` to a rationale text (issue #9); when given, the answer is
    "<rationale> Answer: <state>" after a "Rationale" post prompt. Otherwise the post prompt is
    "Answer:" and the answer is the state name alone.
    """

    def __init__(
        self,
        windows_dir: Path | str,
        split: str,
        eos_token: str = "",
        sensor_dropout: bool = False,
        seed: int = 0,
        rationales: dict[str, str] | None = None,
    ):
        windows_dir = Path(windows_dir)
        data = np.load(windows_dir / f"{split}.npz")
        self.X, self.mask, self.y = data["X"], data["mask"].astype(bool), data["y"]
        if "stats_mean" in data.files and "stats_std" in data.files:
            self.prompt_mean, self.prompt_std = data["stats_mean"], data["stats_std"]
        else:
            self.prompt_mean, self.prompt_std = window_stats(self.X, self.mask)
        self.rows = pd.read_csv(windows_dir / f"{split}.csv", dtype={"window_id": str})
        if len(self.rows) != len(self.X):
            raise ValueError(f"{split}: {len(self.rows)} csv rows but {len(self.X)} windows")

        meta = json.loads((windows_dir / "meta.json").read_text(encoding="utf-8"))
        self.variables = meta["variables"]
        unknown = set(self.variables) - set(SENSORS)
        if unknown:
            raise ValueError(f"No description for sensors {sorted(unknown)}")
        self.names = {int(code): name for code, name in meta["classes"].items()}
        self.pre_prompt = PRE_PROMPT.format(labels=", ".join(self.names.values()))
        self.eos_token = eos_token
        self.sensor_dropout = sensor_dropout
        self.rng = np.random.default_rng(seed)
        self.rationales = rationales

    def __len__(self) -> int:
        return len(self.X)

    def answer(self, i: int) -> str:
        label = self.names[int(self.y[i])]
        if self.rationales is not None:
            label = f"{self.rationales[self.rows.window_id.iat[i]].strip()} Answer: {label}"
        return label + self.eos_token

    def __getitem__(self, i: int) -> dict:
        mask = self.mask[i]
        if self.sensor_dropout:
            mask = sensor_dropout(mask, self.rng)
        series, texts = [], []
        n_points = self.X.shape[-1]
        for j in np.flatnonzero(mask):
            x = self.X[i, j]
            n_missing = int(np.isnan(x).sum())
            if n_missing == n_points:
                continue
            series.append(encoder_series(x))
            texts.append(
                sensor_text(
                    self.variables[j], float(self.prompt_mean[i, j]), float(self.prompt_std[i, j]),
                    n_missing, n_points,
                )
            )
        return {
            "pre_prompt": self.pre_prompt,
            "time_series": series,
            "time_series_text": texts,
            "post_prompt": POST_PROMPT_RATIONALE if self.rationales is not None else POST_PROMPT_LABEL,
            "answer": self.answer(i),
            "window_id": self.rows.window_id.iat[i],
            "label": int(self.y[i]),
        }

    def sample_weights(self) -> np.ndarray:
        return balanced_weights(self.y)


def check(ds: ThreeWQADataset) -> dict:
    """Build every sample once and check what OpenTSLM requires."""
    n_series = []
    for i in range(len(ds)):
        s = ds[i]
        if len(s["time_series"]) < MIN_SENSORS:
            raise ValueError(f"{s['window_id']}: fewer than {MIN_SENSORS} sensors")
        if len(s["time_series"]) != len(s["time_series_text"]):
            raise ValueError(f"{s['window_id']}: series and texts differ in number")
        for ts in s["time_series"]:
            if ts.ndim != 1 or not np.isfinite(ts).all():
                raise ValueError(f"{s['window_id']}: non-finite encoder input")
        n_series.append(len(s["time_series"]))
    counts = np.unique(n_series, return_counts=True)
    return {"windows": len(ds), "sensors_per_window": {int(k): int(v) for k, v in zip(*counts)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--show", type=int, default=1, help="number of example prompts to print")
    parser.add_argument("--dropout", action="store_true", help="apply the training sensor dropout")
    parser.add_argument("--no-check", action="store_true", help="skip the check of every window")
    args = parser.parse_args()

    ds = ThreeWQADataset(args.windows_dir, args.split, sensor_dropout=args.dropout)
    for i in np.linspace(0, len(ds) - 1, args.show, dtype=int):
        s = ds[int(i)]
        print("=" * 80)
        print(s["pre_prompt"])
        for text, ts in zip(s["time_series_text"], s["time_series"]):
            print(f"{text} <{len(ts)} points>")
        print(s["post_prompt"], "|", s["answer"])
    if not args.no_check:
        print("=" * 80)
        print("check:", check(ds))
        w = ds.sample_weights()
        print("weight per class:", {int(c): round(float(w[ds.y == c].sum()), 3) for c in np.unique(ds.y)})


if __name__ == "__main__":
    main()
