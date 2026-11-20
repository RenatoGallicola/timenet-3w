"""Cut Petrobras 3W real instances into fixed windows, following docs/DATA_CONTRACT.md.

Usage:
    uv run python scripts/make_windows.py <source> <output folder> [--source timenet|parquet] [--splits ...]

With `--source timenet` (the default), <source> is a local TimeNet registry that holds the
`petrobras/three-w` dataset built by the connector of OpenTSLM/TimeNet PR #833. With
`--source parquet`, <source> is the `dataset/` folder of a 3W checkout and the Parquet files are read
directly. Both sources give the same windows (checked by scripts/compare_windows.py). Writes, for each split,
`{split}.npz` (X [N, 6, 180] float32 with NaN, mask [N, 6] bool, y [N] int) and `{split}.csv`
(one row per window, same order), plus `meta.json`. If the splits file does not exist, a split by
well is searched with a fixed seed and saved there.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

THREEW_COMMIT = "c86cc7c70063646910be7292de8a54a6fb2a8399"
TIMENET_DATASET = "petrobras/three-w"
VARIABLES = ["P-TPT", "T-TPT", "P-ANULAR", "P-MON-CKP", "P-JUS-CKGL", "T-JUS-CKP"]
CLASSES = [0, 2, 4, 7, 8, 9]
CLASS_NAMES = {
    0: "Normal operation",
    1: "Abrupt increase of BSW",
    2: "Spurious closure of DHSV",
    3: "Severe slugging",
    4: "Flow instability",
    5: "Rapid productivity loss",
    6: "Quick restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in production line",
    9: "Hydrate in service line",
}

WINDOW_S = 1800
STRIDE_S = 900  # normal windows
EVENT_STRIDE_S = 300  # event windows: events last 1-3 h, a shorter stride gives enough windows
BLOCK_S = 10
N_POINTS = WINDOW_S // BLOCK_S
MAX_WINDOWS_PER_INSTANCE = 20
EVENT_MIN_FRAC = 0.5
NORMAL_MIN_FRAC = 0.95
MAX_UNLABELED_FRAC = 0.2
VAR_PRESENT_MIN_FRAC = 0.8
MIN_PRESENT_VARS = 2
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEARCH_SEEDS = 5000
MIN_CLASS_SHARE = 0.05  # every class keeps at least this share of its windows in val and in test


def read_instance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return values [6, T] (NaN where missing or absent) and labels [T] (NaN where unlabeled)."""
    schema_names = set(pq.read_schema(path).names)
    wanted = [v for v in VARIABLES if v in schema_names]
    table = pq.read_table(path, columns=wanted + ["class"])
    n = table.num_rows
    values = np.full((len(VARIABLES), n), np.nan, dtype=np.float64)
    for i, var in enumerate(VARIABLES):
        if var in schema_names:
            values[i] = table.column(var).to_numpy(zero_copy_only=False).astype(np.float64)
    labels = table.column("class").to_numpy(zero_copy_only=False).astype(np.float64)
    return values, labels


def iter_parquet(dataset_dir: Path) -> Iterator[tuple[str, int, np.ndarray, np.ndarray]]:
    """Yield (instance, event class, values, labels) for each real instance of a 3W checkout."""
    files = sorted(dataset_dir.glob("*/WELL-*.parquet"))
    if not files:
        raise SystemExit(f"No real instances found under {dataset_dir}")
    for path in files:
        values, labels = read_instance(path)
        yield path.stem, int(path.parent.name), values, labels


def iter_timenet(registry: Path, provenance: str = "real") -> Iterator[tuple[str, int, np.ndarray, np.ndarray]]:
    """Yield (instance, event class, values, labels) for each record of the TimeNet dataset.

    Only records of the given provenance (`real`, `simulated` or `drawn`) are read.

    The per-second labels are rebuilt from the `event_phase` interval annotations of the
    connector: `normal` is 0, `steady` is the event code, `transient` is 100 + the event code, and
    a second outside every interval stays unlabeled (NaN), as in the source.
    """
    import timenet
    from timenet.types.text import read_value

    us_per_s = 1_000_000
    with timenet.TimeNet(sources=[str(registry)]).open(TIMENET_DATASET) as reader:
        dataset = reader.read()
        records = [r for r in dataset.records if r.metadata["provenance"] == provenance]
        if not records:
            raise SystemExit(f"No {provenance} records found in {TIMENET_DATASET} under {registry}")
        records.sort(key=lambda r: (int(r.metadata["event_code"]), str(r.metadata["instance"])))
        for record in records:
            event = int(record.metadata["event_code"])
            n = record.time_span.exclusive_end // us_per_s
            signals = {s.name: s for source in record.sources for s in source.signals}
            values = np.full((len(VARIABLES), n), np.nan, dtype=np.float64)
            for i, var in enumerate(VARIABLES):
                if var in signals:
                    values[i] = signals[var].to_arrow().to_numpy(zero_copy_only=False).astype(np.float64)
            labels = np.full(n, np.nan, dtype=np.float64)
            for annotation in record.annotations:
                if annotation.key != "event_phase" or annotation.span is None:
                    continue
                phase = read_value(annotation.value)
                code = {"normal": 0, "steady": event, "transient": 100 + event}[phase]
                labels[annotation.span.start_us // us_per_s : annotation.span.exclusive_end // us_per_s] = code
            yield str(record.metadata["instance"]), event, values, labels


def label_window(labels: np.ndarray, event_class: int) -> tuple[int | None, str, float, float]:
    """Apply the labelling rule of the contract. Returns (label or None, phase, event_frac, transient_frac)."""
    n = len(labels)
    unlabeled = np.isnan(labels).sum() / n
    if unlabeled > MAX_UNLABELED_FRAC:
        return None, "", 0.0, 0.0
    normal = (labels == 0).sum() / n
    if normal >= NORMAL_MIN_FRAC:
        return 0, "normal", 0.0, 0.0
    if event_class == 0:
        return None, "", 0.0, 0.0
    steady = (labels == event_class).sum() / n
    transient = (labels == 100 + event_class).sum() / n
    event = steady + transient
    if event >= EVENT_MIN_FRAC:
        return event_class, ("transient" if transient > steady else "steady"), event, transient
    return None, "", event, transient


def resample(values: np.ndarray) -> np.ndarray:
    """Mean over 10 s blocks: [6, 1800] -> [6, 180]. A block with no values stays NaN."""
    blocks = values.reshape(values.shape[0], N_POINTS, BLOCK_S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=2).astype(np.float32)


def windows_of_instance(instance: str, event_class: int, values: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Cut one real instance into labelled windows (at most MAX_WINDOWS_PER_INSTANCE)."""
    n = labels.shape[0]
    candidates = []
    for start in range(0, n - WINDOW_S + 1, EVENT_STRIDE_S):
        label, phase, event_frac, transient_frac = label_window(labels[start : start + WINDOW_S], event_class)
        if label is None or label not in CLASSES:
            continue
        if label == 0 and start % STRIDE_S != 0:
            continue
        x = resample(values[:, start : start + WINDOW_S])
        mask = (~np.isnan(x)).mean(axis=1) >= VAR_PRESENT_MIN_FRAC
        if mask.sum() < MIN_PRESENT_VARS:
            continue
        candidates.append(
            {
                "instance": instance,
                "well": instance.split("_")[0],
                "label": label,
                "phase": phase,
                "start_s": start,
                "event_frac": round(event_frac, 3),
                "transient_frac": round(transient_frac, 3),
                "x": x,
                "mask": mask,
            }
        )
    # Cap per instance and per label, evenly spaced in time, so long instances do not dominate.
    kept = []
    by_label = defaultdict(list)
    for w in candidates:
        by_label[w["label"]].append(w)
    for group in by_label.values():
        if len(group) > MAX_WINDOWS_PER_INSTANCE:
            idx = np.linspace(0, len(group) - 1, MAX_WINDOWS_PER_INSTANCE).round().astype(int)
            group = [group[i] for i in sorted(set(idx))]
        kept.extend(group)
    return kept


def search_split(counts: pd.DataFrame) -> dict[str, str]:
    """Assign wells to splits so every class is in every split and window shares are close to targets.

    `counts` has one row per well and one column per class (number of windows).
    """
    wells = counts.index.to_numpy()
    totals = counts.sum(axis=0)
    best, best_score = None, np.inf
    for seed in range(SPLIT_SEARCH_SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(wells)
        n_test = max(1, round(len(wells) * SPLIT_FRACS["test"]))
        n_val = max(1, round(len(wells) * SPLIT_FRACS["val"]))
        assign = {w: "test" for w in order[:n_test]}
        assign.update({w: "val" for w in order[n_test : n_test + n_val]})
        assign.update({w: "train" for w in order[n_test + n_val :]})
        score = 0.0
        for split, target in SPLIT_FRACS.items():
            share = counts.loc[[w for w in wells if assign[w] == split]].sum(axis=0) / totals
            if split != "train" and (share < MIN_CLASS_SHARE).any() or (share == 0).any():
                score = np.inf
                break
            score += float(np.abs(share - target).sum())
        if score < best_score:
            best, best_score = assign, score
    if best is None:
        raise RuntimeError("No split puts every class in every split")
    return {w: best[w] for w in sorted(best)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="TimeNet registry (default) or 3W dataset/ folder")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--source", choices=("timenet", "parquet"), default="timenet")
    parser.add_argument("--splits", type=Path, default=Path(__file__).resolve().parent.parent / "splits" / "wells.json")
    args = parser.parse_args()

    instances = iter_timenet(args.source_dir) if args.source == "timenet" else iter_parquet(args.source_dir)
    windows: list[dict] = []
    for i, (instance, event_class, values, labels) in enumerate(instances, 1):
        windows.extend(windows_of_instance(instance, event_class, values, labels))
        if i % 100 == 0:
            print(f"{i} instances, {len(windows)} windows")
    print(f"{i} instances, {len(windows)} windows")

    meta_rows = pd.DataFrame([{k: v for k, v in w.items() if k not in ("x", "mask")} for w in windows])
    counts = meta_rows.groupby(["well", "label"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)

    if args.splits.exists():
        assignment = json.loads(args.splits.read_text())["wells"]
        missing = set(counts.index) - set(assignment)
        if missing:
            raise SystemExit(f"Wells missing from {args.splits}: {sorted(missing)}")
    else:
        assignment = search_split(counts)
        args.splits.parent.mkdir(parents=True, exist_ok=True)
        args.splits.write_text(
            json.dumps(
                {"threew_commit": THREEW_COMMIT, "method": "random search over seeds, see scripts/make_windows.py", "wells": assignment},
                indent=1,
            )
        )
        print(f"Saved new split to {args.splits}")

    meta_rows["split"] = meta_rows["well"].map(assignment)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLIT_FRACS:
        idx = np.flatnonzero(meta_rows["split"].to_numpy() == split)
        X = np.stack([windows[i]["x"] for i in idx]).astype(np.float32)
        mask = np.stack([windows[i]["mask"] for i in idx])
        y = meta_rows["label"].to_numpy()[idx].astype(np.int64)
        np.savez_compressed(args.out_dir / f"{split}.npz", X=X, mask=mask, y=y)
        rows = meta_rows.iloc[idx].drop(columns="split").reset_index(drop=True)
        rows.insert(0, "window_id", [f"{r.instance}@{r.start_s}" for r in rows.itertuples()])
        rows.to_csv(args.out_dir / f"{split}.csv", index=False)
        summary[split] = {
            "windows": int(len(idx)),
            "wells": int(rows["well"].nunique()),
            "by_class": {str(c): int((y == c).sum()) for c in CLASSES},
            "transient_by_class": {str(c): int(((y == c) & (rows["phase"] == "transient").to_numpy()).sum()) for c in CLASSES},
        }

    meta = {
        "source": f"TimeNet {TIMENET_DATASET}" if args.source == "timenet" else "3W Parquet files",
        "threew_commit": THREEW_COMMIT,
        "variables": VARIABLES,
        "classes": {str(c): CLASS_NAMES[c] for c in CLASSES},
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "event_stride_s": EVENT_STRIDE_S,
        "sample_period_s": BLOCK_S,
        "n_points": N_POINTS,
        "max_windows_per_instance": MAX_WINDOWS_PER_INSTANCE,
        "summary": summary,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
