"""Explore the Petrobras 3W dataset before designing the TimeNet connector.

Usage:
    python explore_3w.py <path-to-3W-dataset-folder> <output-json>

The dataset folder is the `dataset/` directory of https://github.com/petrobras/3W, which holds one
subfolder per class (0-9) with one Parquet file per instance. File names start with WELL-, SIMULATED_
or DRAWN_, which gives the provenance of each instance.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LABEL_COLUMNS = {"class", "state"}


def provenance(name: str) -> str:
    """Return real, simulated or drawn from an instance file name."""
    if name.startswith("WELL-"):
        return "real"
    if name.startswith("SIMULATED"):
        return "simulated"
    if name.startswith("DRAWN"):
        return "drawn"
    return "unknown"


def well_id(name: str) -> str | None:
    """Return the well id (e.g. WELL-00001) of a real instance."""
    return name.split("_")[0] if name.startswith("WELL-") else None


def summarize_instance(path: Path) -> dict:
    """Read one instance and summarize its labels and variables."""
    table = pq.read_table(path)
    columns = table.column_names
    n_rows = table.num_rows
    info: dict = {
        "file": path.name,
        "class_folder": path.parent.name,
        "provenance": provenance(path.name),
        "well": well_id(path.name),
        "n_rows": n_rows,
        "columns": columns,
        "variables": {},
    }
    if "class" in columns:
        labels = table.column("class").to_numpy(zero_copy_only=False)
        labels = labels[~np.isnan(labels.astype(float))] if labels.dtype.kind == "f" else labels
        info["label_counts"] = {str(int(k)): int(v) for k, v in Counter(labels.tolist()).items()}
    for name in columns:
        if name in LABEL_COLUMNS:
            continue
        arr = table.column(name).to_numpy(zero_copy_only=False)
        if arr.dtype.kind not in "fiub":
            continue
        values = arr.astype(float)
        present = ~np.isnan(values)
        n_present = int(present.sum())
        frozen = bool(n_present > 1 and np.nanstd(values) == 0)
        info["variables"][name] = {"present_frac": n_present / n_rows if n_rows else 0.0, "frozen": frozen}
    return info


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    files = sorted(root.glob("*/*.parquet"))
    instances = [summarize_instance(f) for f in files]

    by_class_prov: dict = defaultdict(Counter)
    wells_by_class: dict = defaultdict(set)
    var_usable: dict = defaultdict(Counter)
    durations: dict = defaultdict(list)
    label_kinds: dict = defaultdict(Counter)
    all_columns: Counter = Counter()

    for inst in instances:
        cls, prov = inst["class_folder"], inst["provenance"]
        by_class_prov[cls][prov] += 1
        durations[prov].append(inst["n_rows"])
        all_columns.update(inst["columns"])
        if inst["well"]:
            wells_by_class[cls].add(inst["well"])
        if prov == "real":
            for var, stats in inst["variables"].items():
                usable = stats["present_frac"] > 0.5 and not stats["frozen"]
                var_usable[var]["usable" if usable else "unusable"] += 1
        for label, count in inst.get("label_counts", {}).items():
            kind = "normal" if label == "0" else ("transient" if int(label) >= 100 else "steady")
            label_kinds[prov][kind] += count

    n_real = sum(c["real"] for c in by_class_prov.values())
    summary = {
        "n_files": len(files),
        "columns_seen": dict(all_columns),
        "by_class": {
            cls: {**dict(by_class_prov[cls]), "real_wells": len(wells_by_class[cls])}
            for cls in sorted(by_class_prov, key=int)
        },
        "n_distinct_wells": len(set().union(*wells_by_class.values())) if wells_by_class else 0,
        "real_variable_usable_frac": {
            var: round(c["usable"] / n_real, 3) for var, c in sorted(var_usable.items())
        },
        "duration_rows": {
            prov: {
                "min": int(np.min(v)),
                "median": int(np.median(v)),
                "max": int(np.max(v)),
            }
            for prov, v in durations.items()
        },
        "label_rows_by_provenance": {prov: dict(c) for prov, c in label_kinds.items()},
    }
    out.write_text(json.dumps({"summary": summary, "instances": instances}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
