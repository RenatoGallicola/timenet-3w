# Data contract

The common format between the "Data and TimeNet" part and the "Model and demo" part.
Approved on 2026-11-20. Every change must be agreed and recorded in `docs/DECISIONS.md`.

The script that implements it is `scripts/make_windows.py`. By default it reads the
`petrobras/three-w` dataset that the connector of
[OpenTSLM/TimeNet PR #833](https://github.com/OpenTSLM/TimeNet/pull/833) builds from 3W (commit
`c86cc7c70063646910be7292de8a54a6fb2a8399`), so the model trains on data that comes from TimeNet.
With `--source parquet` it reads the 3W Parquet files directly. `scripts/compare_windows.py` checked
on 2026-11-20 that both sources give the same 18,737 windows, value for value.

```powershell
# Build the TimeNet dataset once (downloads 3W, about 20 minutes, 1.1 GB)
uvx --from "git+https://github.com/RenatoGallicola/TimeNet@feature/petrobras-three-w#subdirectory=packages/timenet-connectors" timenet-build build petrobras/three-w --local-registry ..\registry
# Cut the windows from it, with the connector's split
uv run python scripts/make_windows.py ..\registry ..\data\windows --splits splits/wells_connector.json
```

## Main evaluation

| Item | Decision |
|---|---|
| Instances used | **Real** only (`WELL-*`) |
| Classes | **0, 2, 4, 7, 8, 9**: the only ones with at least 6 real wells |
| Rare classes (1, 3, 5, 6) | Outside the main evaluation. Optional separate experiment: train on simulated instances, test on real ones |
| Variables (6, in this order) | `P-TPT`, `T-TPT`, `P-ANULAR`, `P-MON-CKP`, `P-JUS-CKGL`, `T-JUS-CKP` |
| Units | The original 3W units (Pa, °C), not normalized. The model does the normalization |
| Window | **30 minutes** (1,800 s at 1 Hz) |
| Resampling | Mean over 10 s blocks → **0.1 Hz, 180 points** per variable. A block without values stays NaN |
| Stride between windows | 15 minutes (900 s) for normal windows; **5 minutes** (300 s) for windows with an event, which lasts only 1-3 hours |
| Maximum per instance | 20 windows per instance, chosen at regular intervals, so that long instances do not weigh too much |

## Window label

Each second of 3W has a label: 0 (normal), `c` (event `c`, steady state), `100 + c` (event `c`,
transient), or empty.

| Window label | Rule |
|---|---|
| Event `c` | At least **50%** of the seconds are labeled `c` or `100 + c` |
| Normal (0) | At least **95%** of the seconds are labeled 0 |
| Dropped | Every other case, and windows with more than 20% of unlabeled seconds |

Normal windows can also come from the normal part of instances with an event, so that the normal
class covers more wells.

Each window also stores its **phase**: `transient` if most of the event's seconds are transient,
otherwise `steady` (`normal` for normal windows). Metrics are reported on all windows and,
separately, on the `transient` windows only: that is the early-warning measure. Classes 3 and 4 of
3W have no transient phase.

## Missing variables (never filled)
- A variable is **present** in a window if at least 80% of its 180 points are not NaN.
- Missing values stay NaN in the files. They are not interpolated.
- Only the present variables enter the OpenTSLM prompt (the mask is saved).
- Windows with fewer than 2 present variables are dropped.

## Split
- **By well:** no well appears in more than one split.
- **Decided on 2026-11-20: the connector's split from PR #833**, saved in
  `splits/wells_connector.json` (passed with `--splits`).
- The alternative split `splits/wells.json` (about 70% / 15% / 15% of the wells, chosen by a
  fixed-seed search so that each class has at least 5% of its windows in both validation and test)
  stays in the repo for comparison.

## Exchange format
In the `data/windows/` folder (outside git):

| File | Content |
|---|---|
| `{train,val,test}.npz` | `X`: float32 `[N, 6, 180]` with NaN for missing values; `mask`: bool `[N, 6]` (variable present); `y`: int `[N]` (3W class: 0, 2, 4, 7, 8, 9) |
| `{train,val,test}.csv` | One row per window, in the same order: `window_id`, `instance`, `well`, `label`, `phase`, `start_s`, `event_frac`, `transient_frac` |
| `meta.json` | 3W commit, variables, class names, window parameters, counts |

## Predictions format
Every model (baselines, OpenTSLM, perturbed runs) writes its predictions for a split as a CSV with
one row per window, read by `scripts/evaluate.py`:

| Column | Content |
|---|---|
| `window_id` | The id from `{split}.csv` |
| `prediction` | A 3W class code (0, 2, 4, 7, 8, 9), a class name, or the model's raw answer ("... Answer: Flow instability") |

An answer that maps to no class, and a window without a prediction, count as errors; their share is
reported. The baselines write theirs with `scripts/baselines.py --predictions-dir results/predictions`.

```powershell
# Score one model, or compare two on the same windows.
# Use the validation split while developing, the test split once at the end.
uv run python scripts/evaluate.py ..\data\windows results\predictions\xgboost_shape_val.csv --split val
uv run python scripts/evaluate.py ..\data\windows A_test.csv --compare B_test.csv --split test --out results\eval.json
```
