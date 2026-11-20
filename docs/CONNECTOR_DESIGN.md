# Design of the `petrobras/three-w` connector

**Status:** approved on 2026-11-17 and implemented in the fork, branch `feature/petrobras-three-w`,
folder `packages/timenet-connectors/src/timenet_connectors/datasets/petrobras/three_w/`. Submitted
as [PR #833](https://github.com/OpenTSLM/TimeNet/pull/833). The connector's own `README.md` states
the final decisions with the numbers measured on the release; this document records the design and
the choices made on the way.

Based on 3W v2.0.0 at commit `c86cc7c70063646910be7292de8a54a6fb2a8399`, the exploration in
`docs/3W_EXPLORATION.md`, and the existing TimeNet connectors (mainly `physionet/sleep-edfx`, which
like 3W has long recordings with interval labels).

## 1. Identity
| Item | Value |
|---|---|
| Id | `petrobras/three-w`, folder `datasets/petrobras/three_w/`. TimeNet derives the folder name from the id (lowercase, hyphens to underscores): `petrobras/3w` would give a folder `3w`, which is not an importable Python module |
| Version | `1.0.0` (version of the TimeNet conversion; the 3W version, 2.0.0, is in the description) |
| License | `CC-BY-4.0`, with `license_url` pointing to `dataset/LICENSE-CC-BY` of the release |
| Domains | `industrial`, `energy` |
| Card | `source_url`, `source_revision` (the pinned commit) and the citation of the 2.0.0 paper |

## 2. Source and download
- No existing base fits (neither Hugging Face nor PhysioNet): a direct `BaseConnector` subclass.
- The 2,228 Parquet files are in plain git without LFS, so each has a commit-pinned URL:
  `https://raw.githubusercontent.com/petrobras/3W/<commit>/dataset/<class>/<file>.parquet`.
- **Manifest** `files.csv` in the connector folder: path, size and SHA-256 of every file, generated
  once by `generate_manifest.py` and committed. `download` passes the 2,228 `Artifact`s to
  `download_files`, which fetches them in parallel and checks every SHA-256.
- `download` does I/O only; `convert` reads the files without the network, and signal values load
  lazily.

## 3. Structure: one record per instance

| TimeNet element | From 3W |
|---|---|
| **Record** | One Parquet file (one instance), sampled at exactly 1 Hz (checked) |
| `Record.metadata` | `instance`, `well` (real only), `provenance` (`real` / `simulated` / `drawn`), `event_code` (0-9), `start` (first timestamp as text: 3W states no time zone, so there is no absolute start time) |
| **Source** | Sensor groups by measuring point |
| **Signal** | One 3W variable, named after its column, with its `dataset.ini` description in the metadata |
| **Interval annotations** | The event phases from the `class` column |
| **Record annotations** | Provenance, well, event type |
| **Tasks** | Event-type classification and localization of the transient phase |

The signals share six specs by quantity: `pressure`, `temperature`, `flow_rate`, `valve_opening`,
`valve_state`, `well_status`.

### Missing data (repo rule: never fill)
- A variable **empty for a whole instance** becomes no signal of that record: that well has no such
  sensor.
- A variable **partly empty** becomes a signal with `MissingValues(reason=NOT_RECORDED)`; nulls stay
  null.
- A variable **constant for a whole instance** (for example `P-PDG` stuck at 0) stays as it is, with
  a `constant_value` annotation on the signal, because it usually means a stuck sensor.

### Event phases (from the `class` column)
| `class` value | Annotation |
|---|---|
| 0 | `event_phase = normal` |
| 1-9 | `event_phase = steady` |
| 101-109 | `event_phase = transient` |
| empty | no annotation (about 5% of the seconds) |

## 4. Tasks
| Task | One per | Answer |
|---|---|---|
| `ClassificationTask` "event type" | Every instance (2,228) | One of the 10 labels of `dataset.ini` |
| `TemporalLocalizationTask` "transient phase" | Every instance whose event has a transient phase (1,142) | The transient intervals, `EXHAUSTIVE` mode |

The **windows** for training are not in the connector: they are a choice of this project, defined
in `docs/DATA_CONTRACT.md`. The connector stays faithful to the original data, like the other
TimeNet connectors.

## 5. Split
- **Real instances:** split by well (no well in more than one split) with TimeNet's
  `StratifiedSplitter`, stratified by event type and grouped by well, 3/1/1 folds.
- **Simulated and drawn instances:** always in train, because they come from no well.
- Classes 3 and 6 have only 2 real wells, so validation or test can miss them; the card says so.

## 6. Decisions taken with the team (2026-11-17)
1. Id `petrobras/three-w`.
2. Variables empty for a whole instance are dropped from the record.
3. The `state` codes (0-8) are kept as numbers, since 3W does not document them.
4. Simulated and drawn instances always go to train.
5. The localization task is in the first PR, because it supports early warning.
