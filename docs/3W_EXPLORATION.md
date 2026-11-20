# Exploration of 3W

Run on 2026-11-17 with `scripts/explore_3w.py` on 3W v2.0.0, commit
`c86cc7c70063646910be7292de8a54a6fb2a8399` (November 14, 2026), folder `dataset/`: 2,228 Parquet
files, 1.9 GB.

## File structure
- One folder per class (0-9), one Parquet file per instance.
- The file name prefix gives the provenance: `WELL-xxxxx_` (real, with the well id), `SIMULATED_`,
  `DRAWN_`.
- Columns: `timestamp`, 27 variables, `class` (per-second label), `state` (operational status of the
  well).
- 1 Hz. Transient label = class + 100.

## Instances per class

| Class | Event | Real | Real wells | Simulated | Drawn |
|---|---|---|---|---|---|
| 0 | Normal operation | 594 | 9 | 0 | 0 |
| 1 | Abrupt increase of BSW | 4 | 3 | 114 | 10 |
| 2 | Spurious closure of DHSV | 22 | 7 | 16 | 0 |
| 3 | Severe slugging | 32 | 2 | 74 | 0 |
| 4 | Flow instability | 343 | 7 | 0 | 0 |
| 5 | Rapid productivity loss | 11 | 3 | 439 | 0 |
| 6 | Quick restriction in PCK | 6 | 2 | 215 | 0 |
| 7 | Scaling in PCK | 36 | 6 | 0 | 10 |
| 8 | Hydrate in production line | 14 | 9 | 81 | 0 |
| 9 | Hydrate in service line | 57 | 15 | 150 | 0 |

Total: 1,119 real instances from 40 wells, 1,089 simulated, 20 drawn.

**Consequence for the split:** classes 0, 2, 4, 7, 8 and 9 have at least 6 real wells, so they can be
evaluated by well. Classes 1, 3, 5 and 6 have 2-3 real wells: they belong in a sim-to-real
experiment (train on simulated instances, test on the few real ones).

## Variables actually usable in real instances
Share of real instances in which the variable is present for more than half of the time and is not
constant:

| Variable | Meaning | Share |
|---|---|---|
| T-TPT | Temperature at the subsea Xmas tree (TPT) | 91% |
| P-TPT | Pressure at the TPT | 82% |
| P-ANULAR | Pressure in the well annulus | 82% |
| P-MON-CKP | Pressure upstream of the production choke (PCK) | 65% |
| P-JUS-CKGL | Pressure downstream of the gas lift choke | 60% |
| T-JUS-CKP | Temperature downstream of the PCK | 50% |
| T-PDG | Downhole temperature (PDG) | 25% |
| ABER-CKP | Opening of the PCK | 25% |
| P-PDG | Downhole pressure (PDG) | 16% |
| QGL | Gas lift flow rate | 13% |
| all others | | < 5% |

The `ESTADO-*` variables are valve states (0, 0.5, 1): they are almost always constant, which is
normal for a state, but it means they carry information only when they change (for example the
closure of the DHSV).

**Consequence:** the core is about 6 variables. The others are often missing and must be declared as
missing in the TimeNet format, never filled.

## Instance length (seconds, i.e. rows at 1 Hz)
| Provenance | Minimum | Median | Maximum |
|---|---|---|---|
| Real | 5,215 (1.4 h) | 21,354 (5.9 h) | 768,314 (8.9 days) |
| Simulated | 12,999 | 29,299 (8.1 h) | 88,800 |
| Drawn | 14,400 | 86,400 | 345,601 |

**Consequence:** instances are too long for the model: windows are needed (for example 30-60
minutes), resampled to a lower rate.

## Seconds per label type
| Provenance | Normal | Transient | Steady |
|---|---|---|---|
| Real | 14.6 M | 11.0 M | 3.3 M |
| Simulated | 2.4 M | 12.7 M | 25.7 M |
| Drawn | 0.4 M | 2.5 M | 0.1 M |

In real instances the transient phase is well represented: there is material for early warning.
