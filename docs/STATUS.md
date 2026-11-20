# Status

Update it whenever a block starts or ends (see `CLAUDE.md`). Newest entries first.

## Now
About 2 days left: submit by **Sunday 22 at 20:00** (hard deadline: Monday 23 at 08:59 Italian
time).

| Who | Working on | Blockers |
|---|---|---|
| RenatoGallicola (Data and TimeNet) | **In progress:** #15 (lead-time analysis). Next: #4, #5 with XGBoost; #3 (perturbation tests) once an OpenTSLM model exists | None |
| ariannabalducci (Model and demo) | Window-to-OpenTSLM bridge; training on Modal | None: access to Llama 3.2 1B granted |

The open blocks and who holds them are the GitHub issues: `gh issue list`.

## Final deliverables
Not repo work, so not issues. Tick each one when done.

| Deliverable | Owner | When | Done |
|---|---|---|---|
| Post 1 on X and LinkedIn ("bringing 3W to TimeNet", with the PR link) | RenatoGallicola | Friday evening or Saturday morning | |
| Post 2 on X and LinkedIn (results and demo, with metric screenshots) | both | Sunday | |
| Post on Discord | ariannabalducci | Sunday | |
| Submission form (PR link, demo link, repo, post links) | RenatoGallicola | **Sunday by 20:00** | |
| Check whether the Nebius voucher arrived | ariannabalducci | as soon as possible | |

## Done
- 2026-11-20 (RenatoGallicola): **#2 done** (PR #16): `scripts/evaluate.py` scores any predictions
  file (`window_id`, `prediction`) the same way, with per-class counts, event-vs-normal detection,
  invalid answers counted as errors, and paired comparisons. Baseline predictions are in
  `results/predictions/`. **ariannabalducci: write OpenTSLM predictions in this format** (see
  `docs/DATA_CONTRACT.md`).
- 2026-11-20 (RenatoGallicola): **#12 done** (PR #14): baselines rerun on the TimeNet windows with
  the connector split. Macro-F1 shape 0.40 test / 0.22 val / 0.38 ± 0.13 CV; mean_std 0.26 test /
  0.16 val / 0.37 ± 0.15 CV. In the test split classes 2, 7 and 9 come from one well each and class
  8 from two, so per-class numbers rest on 1-3 events.
- 2026-11-20 (RenatoGallicola): **#1 done** (PR #13): `scripts/make_windows.py` reads the
  `petrobras/three-w` TimeNet dataset by default; `scripts/compare_windows.py` confirms the windows
  are identical to the Parquet ones (18,737 windows with the connector split: 12,827 / 2,690 /
  3,220). Commands to build the dataset and cut the windows are in `docs/DATA_CONTRACT.md` and
  `docs/SETUP.md`. **ariannabalducci: train on these windows** (`--splits
  splits/wells_connector.json`).
- 2026-11-20 (RenatoGallicola): **PR #833 fixes done** (3 commits, marked "Ready for review"): no
  absolute start time (3W states no time zone; the start stays as text in `metadata["start"]`); card
  with `source_revision`, `license_url` and the 2.0.0 citation; connector `README.md`. **Split
  decided: the connector's.**
- 2026-11-20 (ariannabalducci): **data contract** approved and implemented (`docs/DATA_CONTRACT.md`,
  `scripts/make_windows.py`): 18,737 30-minute windows from 39 real wells, in about 40 seconds. Well
  `WELL-00008` has no windows of the main classes.
- 2026-11-20 (ariannabalducci): **XGBoost baselines** (`scripts/baselines.py`, results in
  `results/baselines.json`). Macro-F1 over 6 classes on unseen wells: shape features 0.35 on test,
  0.39 ± 0.13 in cross-validation by well; mean and standard deviation only 0.32 on test, 0.40 ±
  0.14 in cross-validation. Two findings to report:
  1. each well has different sensors: without countermeasures the model learns which sensors exist,
     and on new wells it recognizes 0 flow-instability windows (with random sensor dropout: F1 0.6);
  2. mean and standard deviation are almost as good as shape features. OpenTSLM writes exactly those
     numbers into the prompt, so perturbation tests are required.
- 2026-11-20 (ariannabalducci): looked at the hackathon PRs on TimeNet. No other team uses 3W.
  Another team took the CARE wind turbines (PR #792). Maintainer TimoStoff reviews in detail (PR
  #788: named types, stricter validation). Some open maintainer PRs change the connector API (#821,
  #822, #827): PR #833 may need a rebase.
- 2026-11-20: **PR opened**: https://github.com/OpenTSLM/TimeNet/pull/833. The `area:connectors`
  label must be added by the maintainers: we lack the permission.
- 2026-11-17: **3W connector completed** in the fork, branch `feature/petrobras-three-w`
  (`datasets/petrobras/three_w/`). Id `petrobras/three-w`. Verified: ruff, ty (also Python 3.11), 11
  tests, checks of all connectors, full build (18 min, 1.1 GB: 2,228 records, 2,228 classification
  tasks, 1,142 localization tasks). The built dataset is in the `registry/` folder next to the repos
  on RenatoGallicola's computer.
- 2026-11-17: roles confirmed; `docs/SETUP.md` and `.env.example`; ariannabalducci collaborator on
  both repos.
- 2026-11-17: TimeNet environment working; 3W downloaded at commit `c86cc7c7` and explored
  (`docs/3W_EXPLORATION.md`); connector design approved (`docs/CONNECTOR_DESIGN.md`).
- 2026-11-12: registered for the hackathon with the 3W pitch.
