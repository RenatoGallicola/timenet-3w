# Status

Update it whenever a block starts or ends (see `CLAUDE.md`). Newest entries first.

## Now
Sunday evening: submit as soon as the README is in (hard deadline: Monday 23 at 08:59 Italian time).

| Who | Working on | Blockers |
|---|---|---|
| RenatoGallicola (Data and TimeNet) | Posts and the submission form | None |
| ariannabalducci (Model and demo) | Posts and Discord; demo update optional | None |

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
- 2026-11-22 (RenatoGallicola): **#41 done** (PR #42): project name **WellRead** (repo and Space URLs
  unchanged). Demo restyled as a process-control console and republished; README rewritten for the
  judges (screenshot, why it matters, what we built, key results, citation), with the detailed method
  and results moved unchanged to `docs/RESULTS.md`. Repo description fixed (no early-warning claim).
- 2026-11-22 (RenatoGallicola): **#39 done** (PR #40): demo republished with the OpenTSLM perturbation
  results of #3 (validation and test) and the #9 outcome in the limits.
- 2026-11-22 (ariannabalducci): **#37 negative** (PR #38): per-well normalization of the statistics hurts XGBoost
  on the 3 folds by well (shape 0.317 -> 0.269; per-well mean/std 0.142 vs absolute 0.336), below the
  pre-registered 0.347, so no OpenTSLM run. Absolute levels carry physics that transfers across wells.
- 2026-11-22 (ariannabalducci): **#3 done** (PRs #35, #36): `perturb.py score`. OpenTSLM run 1 macro-F1 val / test:
  original 0.51 / 0.35, time shuffle 0.19 / 0.11, noise 0.21 / 0.11, no statistics 0.09 / 0.22, signal from
  another class 0.26 / 0.16, statistics from another class 0.11 / 0.12; every drop has a paired interval below
  zero. It reads the temporal shape; in conflicts it follows the statistics about 2:1 on validation, while on
  test the conflict is inconclusive. **#11 done**: README rewritten for the judges and audited against
  `results/`.
- 2026-11-22 (ariannabalducci): **#26 closed** by the exit rule: phase 3 (ensemble, threshold) not ready by
  Sunday morning, so the submission uses run 1; no second look at the test split. **#9 merged** (PR #33).
- 2026-11-21 (ariannabalducci): **#9 done** (branch `issue/9-rationales`): `scripts/rationales.py` writes
  rationales from the shown sensors (trend, step, oscillation, never the class) and checks every claim a
  model writes. CV by well, balancing 1.0, 1 epoch, **with vs without rationales**: macro-F1 0.363 ± 0.086
  vs 0.343 ± 0.029 (folds 0.387/0.435/0.267 vs 0.326/0.377/0.327), event recall 0.39 vs 0.43, **false
  alarms 0.28 vs 0.14**: no robust gain. Rationale check (CV mean): hallucinated sensors 0.05%, wrong
  direction 3%, numbers within tolerance 62%; it often says "stable" over a salient change. Class 4 is
  visible almost only in the PCK sensors (0.5% "stable" rationales with them, 95% without). Rationale runs
  need an A100-80GB (`TIMENET_GPU`).
- 2026-11-21 (ariannabalducci): **#26 phase 1 done** (branch `issue/26-opentslm-cv`): OpenTSLM on the 3 folds by
  well, 1 epoch, balancing 0.5 / 0.75 / 1.0: macro-F1 0.340 / 0.351 / 0.343 (XGBoost shape 0.317), event
  recall 0.15 / 0.18 / 0.43 at false alarms 0.10 / 0.14 / 0.14 (XGBoost 0.39 / 0.14). **Balancing 1.0
  kept** (see `docs/DECISIONS.md`). Fold 0 misses almost all events (class 4 from one unseen well).
- 2026-11-21 (ariannabalducci): **#29 done** (PR #30): OpenTSLM run 1 predictions on the 6 perturbed conditions,
  val and test; `original` identical to run 1.
- 2026-11-21 (RenatoGallicola): **#10 done** (PR #31): demo live at
  https://huggingface.co/spaces/timenet-3w/3w-on-timenet (free static Space). Explorer of test
  windows with both models' answers, test results, split leakage, lead time, perturbation check.
  To update it: `scripts/export_demo.py`, then upload `demo/` with a Write `HF_TOKEN` in `.env`.
- 2026-11-21 (RenatoGallicola): **#22 done** (PR #28): `--no-isolation` in the build commands,
  `.env.example` tracked (with the Modal tokens), Write token for Hugging Face, and `timenet` pinned to
  upstream `df1d9d67`, the reader of the connector version that the maintainer masquare pushed to the
  PR branch (new connector API, control schema v3). Windows from the rebuilt dataset are identical.
  **Both: run `uv sync` after pulling.**
- 2026-11-21 (RenatoGallicola): **#25 done** (PR #27): 3 folds by well over the train and
  validation wells (`splits/folds_by_well.json`, `scripts/make_folds.py`); every event class in every
  held-out fold. XGBoost shape reference on the folds: macro-F1 0.317 ± 0.031.
- 2026-11-21 (ariannabalducci): **OpenTSLM run 1 on test** (run once, PR #24): macro-F1 **0.349**
  against 0.403 for XGBoost shape (paired -0.05, CI -0.16 to 0.02); **false alarms 0.59**: 1,000 of
  1,805 normal test windows are called scaling in PCK, and no flow-instability window of the two
  test wells is recognized (validation class 4 came from one well). Per class 0: 0.43, 2: 0.88,
  4: 0.00, 7: 0.02, 8: 0.28, 9: 0.48. Run 2 (balancing power 0.5) scored 0.315 on validation, so
  run 1 stays the model. The validation gain does not transfer: the README must say so, and #3
  becomes the main evidence about what the model reads.
- 2026-11-21 (ariannabalducci): **OpenTSLM run 1** (#8, branch `issue/8-train-opentslm`):
  `scripts/opentslm_modal.py` trains on Modal from `OpenTSLM/llama-3.2-1b-har-sp` with the bridge of
  #7 (`scripts/opentslm_bridge.py`, branch `issue/7-opentslm-bridge`). **Validation macro-F1 0.510**
  after epoch 1 (XGBoost shape 0.221 on the same windows; paired bootstrap over instances +0.29, CI
  0.18-0.37); per class 0: 0.57, 2: 0.62, 4: 0.95, 7: 0.25, 8: 0.68, **9: 0.00**; false alarms 0.20
  (XGBoost 0.12). Epoch 2 overfits (0.36). About 12 min per epoch on an A100, within the free
  credits. Checkpoint in the private HF repo `timenet-3w/timenet-3w-opentslm`. Test not run yet.
  **RenatoGallicola: the model is ready for #3** (how to run it: comment on #3).
- 2026-11-20 (RenatoGallicola): **#3 tool ready** (PR #21; issue still open): `scripts/perturb.py`
  writes perturbed window folders (time shuffle, noise with the same statistics, signal or statistics
  from another class, no statistics) with the prompt statistics in `stats_mean`/`stats_std`. Checked
  on XGBoost (validation): shape 0.22 → 0.06 with time shuffle, mean_std unchanged.
  **ariannabalducci: the bridge must read `stats_mean`/`stats_std` when present** (comment on #7).
- 2026-11-20 (RenatoGallicola): **#5 done** (PR #20): simulated windows of classes 2, 8, 9 added to
  the XGBoost training, scored on real wells only. Validation 0.221 → 0.244, CV by well 0.316 → 0.347
  (± 0.09-0.13): a weak positive inside the noise; class 9 0.02 → 0.09. Simulated instances never
  record P-ANULAR (possible shortcut).
- 2026-11-20 (RenatoGallicola): **#4 done** (PR #18): same XGBoost model and windows, 5-fold CV on
  train+val: random split 0.84, by instance 0.71, by well 0.32 macro-F1. Chart in
  `results/split_leakage/split_leakage.png`, good for the posts.
- 2026-11-20 (RenatoGallicola): **#15 done** (PR #17): `scripts/lead_time.py` scores windows slid
  across the event onset (validation only, test frozen). XGBoost shape, class average: flagged as
  an event 0.18 at -30 min, 0.09 at onset, 0.37 at +15 min (false alarms 0.12). Clear early signal
  only for class 2 (0.00 before, 0.92 in the first 30 min, 2 instances) and partly class 9; class 4
  flat, class 7 never detected. Run it for OpenTSLM too before claiming early warning.
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
