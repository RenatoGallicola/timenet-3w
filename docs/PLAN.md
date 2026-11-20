# Plan

Updated on 2026-11-20. The goal is to do **everything**, in the priority order below, so that a
complete project can be submitted at any moment.

## Roles
| Role | Responsibilities | Who |
|---|---|---|
| **Data and TimeNet** | 3W connector, tests, CI, PR and contact with the maintainers; data contract and split; evaluation, baselines and statistics | RenatoGallicola |
| **Model and demo** | Window-to-OpenTSLM bridge, explanations, training on Modal, demo | ariannabalducci |
| **Both** | Posts on X and LinkedIn, video, submission | |

After the connector, the Data and TimeNet member also takes whole blocks of the model work, chosen
from: strong baselines, perturbation tests, a second training run (Flamingo or sim-to-real), and
checks of explanation quality. The final evaluation on the test wells stays with Data and TimeNet,
so it is independent of whoever trains. Every block is a GitHub issue and must be claimed before
work starts (see `CLAUDE.md`).

Anyone starting on a new computer follows `docs/SETUP.md`.

The two parts run in parallel. Their contact point is `docs/DATA_CONTRACT.md`.

## Priority order
Updated after the external review of the evaluation plan (2026-11-20).

1. **Must have**
   - Connector PR. *Done: PR #833.*
   - Windows read from the TimeNet dataset. *Done: #1.*
   - Baselines on the connector split. *Done: #12.*
   - Evaluation script with count-aware, paired reporting and event-vs-normal detection (#2).
   - OpenTSLM run 1 (#7, #8): SoftPrompt from an existing `-sp` checkpoint, label-only answers,
     sensor dropout, balanced sampling. Settings chosen without the test wells.
   - Time-shuffle perturbation on the test windows, next to the mean/std baseline (#3).
   - Demo with precomputed examples (#10), README (#11), posts, submission.
2. **Then**
   - Lead-time analysis around event onset (#15), needed to claim early warning.
   - Random split vs split by well (#4) and baselines with simulated instances (#5): cheap, good
     charts for the posts.
   - The other perturbation tests (signal swapped across classes, statistics removed) (#3).
   - Template rationales whose cited numbers are checked automatically (#9); a second training run
     with rationales if run 1 is done by Saturday evening.
3. **If time allows**
   - Flamingo and a comparison of the two architectures.
   - A large LLM without fine-tuning as a reference.
   - Training on simulated instances and testing on real ones for the rare classes.
   - Temporal localization as a separate task.
   - Explanations rewritten by an open model; live GPU demo; polished video.

## Claims
Safe: a leakage-free benchmark held out by well on TimeNet; classical baselines around 0.35-0.40
macro-F1 with large variance across wells; the sensor-presence shortcut found and fixed; mean and
standard deviation carry most of the signal; event detection with its false-alarm rate; OpenTSLM's
numbers with per-class counts; the perturbation result, whichever way it goes.

Avoid: "early warning" without #15; "OpenTSLM beats the baselines" without a paired test; any claim
about classes 7 or 9 generalizing (one test well each); comparing directly with the 35.1% of arXiv
2608.04041 (different protocol); calling the explanations causal (at most: the numbers they cite
are correct).

## Schedule
| Day | Data and TimeNet | Model and demo |
|---|---|---|
| Fri 20 | PR fixes (done); windows read from TimeNet; first post | Modal and Hugging Face set up; OpenTSLM bridge; short test run on Modal |
| Sat 21 | Evaluation script, perturbation tests, random vs well split, baselines with simulated instances | **Full training** on A100; feature-based explanations; second training with explanations |
| Sun 22 | Final evaluation on the test wells, charts, README | Demo on Hugging Face |
| Sun 22 | Posts with metric screenshots, Discord post, **submission by 20:00** (both) | |
