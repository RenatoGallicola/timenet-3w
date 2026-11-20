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
1. **Must have**
   - Connector PR: event classification, transient intervals as annotations, real/simulated/drawn
     provenance, well id, split by well. *Done: PR #833.*
   - Windows read from the TimeNet dataset built by the connector, identical to the windows read
     from the Parquet files.
   - One trained model (OpenTSLM SoftPrompt, Llama 3.2 1B) on the classes with enough real
     instances.
   - Evaluation on wells held out from training.
   - Baselines: XGBoost on shape features and a classifier that sees only mean and standard
     deviation. *Done.*
   - Explanations built from computed features, checked automatically.
   - One perturbation test (time-shuffled signal).
   - Demo with precomputed examples, README, posts, submission.
2. **Then**
   - Flamingo and a comparison of the two architectures.
   - Training on simulated instances and testing on real ones for the rare classes.
   - All perturbation tests (signal swapped across classes, statistics removed).
   - How early events are recognized (metrics on transient windows).
   - Random split against split by well.
   - A large LLM without fine-tuning as a reference.
3. **If time allows**
   - Temporal localization as a separate task.
   - Explanations rewritten by an open model.
   - Live GPU demo, polished video.

## Schedule
| Day | Data and TimeNet | Model and demo |
|---|---|---|
| Fri 20 | PR fixes (done); windows read from TimeNet; first post | Modal and Hugging Face set up; OpenTSLM bridge; short test run on Modal |
| Sat 21 | Evaluation script, perturbation tests, random vs well split, baselines with simulated instances | **Full training** on A100; feature-based explanations; second training with explanations |
| Sun 22 | Final evaluation on the test wells, charts, README | Demo on Hugging Face |
| Sun 22 | Posts with metric screenshots, Discord post, **submission by 20:00** (both) | |
