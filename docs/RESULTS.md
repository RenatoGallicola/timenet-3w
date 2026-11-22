# WellRead: full results

Every number of the project, with the method behind it. The short version is in the
[README](../README.md). Scripts are in `scripts/`, outputs in `results/`.

## Method in detail

**Baselines** (`scripts/baselines.py`): XGBoost on either scale-free shape features (trend,
change, roughness, autocorrelation) or only the mean and std of each sensor. Two findings shaped the rest of the
project:

1. **Shortcut: which sensors exist.** Each well has different sensors, so without countermeasures
   the model learns sensor presence. On new wells it then recognized 0 flow-instability windows.
   Random sensor dropout during training fixed it (class F1 0.50 on the test wells). Absolute levels
   are left out.
2. **Mean and std alone carry much of the signal**: 0.37 against 0.38 for shape features in 5-fold
   CV by well (but 0.26 against 0.40 on test). OpenTSLM writes exactly those numbers into its prompt, so it needs perturbation tests (below).

**OpenTSLM** (`scripts/opentslm_bridge.py`, `scripts/opentslm_modal.py`): OpenTSLM-SoftPrompt on
Llama 3.2 1B. Training starts from the published `OpenTSLM/llama-3.2-1b-har-sp` checkpoint, which was
trained on multichannel accelerometer classification. The bridge turns each window into an OpenTSLM
sample:
- each present sensor is z-normalized for the encoder;
- the prompt gives each present sensor's code, description, mean and std;
- the target is the class name.

The model is trained with fully class-balanced sampling and random sensor dropout. The encoder,
projector and LoRA adapters are trained; the LLM stays frozen. Two epochs on an A100 (about 12 minutes
each) on Modal's free credits; epoch 1 was kept, on validation macro-F1. The checkpoint and the
settings were chosen on validation wells only. The test wells were frozen from 2026-11-20 and scored once.

**Checked explanations** (`scripts/rationales.py`): for every training window a script writes what
the shown sensors did, never mentioning the class. For example:

> P-TPT drops by 52.5 bar (from 144.02 to 91.52 bar). P-JUS-CKGL drops by 10.1 bar (from 122.4 to
> 112.32 bar). P-MON-CKP drops by 12.2 bar (from 26.40 to 14.20 bar). The other sensors are stable.
> Answer: Hydrate in production line

A second OpenTSLM model is trained to write such a text and then the answer. Every sentence it
generates is parsed and re-measured on the window: does the sensor exist, is the direction right, is
the number within tolerance, did it call a salient change "stable"?

## Results

All numbers are macro-F1 over the 6 classes unless stated otherwise. **Event recall** is the share of
event windows flagged as any event. **False alarms** is the share of normal windows flagged as an
event. The scripts that produce each number are in `scripts/`, and the outputs are in `results/`.

### Split leakage (`results/split_leakage/`)

Same XGBoost model and windows, 5-fold cross-validation on train and validation:

| Split | Macro-F1 |
|---|---|
| Random windows | 0.84 |
| By instance | 0.71 |
| **By well** | **0.32** |

![Split leakage](../results/split_leakage/split_leakage.png)

### Test wells (scored once)

| Model | Macro-F1 | Event recall | False alarms |
|---|---|---|---|
| XGBoost, mean and std | 0.26 | 0.05 | 0.03 |
| XGBoost, shape features | **0.40** | 0.34 | **0.09** |
| OpenTSLM run 1 | 0.35 | **0.37** | 0.59 |

The paired bootstrap over instances gives OpenTSLM − XGBoost = −0.05 (CI −0.16 to 0.02). Per class
(F1, with the number of test wells and instances):

| Class | Wells | Instances | XGBoost | OpenTSLM |
|---|---|---|---|---|
| 0 Normal operation | 8 | 130 | 0.75 | 0.43 |
| 2 Spurious closure of DHSV | 1 | 3 | 0.99 | 0.88 |
| 4 Flow instability | 2 | 59 | 0.50 | 0.00 |
| 7 Scaling in PCK | 1 | 8 | 0.10 | 0.02 |
| 8 Hydrate in production line | 2 | 2 | 0.07 | 0.28 |
| 9 Hydrate in service line | 1 | 3 | 0.00 | 0.48 |

In the test split, classes 2, 7 and 9 come from one well each, so per-class numbers rest on 2 to 8
instances from one well. OpenTSLM's edge on classes 8 and 9 rests on 1 to 2 wells and is not evidence
that it generalizes. OpenTSLM called 1,000 of the 1,805 normal test windows "Scaling in PCK" and recognized no
flow-instability window. On validation it had scored 0.51 against 0.22 for XGBoost, but that gain did
not transfer: validation flow instability came from a single well.

### Cross-validation by well (3 folds over train and validation wells)

| Model | Macro-F1 (mean ± sd) | Event recall | False alarms |
|---|---|---|---|
| XGBoost, shape features | 0.317 ± 0.031 | 0.39 | 0.14 |
| OpenTSLM, balancing 1.0 | 0.343 ± 0.029 | 0.43 | 0.14 |
| OpenTSLM with explanations | 0.363 ± 0.086 | 0.39 | 0.28 |

The fold split is the one of `splits/folds_by_well.json`; the 5-fold figure of the split-leakage chart
(0.32) and of `results/baselines.json` (0.38 ± 0.13) use other folds. Class-balancing powers 0.5, 0.75
and 1.0 tie on macro-F1 (0.340, 0.351, 0.343). We kept 1.0 for its event recall (0.43 against 0.15 to
0.18 at similar false-alarm rates, 0.10 to 0.15). Explanations do not give a
robust gain (one fold worse) and double the false alarms, so the classifier stays label-only.

### Does OpenTSLM read the signal? (`results/perturbation/`)

`scripts/perturb.py` builds 6 versions of every window. OpenTSLM gets the series and the statistics
through two separate channels: the series goes to the encoder, normalized, and the mean and std go
into the prompt as text. Each perturbation breaks one channel:

| Condition | Series | Statistics in prompt | Validation | Test |
|---|---|---|---|---|
| original | own | own | 0.51 | 0.35 |
| time shuffle | own, time steps shuffled | own | 0.19 | 0.11 |
| noise | Gaussian noise with the window's mean and std | own | 0.21 | 0.11 |
| no statistics | own | none | 0.09 | 0.22 |
| signal from another class | other class | own | 0.26 | 0.16 |
| statistics from another class | own | other class | 0.11 | 0.12 |

Every drop from `original` has a paired bootstrap interval entirely below zero. On validation, time
shuffle costs −0.32 (CI −0.40 to −0.18), so the model uses the temporal shape. It is not just reading the
numbers in the prompt.

In the two conflict conditions, does the answer follow the window's own class or the class of the
part that was swapped in?

| Condition | Split | Follows the statistics | Follows the signal |
|---|---|---|---|
| signal from another class | validation | 0.45 | 0.19 |
| statistics from another class | validation | 0.41 | 0.16 |
| signal from another class | test | 0.25 | 0.18 |
| statistics from another class | test | 0.18 | 0.24 |

The rest of the answers name a third class. On validation, when the two channels disagree, the
statistics win about twice as often as the signal (the unperturbed answers matched the swapped-in
class only 7% of the time, so this is not chance). On test the conflict is inconclusive: the model
is weaker there, and its unperturbed answers already matched the swapped-in class 27% of the time.

The validation result matches an independent probe of the published `OpenTSLM/llama-3.2-1b-har-sp`
checkpoint on human activity recognition, which uses the same conditions
([opentslm-signal-probe](https://github.com/ariannabalducci/opentslm-signal-probe)). There, the
answer follows the statistics 2.2 to 2.5 times as often as the series, and shuffling time costs
about as much as replacing the series with noise. That model was trained on teacher rationales; ours
answers with the label only. The preference survives without rationales, but our model starts from
that checkpoint and may have inherited it. When the
signal carries no information, the model falls back to one default answer: "Normal operation" on
validation, "Scaling in PCK" on test. XGBoost shape features, which see only the series, drop from
0.22 to 0.06 under time shuffle on validation and from 0.39 to 0.14 on test (retrained, so 0.39 instead
of 0.40), and the statistics-only XGBoost does not move.

### Checked explanations (cross-validation mean)

| Check | Share of claims |
|---|---|
| Names a sensor that is not shown | 0.05% |
| Wrong direction of change | 3% |
| Fully correct (sensor, direction, numbers) | 62% |

Most errors are wrong numbers; the model also often calls a salient change "stable" (about 1,400
claims per fold). A property of the data shows the limit of these descriptions: in the training
windows of flow instability, the reference rationales say "all shown sensors are stable" for 0.5% of
the wells that record a sensor at the production choke (`P-MON-CKP` or `T-JUS-CKP`) and for 95% of
those that do not.

### Other results

- **Simulated instances** (`scripts/sim_augment.py`): adding 3W's simulated windows of classes 2, 8
  and 9 to the XGBoost training moves CV macro-F1 from 0.316 to 0.347 (± 0.09 to 0.13). The gain is
  weak and inside the noise. Simulated instances never record `P-ANULAR`, which is a possible
  shortcut.
- **Per-well normalization** (`scripts/well_norm.py`, XGBoost, 3 folds by well; rule fixed before
  the run in `docs/DECISIONS.md`): expressing each window's mean and std relative to its own well
  (median over all its windows, no labels) **hurts**. Shape features drop from 0.317 to 0.269 when the
  per-well levels are added. Relative mean and std reach only 0.14, while absolute mean and std reach
  0.34; spurious DHSV closure falls from 0.78 to 0.01-0.22. The absolute levels carry physics that
  transfers across wells, for example a pressure falling to a typical value when a valve closes, so
  they are not just a well-specific shortcut. A reference built from all of a well's windows also
  includes its events. Because XGBoost did not improve, the OpenTSLM run was not launched.
- **Lead time** (`scripts/lead_time.py`, XGBoost only, validation): windows slid across the event
  onset. A clear early signal appears only for class 2. We did not run it for OpenTSLM, so **we make
  no early-warning claim**.
