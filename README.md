# timenet-3w

Explainable early warning of oil-well events with OpenTSLM and TimeNet.
Project for the [TimeNet Online Hackathon](https://challenge.timenet.ai/), Round 1 (November 2026).

> Work in progress. The project name is provisional.

## What we are building

Abnormal events in oil wells, such as hydrate blockages, flow instability or a spurious
safety-valve closure, cost production and raise safety and spill risks. We fine-tune
[OpenTSLM](https://github.com/OpenTSLM/OpenTSLM), a time-series language model, to recognize a
developing event from a well's sensor data while it is still in its transient phase, and to
explain its decision in physical terms.

- **Data:** [Petrobras 3W](https://github.com/petrobras/3W) v2.0.0 (CC BY 4.0), onboarded into
  [TimeNet](https://github.com/OpenTSLM/TimeNet) through a new dataset connector.
- **Model:** OpenTSLM on a 1B backbone, trained with explanations grounded in computed signal
  features.
- **Evaluation:** wells held out from training, classical and LLM baselines, transfer from
  simulated to real wells, and perturbation tests that check the model reads the signal.

## Repository layout

| Path | Content |
|---|---|
| `docs/` | Plan, decisions, data contract and status of the team |
| `CLAUDE.md` | Shared context for the Claude Code sessions of both team members |

More folders (training bridge, training, evaluation, demo) will be added as the work progresses.
The TimeNet connector lives in our fork of TimeNet and is submitted upstream as a pull request.

## License

Code in this repository is released under the MIT License. The 3W data keeps its own license
(CC BY 4.0, Petrobras).
