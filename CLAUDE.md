# CLAUDE.md

Shared context for the Claude Code sessions of both team members.

- **Talk to the user in Italian.** Everything written to the repository and to GitHub (code,
  comments, docs, commit messages, issues, PRs) is in **English**.

## At the start of every session
A `SessionStart` hook (`.claude/hooks/session_start.py`) fetches the remote, fast-forwards `main`
if it is checked out, and shows you the "Now" and "Final deliverables" sections of
`docs/STATUS.md` on `origin/main` plus the open GitHub issues. If the
hook output is missing, do the same by hand. Then read `docs/DECISIONS.md` before proposing to
change anything already decided.

## Claim before you work (avoid overlaps)
1. Every block of work that lives in the repo is a **GitHub issue** on
   `RenatoGallicola/timenet-3w`. Posts and the submission form are not issues: they are tracked in
   the "Final deliverables" section of `docs/STATUS.md`.
2. **Before starting a block**, check the issues (`gh issue list`). If the block is assigned to the
   other member, stop and tell the user. If it is free, assign it to the current user
   (`gh issue edit <n> --add-assignee @me`), add it to "Now" in `docs/STATUS.md`, and **commit and
   push that change right away**. If the block has no issue yet, create one first.
3. **When a block is done**, close its issue with a short summary, update `docs/STATUS.md`, and
   push.
4. **When a new task comes up that will not be done right away** (a review comment, a bug found in
   the data, an idea for later, work for the other member), create an issue for it at once, with
   no assignee if it is not clear who will do it, and tell the user. That way the other member's
   session sees it at its next start.
5. Record every decision in `docs/DECISIONS.md` with its date and reason.

## Branches
- **Code and results of an issue** go on a branch named `issue/<number>-<short-name>` (for example
  `issue/3-perturbation-tests`), created from an up-to-date `main`. When the block is done, open a PR
  to `main` with `Closes #<number>` in the description, so GitHub closes the issue on merge. No
  review is required: whoever did the work merges it (squash merge), so nobody waits on the other.
  Before opening the PR, bring the branch up to date with `origin/main`.
- **Claims and decisions** (`docs/STATUS.md`, `docs/DECISIONS.md`) are committed **directly on
  `main`** and pushed at once, so the other member sees them right away. To do that while on an
  issue branch, switch to `main`, pull, commit, push, and switch back.
- The `SessionStart` hook fast-forwards `main` only when `main` is checked out, and always shows
  the status as it is on `origin/main`.

## Commits and PRs
- Small, frequent commits in Conventional Commits form (`feat: ...`, `docs: ...`), then push.
- **No Claude attribution:** no `Co-Authored-By: Claude` trailer and no "Generated with Claude Code"
  line. Commits and PRs show only the person who makes them. `.claude/settings.json` turns the
  automatic attribution off.

## The project in short
- **Hackathon:** TimeNet Online Hackathon, Round 1 (Aionic Labs + ETH Zurich Agentic Systems Lab).
  Deadline: **2026-11-22 23:59 PST = 2026-11-23 08:59 Italian time**. Internal target: submit on
  **Sunday 22 by 20:00**.
- **Use case:** detect and explain abnormal events in offshore oil wells with
  [Petrobras 3W](https://github.com/petrobras/3W) v2.0.0.
- **Judging criteria:** use case and impact; technical execution; data and TimeNet; results and
  evaluation; showcase and social engagement (the X and LinkedIn posts count).
- **Deliverables:** link to the PR on OpenTSLM/TimeNet that adds the dataset; link to a demo or
  site; this repo; public posts on X and LinkedIn with metric screenshots; a Discord post; the
  submission form.

Full plan: `docs/PLAN.md`. Data format between the two parts: `docs/DATA_CONTRACT.md`.

## Repositories
| Repo | Use |
|---|---|
| https://github.com/RenatoGallicola/timenet-3w | This repo: windows, TimeNet-to-OpenTSLM bridge, training, evaluation, demo, docs |
| https://github.com/RenatoGallicola/TimeNet | Fork of OpenTSLM/TimeNet: only the 3W connector, PR [#833](https://github.com/OpenTSLM/TimeNet/pull/833) |

## Constraints
- **Zero spend.** Free resources only: Modal ($30 a month per workspace, spend limit set to 0),
  Kaggle, Hugging Face ZeroGPU. No paid APIs. Nebius only if the voucher arrives and the team accepts
  its mandatory USD 25 top-up.
- **OpenTSLM uses bf16:** train on A100/H100 (Modal). Kaggle GPUs (T4/P100) have no bf16; they are
  fine for generating text with open models.
- **Secrets:** tokens and keys (Hugging Face, Modal) go only in `.env`, never in the repo.

## Rules of the TimeNet repo (for the connector)
- Follow `AGENTS.md` and the `.agents/skills/add-dataset-connector` skill of the TimeNet repo.
- Every source file is pinned to a commit with a SHA-256. Missing data is declared, **never filled**.
- Conventional Commits; never `--no-verify`.

## Verified facts (do not check them again)
- 3W v2.0.0: data license CC BY 4.0; 2,228 Parquet files (1.87 GB) in plain git without LFS; 1,119
  real instances from 40 wells, 1,089 simulated, 20 hand-drawn; 27 variables at 1 Hz; 10 classes;
  the transient label is class + 100. Real instances per class: normal 594, abrupt increase of BSW
  4, spurious closure of DHSV 22, severe slugging 32, flow instability 343, rapid productivity loss
  11, quick restriction in PCK 6, scaling in PCK 36, hydrate in production line 14, hydrate in
  service line 57. The timestamps carry no time zone.
- OpenTSLM has no TimeNet loader: we write the bridge. `OpenTSLMSP` hard-codes bf16 and eager
  attention; LoRA r=16 in the rationale stages. Ready checkpoints:
  `OpenTSLM/{llama-3.2-1b,...}-{tsqa,m4,har,sleep,ecg}-{sp,flamingo}`.
- OpenTSLM writes each series' mean and standard deviation into the prompt: the model may use those
  statistics instead of the signal. The baselines confirm it (see `docs/STATUS.md`), so perturbation
  tests are required.
- Reference paper on 3W: arXiv 2608.04041 (a 397B LLM given computed metrics, 35.1% top-1 over 9
  classes; a different protocol from ours).
