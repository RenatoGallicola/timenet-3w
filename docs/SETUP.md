# Setup from scratch

How to prepare a new computer and start working on the project. Expected time: 45-60 minutes, plus
the wait for the Llama approval on Hugging Face.

The commands are for **Windows (PowerShell)**. They work the same on macOS and Linux unless stated.

---

## 1. Accept the GitHub invitations
Collaborators are invited to two repos. Accept both invitations (GitHub email, or
https://github.com/notifications):
- https://github.com/RenatoGallicola/timenet-3w — the project
- https://github.com/RenatoGallicola/TimeNet — the TimeNet fork with the connector

Without accepting them you cannot push.

## 2. Install the tools
| Tool | How | Check |
|---|---|---|
| Git | https://git-scm.com/downloads | `git --version` |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh`) | `uv --version` |
| Python | No manual install needed: uv downloads the right version | |
| VS Code + Claude Code | https://code.visualstudio.com, then the "Claude Code" extension | |
| GitHub CLI (needed for the issues) | https://cli.github.com, then `gh auth login` | `gh auth status` |

Set your git identity once:
```powershell
git config --global user.name "Your name"
git config --global user.email "your-github-email"
```

## 3. Clone the project repo
Pick a work folder, for example `Desktop\TimeNet`, and always work inside it:
```powershell
mkdir $HOME\Desktop\TimeNet
cd $HOME\Desktop\TimeNet
git clone https://github.com/RenatoGallicola/timenet-3w.git
```
The TimeNet fork is only needed by whoever works on the connector.

## 4. Create the accounts (all free)

### Hugging Face
1. Sign up at https://huggingface.co
2. Request access to https://huggingface.co/meta-llama/Llama-3.2-1B (button at the top of the
   page). Approval can take a few hours: do this first.
3. Accept the Gemma license at https://huggingface.co/google/gemma-3-1b-pt
4. Create a token: Settings → Access Tokens → New token, type **Read**.

### Modal (GPUs for training)
1. Sign up at https://modal.com with your GitHub account.
2. Add a card (Modal requires one), then **immediately** go to Settings → Usage & Billing and set
   the **spend limit to 0** (or the minimum allowed). You then use only the free $30 a month and
   never pay.
3. Use your own workspace: the $30 are per workspace, so do not join the other member's.
4. Connect the terminal with `uvx modal token new` (or `uv run modal token new` once `modal` is a
   project dependency).

### Kaggle (backup GPUs)
1. Sign up at https://www.kaggle.com
2. Settings → verify your phone number (GPUs are not available without it).

## 5. Put the tokens in `.env`
The repo has `.env.example`. Copy it to `.env` and fill in the values:
```powershell
cd $HOME\Desktop\TimeNet\timenet-3w
copy .env.example .env
notepad .env
```
`.env` is ignored by git: **never commit it**, and never write tokens in any other file.

## 6. Download 3W at the same commit
The whole team uses 3W v2.0.0 at commit `c86cc7c70063646910be7292de8a54a6fb2a8399`. Download it into
a `data` folder next to the repo, outside `timenet-3w`. It is about 1.9 GB:
```powershell
cd $HOME\Desktop\TimeNet
mkdir data
cd data
git -c core.autocrlf=false clone --filter=blob:none --sparse --no-checkout https://github.com/petrobras/3W.git 3w
cd 3w
git sparse-checkout set dataset
git checkout c86cc7c70063646910be7292de8a54a6fb2a8399
```
Check: the `dataset` folder must hold 10 subfolders (0-9) and 2,228 `.parquet` files. Then write the
path in `.env` as `THREEW_DATASET_DIR`, for example
`C:\Users\<you>\Desktop\TimeNet\data\3w\dataset`.

## 7. Start Claude Code
Open the `timenet-3w` folder in VS Code and open Claude Code. The first time, Claude Code may ask
you to approve the project settings: they hold the `SessionStart` hook that pulls the repo and shows
the status and the open issues. Approve them.

Claude finds the whole context in `CLAUDE.md`: deadlines, constraints, decisions, verified facts,
and the "claim before you work" rule.

## 8. Rules of every session
- **Claim before you work:** every block of work is a GitHub issue. Assign it to yourself and
  update `docs/STATUS.md` before starting, then push right away.
- **During the work:** small, frequent commits in the form `type: description`
  (e.g. `feat: add 3W window loader`).
- **At the end:** close the issue, update `docs/STATUS.md`, `git push`.
- **Decisions:** record them in `docs/DECISIONS.md` with date and reason.
