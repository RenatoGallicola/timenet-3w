"""Train OpenTSLM-SP on the 3W windows and predict with it, on Modal (issue #8).

Three commands, run from the repository root (the Modal and Hugging Face tokens are read from `.env`):

    # 1. Upload a windows folder to the Modal volume (once per folder; also for perturbed folders)
    uvx modal run scripts/opentslm_modal.py::upload --src ..\\data\\windows --name windows

    # 2. Train. A short run first, to measure throughput on the GPU:
    uvx modal run scripts/opentslm_modal.py::train --run-name smoke --max-steps 50
    uvx modal run scripts/opentslm_modal.py::train --run-name run1 --epochs 2

    # 3. Predict a split with a trained run; writes results/predictions/<out>.csv
    uvx modal run scripts/opentslm_modal.py::predict --run-name run1 --windows windows --split val

Training follows the plan of issue #8 and docs/DECISIONS.md:
- start from a published OpenTSLM-SP checkpoint (default `OpenTSLM/llama-3.2-1b-har-sp`, trained
  on multichannel accelerometer classification), with its LoRA adapters;
- train encoder, projector and LoRA (the LLM stays frozen), with the learning rates of the
  OpenTSLM curriculum, linear warmup and decay, gradient clipping;
- samples come from `scripts/opentslm_bridge.py`: present sensors only, random sensor dropout,
  class-balanced sampling;
- the answer is the class name alone (run 1), or with `--rationales` "<rationale> Answer: <state>",
  the rationale computed from the shown sensors by `scripts/rationales.py` (issue #9); predictions
  then hold the whole generated text, which `scripts/rationales.py --check-file` checks;
- after every epoch the model generates answers for the **validation** windows, scored with the
  parser of `scripts/evaluate.py`; the best epoch by validation macro-F1 is kept. The test windows
  are never used here: `predict --split test` runs once, at the end.

Checkpoints, configs and validation predictions are written to the `timenet-3w-runs` volume
(`/runs/<run name>/`). `predict` reads the checkpoint from there, or from a Hugging Face repo
with `--hf-repo`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Put the KEY=value lines of .env into os.environ (no override), without printing them."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#") and m[2]:
            os.environ.setdefault(m[1], m[2].strip("'\""))


_load_dotenv(REPO / ".env")

import modal  # noqa: E402  (after .env, so Modal finds its token)

OPENTSLM_COMMIT = "2968f4b891baab4307f7e9d0043e87677b593a30"
BASE_CHECKPOINT = "OpenTSLM/llama-3.2-1b-har-sp"
GPU = os.environ.get("TIMENET_GPU", "A100")  # A100-80GB for rationale runs that run out of memory
LABEL_TOKENS = 12
RATIONALE_TOKENS = 192  # the longest computed rationale is 53 words, numbers split into many tokens

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    # The exact environment of OpenTSLM's lockfile, then OpenTSLM itself at a pinned commit.
    .run_commands(
        "git clone https://github.com/StanfordBDHG/OpenTSLM /opentslm",
        f"cd /opentslm && git checkout {OPENTSLM_COMMIT}",
        "cd /opentslm && uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/req.txt",
        "uv pip install --system -r /tmp/req.txt",
        "uv pip install --system --no-deps /opentslm",
    )
    .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(REPO / "scripts" / "opentslm_bridge.py", "/root/opentslm_bridge.py")
    .add_local_file(REPO / "scripts" / "evaluate.py", "/root/evaluate.py")
    .add_local_file(REPO / "scripts" / "rationales.py", "/root/rationales.py")
)

app = modal.App("timenet-3w-opentslm", image=image)
data_volume = modal.Volume.from_name("timenet-3w-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("timenet-3w-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("timenet-3w-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_volume, "/runs": runs_volume, "/hf-cache": hf_cache}


# Locally the token comes from .env; in the container from this same secret.
HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


# --------------------------------------------------------------------------------------------
# Code that runs in the container
# --------------------------------------------------------------------------------------------

def _build_model(base_checkpoint: str, checkpoint_path: str | None = None):
    """OpenTSLM-SP with LoRA, loaded from a published checkpoint or from one of our runs."""
    from huggingface_hub import hf_hub_download
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    model = OpenTSLMSP(llm_id=OpenTSLM._get_base_llm_id(base_checkpoint), device="cuda")
    model.enable_lora()
    path = checkpoint_path or hf_hub_download(base_checkpoint, "model_checkpoint.pt")
    model.load_from_file(path)
    return model


def _collate(batch):
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=4)


def _generate(model, batch, max_new_tokens: int) -> list[str]:
    """Greedy answers for a batch. OpenTSLM pads on the right; generation needs left padding."""
    import torch

    embeds, mask = model.pad_and_apply_batch(batch)
    lengths = mask.sum(dim=1).tolist()
    width = embeds.size(1)
    left_embeds, left_mask = torch.zeros_like(embeds), torch.zeros_like(mask)
    for i, n in enumerate(lengths):
        left_embeds[i, width - n:] = embeds[i, :n]
        left_mask[i, width - n:] = mask[i, :n]
    out = model.llm.generate(
        inputs_embeds=left_embeds,
        attention_mask=left_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    return [t.strip() for t in model.tokenizer.batch_decode(out, skip_special_tokens=True)]


def _predict_dataset(model, ds, batch_size: int, max_new_tokens: int = 12) -> list[str]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    answers: list[str] = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    with torch.no_grad():
        for batch in loader:
            answers.extend(_generate(model, batch, max_new_tokens))
    return answers


def _score(answers: list[str], y, names: dict[int, str]) -> dict:
    import numpy as np
    from evaluate import parse_prediction
    from sklearn.metrics import f1_score

    pred = np.array([parse_prediction(a, names) for a in answers])
    labels = sorted(names)
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((pred == np.asarray(y)).mean()),
        "unparseable": float((pred == -1).mean()),
        "per_class_f1": dict(
            zip(map(str, labels), f1_score(y, pred, labels=labels, average=None, zero_division=0).round(3).tolist())
        ),
    }


@app.function(gpu=GPU, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 3600)
def train_remote(cfg: dict) -> dict:
    import time

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import get_linear_schedule_with_warmup

    sys.path.insert(0, "/root")
    from opentslm_bridge import ThreeWQADataset

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    windows = Path("/data") / cfg["windows"]
    out = Path("/runs") / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg["base_checkpoint"])
    eos = model.get_eos_token()
    mode = "auto" if cfg.get("rationales") else None
    train_ds = ThreeWQADataset(windows, "train", eos_token=eos, sensor_dropout=cfg["sensor_dropout"],
                               seed=cfg["seed"], rationales=mode)
    val_ds = ThreeWQADataset(windows, "val", rationales=mode)
    max_new = RATIONALE_TOKENS if mode else LABEL_TOKENS
    if cfg["max_val"]:
        idx = np.random.default_rng(0).permutation(len(val_ds))[: cfg["max_val"]]
        val_subset = torch.utils.data.Subset(val_ds, sorted(idx.tolist()))
        val_y = val_ds.y[sorted(idx.tolist())]
    else:
        val_subset, val_y = val_ds, val_ds.y

    # Class-balanced sampling (power 1 = every class equally often, 0 = natural frequencies).
    weights = train_ds.sample_weights() ** cfg["balance_power"]
    steps_per_epoch = cfg["steps_per_epoch"] or len(train_ds) // cfg["batch_size"]
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    epochs = -(-total_steps // steps_per_epoch)

    params = [
        {"params": list(model.encoder.parameters()), "lr": cfg["lr_encoder"], "weight_decay": 1e-2},
        {"params": list(model.projector.parameters()), "lr": cfg["lr_projector"], "weight_decay": 1e-2},
        {"params": model.get_lora_parameters(), "lr": cfg["lr_lora"], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    history, best, step = [], -1.0, 0
    started = time.time()
    for epoch in range(epochs):
        model.train()
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights), num_samples=steps_per_epoch * cfg["batch_size"], replacement=True,
            generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
        )
        loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=_collate)
        running, n = 0.0, 0
        for batch in loader:
            loss = model.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            running, n, step = running + loss.item(), n + 1, step + 1
            if step % 50 == 0:
                rate = step * cfg["batch_size"] / (time.time() - started)
                print(f"step {step}/{total_steps}  loss {running / n:.4f}  {rate:.1f} samples/s", flush=True)
            if step >= total_steps:
                break

        t0 = time.time()
        answers = _predict_dataset(model, val_subset, cfg["eval_batch_size"], max_new)
        scores = _score(answers, val_y, val_ds.names)
        entry = {"epoch": epoch + 1, "step": step, "train_loss": running / max(n, 1),
                 "val": scores, "val_seconds": round(time.time() - t0, 1),
                 "minutes": round((time.time() - started) / 60, 1)}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        model.store_to_file(str(out / "last.pt"))
        if scores["macro_f1"] > best:
            best = scores["macro_f1"]
            model.store_to_file(str(out / "best.pt"))
            ids = [val_ds.rows.window_id.iat[i] for i in (val_subset.indices if cfg["max_val"] else range(len(val_ds)))]
            pd.DataFrame({"window_id": ids, "prediction": answers}).to_csv(out / "val_predictions.csv", index=False)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        runs_volume.commit()
        if step >= total_steps:
            break

    return {"history": history, "best_val_macro_f1": best}


@app.function(gpu=GPU, volumes=VOLUMES, secrets=[HF_SECRET], timeout=3 * 3600)
def predict_remote(run_name: str, windows: str, split: str, base_checkpoint: str,
                   hf_repo: str, batch_size: int, rationales: bool = False) -> list[dict]:
    sys.path.insert(0, "/root")
    from opentslm_bridge import ThreeWQADataset

    if hf_repo:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(hf_repo, "model_checkpoint.pt")
    else:
        ckpt = str(Path("/runs") / run_name / "best.pt")
    model = _build_model(base_checkpoint, ckpt)
    ds = ThreeWQADataset(Path("/data") / windows, split, rationales="auto" if rationales else None)
    answers = _predict_dataset(model, ds, batch_size, RATIONALE_TOKENS if rationales else LABEL_TOKENS)
    print(json.dumps(_score(answers, ds.y, ds.names)), flush=True)
    return [{"window_id": w, "prediction": a} for w, a in zip(ds.rows.window_id, answers)]


@app.function(volumes=VOLUMES, secrets=[HF_SECRET], timeout=1800)
def push_to_hub_remote(run_name: str, hf_repo: str, private: bool, card: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hf_repo, private=private, exist_ok=True)
    run = Path("/runs") / run_name
    api.upload_file(path_or_fileobj=str(run / "best.pt"), path_in_repo="model_checkpoint.pt", repo_id=hf_repo)
    for name in ("config.json", "history.json"):
        api.upload_file(path_or_fileobj=str(run / name), path_in_repo=name, repo_id=hf_repo)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=hf_repo)
    return f"https://huggingface.co/{hf_repo}"


# --------------------------------------------------------------------------------------------
# Local entry points
# --------------------------------------------------------------------------------------------

@app.local_entrypoint()
def upload(src: str = "../data/windows", name: str = "windows"):
    """Upload a windows folder (npz, csv, meta.json) to the data volume as /data/<name>."""
    src_path = Path(src)
    files = sorted(p for p in src_path.iterdir() if p.suffix in {".npz", ".csv", ".json"})
    if not any(p.name == "meta.json" for p in files):
        raise SystemExit(f"{src_path} has no meta.json: is it a windows folder?")
    with data_volume.batch_upload(force=True) as batch:
        for p in files:
            batch.put_file(str(p), f"/{name}/{p.name}")
    print(f"uploaded {len(files)} files to timenet-3w-data:/{name}")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


@app.local_entrypoint()
def train(run_name: str, windows: str = "windows", base_checkpoint: str = BASE_CHECKPOINT,
          epochs: int = 2, max_steps: int = 0, steps_per_epoch: int = 0, batch_size: int = 8,
          eval_batch_size: int = 32, max_val: int = 0, lr_encoder: float = 2e-4,
          lr_projector: float = 1e-4, lr_lora: float = 1e-4, balance_power: float = 1.0,
          no_dropout: bool = False, rationales: bool = False, seed: int = 0):
    cfg = {
        "run_name": run_name, "windows": windows, "base_checkpoint": base_checkpoint,
        "epochs": epochs, "max_steps": max_steps, "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size, "eval_batch_size": eval_batch_size, "max_val": max_val,
        "lr_encoder": lr_encoder, "lr_projector": lr_projector, "lr_lora": lr_lora,
        "balance_power": balance_power, "sensor_dropout": not no_dropout, "rationales": rationales, "seed": seed,
        "repo_commit": _git_commit(), "opentslm_commit": OPENTSLM_COMMIT, "gpu": GPU,
    }
    result = train_remote.remote(cfg)
    out = REPO / "results" / "runs" / f"opentslm_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, **result}, indent=2))
    print(f"best validation macro-F1 {result['best_val_macro_f1']:.3f}; history in {out.relative_to(REPO)}")


@app.local_entrypoint()
def predict(run_name: str = "", windows: str = "windows", split: str = "val", out: str = "",
            base_checkpoint: str = BASE_CHECKPOINT, hf_repo: str = "", batch_size: int = 32,
            rationales: bool = False):
    if not run_name and not hf_repo:
        raise SystemExit("give --run-name or --hf-repo")
    if split == "test":
        print("NOTE: test split. Run it once, on the final model only (docs/DECISIONS.md).")
    rows = predict_remote.remote(
        run_name, windows, split, base_checkpoint, hf_repo, batch_size, rationales
    )
    name = out or f"opentslm_{run_name or hf_repo.split('/')[-1]}_{windows.replace('/', '_')}_{split}"
    path = REPO / "results" / "predictions" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions to {path.relative_to(REPO)}")


@app.local_entrypoint()
def push(run_name: str, hf_repo: str, public: bool = False):
    """Upload a run's best checkpoint, config and history to a Hugging Face model repo."""
    cfg_path = REPO / "results" / "runs" / f"opentslm_{run_name}.json"
    cfg = json.loads(cfg_path.read_text())["config"] if cfg_path.exists() else {}
    card = f"""---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B
tags: [time-series, opentslm, timenet, petrobras-3w]
---
# timenet-3w OpenTSLM-SP ({run_name})

OpenTSLM-SP (Llama 3.2 1B, LoRA) fine-tuned to classify 30-minute windows of offshore oil-well
sensor data from Petrobras 3W v2.0.0 (CC BY 4.0), read through the TimeNet `petrobras/three-w`
dataset. Code: https://github.com/RenatoGallicola/wellread

- Starting checkpoint: `{cfg.get('base_checkpoint', BASE_CHECKPOINT)}`
- OpenTSLM commit: `{cfg.get('opentslm_commit', OPENTSLM_COMMIT)}`
- timenet-3w commit used to train: `{cfg.get('repo_commit', 'unknown')}`
- Settings: see `config.json`; validation history: `history.json`.

Load with `scripts/opentslm_modal.py predict --hf-repo <this repo>`.
"""
    url = push_to_hub_remote.remote(run_name, hf_repo, not public, card)
    print(url)
