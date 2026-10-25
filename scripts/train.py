#!/usr/bin/env python3
"""Pretrain cirrus-s-1a on token shards from scripts/prepare_data.py.

    python scripts/train.py --data data/ablate --out runs/ablate-muon --tokens 3e9
    python scripts/train.py --data data/ablate --out runs/ablate-adamw --tokens 3e9 --optim adamw
    python scripts/train.py --data data/lean --out runs/lean --tokens 200e9 --compile

Built to run unattended for weeks on one consumer GPU:
  * resumes automatically from the newest checkpoint in --out (same command)
  * checkpoints atomically every --ckpt-minutes and on SIGINT / SIGTERM
  * the data order is a pure function of (seed, step), so a resume replays
    nothing and skips nothing
  * keeps a permanent checkpoint at the start of the WSD decay, so the run
    can be branched (longer, different anneal mix) without starting over
  * skips non-finite steps and stops cleanly if they repeat

fp32 master weights with bf16 autocast; Muon + AdamW (cirrus/train/optim.py).
Pretraining uses the dense cross-decoder (the sparse indexer arrives in
mid-training, DSA-style).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cirrus.config import ModelConfig, get_config  # noqa: E402
from cirrus.data.shards import TokenWindows, train_files, val_files  # noqa: E402
from cirrus.model.model import CirrusModel  # noqa: E402
from cirrus.train.optim import build_optimizers, decay_start, wsd_lr  # noqa: E402

# dense BF16 tensor TFLOPS (FP32 accumulate) for MFU: RTX 4090 spec, GB10 measured (run 1)
PEAK_TFLOPS = {"8.9": 165.2, "12.1": 93.0, "9.0": 989.0}
RESUME_KEYS = ("data", "seq", "batch_tokens", "tokens", "seed", "config", "optim")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory written by prepare_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, required=True, help="training tokens (sets the schedule length)")
    ap.add_argument("--config", default="cirrus-s-1a")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch-tokens", type=int, default=2**19)
    ap.add_argument("--micro", type=int, default=8, help="sequences per forward/backward")
    ap.add_argument("--optim", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=1000, help="steps")
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-lr-frac", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mtp", type=float, default=0.3, help="MTP loss weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=64, help="windows per val source")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--keep", type=int, default=3, help="rolling checkpoints kept (milestones are extra)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-overrides", default="{}", help='JSON ModelConfig overrides, e.g. {"d_model": 512}')
    return ap.parse_args(argv)


# ----------------------------------------------------------------- checkpoints
def ckpt_dir(out: Path) -> Path:
    return out / "ckpt"


def latest_ckpt(out: Path) -> Path | None:
    cks = sorted(ckpt_dir(out).glob("step_*.pt"))
    return cks[-1] if cks else None


def save_ckpt(out: Path, model, opts, step: int, args, cfg: ModelConfig, keep: int, milestone: str | None = None):
    d = ckpt_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "opts": [o.state_dict() for o in opts],
        "step": step,
        "args": vars(args),
        "cfg": asdict(cfg),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    path = d / (f"{milestone}.pt" if milestone else f"step_{step:08d}.pt")
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    if not milestone:
        for old in sorted(d.glob("step_*.pt"))[:-keep]:
            old.unlink()
    return path


# ------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, vals: dict[str, TokenWindows], n: int, micro: int, device: str) -> dict[str, float]:
    model.eval()
    res = {}
    for name, tw in vals.items():
        k = min(n, len(tw))
        losses = []
        for s in range(0, k, micro):
            x = torch.from_numpy(np.stack([tw.window(j) for j in range(s, min(k, s + micro))])).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                out = model(x, labels=x, mtp_weight=0.0)
            losses.append(out.metrics["ce"].float().item() * x.shape[0])
        res[name] = sum(losses) / k
    res["mean"] = sum(res.values()) / len(res)
    model.train()
    return res


# ------------------------------------------------------------------------ main
def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ck = latest_ckpt(out)
    state = torch.load(ck, map_location="cpu", weights_only=False) if ck else None
    if state:
        for k in RESUME_KEYS:
            if state["args"][k] != getattr(args, k):
                sys.exit(f"--{k.replace('_', '-')} differs from the run in {out} "
                         f"({state['args'][k]} vs {getattr(args, k)}); use a new --out")

    cfg = replace(get_config(args.config), sparse_topk=None, **json.loads(args.model_overrides))
    torch.manual_seed(args.seed)
    model = CirrusModel(cfg).to(dev)
    opts = build_optimizers(model, args.optim, args.lr, args.wd)
    step = 0
    if state:
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["opts"]):
            o.load_state_dict(s)
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        step = state["step"]
        print(f"resumed from {ck} at step {step:,}")
        del state
    fwd = torch.compile(model) if args.compile else model

    per_micro = args.micro * args.seq
    if args.batch_tokens % per_micro:
        sys.exit(f"--batch-tokens {args.batch_tokens} is not a multiple of micro*seq = {per_micro}")
    accum = args.batch_tokens // per_micro
    total = int(args.tokens // args.batch_tokens)
    d_start = decay_start(total, args.decay_frac)
    # windows of seq tokens; the model shifts labels internally (seq-1 predictions per window)
    train = TokenWindows(train_files(args.data), args.seq - 1, seed=args.seed)
    vals = {k: TokenWindows([p], args.seq - 1) for k, p in val_files(args.data).items()}
    n_params = model.num_params()
    flops_tok = 6 * n_params * 1.15
    peak = PEAK_TFLOPS.get(".".join(map(str, torch.cuda.get_device_capability()))) if dev.startswith("cuda") else None
    epochs = total * accum * args.micro / len(train)
    print(f"model {n_params / 1e6:.1f}M params | {args.optim} lr {args.lr} | {total:,} steps x "
          f"{args.batch_tokens:,} tokens ({accum} x {args.micro} x {args.seq}) | data {train.tokens / 1e9:.2f}B tokens "
          f"in {len(train.files)} shards, {epochs:.2f} epochs | decay from step {d_start:,}")

    stop = {"flag": False}

    def on_signal(sig, frame):
        print(f"\nsignal {sig}: checkpointing after this step", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log = open(out / "log.jsonl", "a")
    last_ckpt = time.time()
    t_log, tok_log, bad = time.time(), 0, 0
    last_metrics: dict = {}
    end = total if args.max_steps is None else min(total, args.max_steps)
    model.train()
    while step < end:
        if step == d_start and not (ckpt_dir(out) / "stable.pt").exists():
            save_ckpt(out, model, opts, step, args, cfg, args.keep, milestone="stable")
            print(f"saved stable-phase checkpoint at step {step:,} (branch point for anneals)")
        mult = wsd_lr(step, total, args.warmup, args.decay_frac, args.min_lr_frac)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * mult
        sums = {"loss": 0.0, "ce": 0.0, "mtp": 0.0}
        base = step * accum * args.micro
        for a in range(accum):
            x = torch.from_numpy(train.batch(base + a * args.micro, args.micro))
            x = x.pin_memory().to(dev, non_blocking=True) if dev.startswith("cuda") else x.to(dev)
            with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16):
                o = fwd(x, labels=x, mtp_weight=args.mtp)
            (o.loss / accum).backward()
            sums["loss"] += o.loss.detach().float() / accum
            sums["ce"] += o.metrics["ce"].float() / accum
            if "mtp" in o.metrics:
                sums["mtp"] += o.metrics["mtp"].float() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not torch.isfinite(gnorm) or not torch.isfinite(sums["loss"]):
            bad += 1
            for o_ in opts:
                o_.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite loss/grad, skipped ({bad} in a row)", flush=True)
            if bad >= 5:
                sys.exit("5 non-finite steps in a row: stopping. Resume restarts from the last checkpoint; "
                         "consider a lower --lr")
            step += 1
            continue
        bad = 0
        for o_ in opts:
            o_.step()
            o_.zero_grad(set_to_none=True)
        step += 1
        tok_log += args.batch_tokens

        if step % args.log_every == 0 or step == end:
            dt = time.time() - t_log
            tps = tok_log / dt
            rec = {"step": step, "tokens": step * args.batch_tokens, "loss": round(sums["loss"].item(), 4),
                   "ce": round(sums["ce"].item(), 4), "mtp": round(float(sums["mtp"]), 4), "lr": args.lr * mult,
                   "gnorm": round(gnorm.item(), 3), "tok_s": round(tps), "time": time.time()}
            if peak:
                rec["mfu"] = round(tps * flops_tok / (peak * 1e12), 3)
            if dev.startswith("cuda"):
                rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            eta_h = (end - step) * args.batch_tokens / tps / 3600
            mfu = f" | MFU {rec['mfu']:.0%}" if peak else ""
            print(f"step {step:>7,}/{end:,} | {rec['tokens'] / 1e9:7.3f}B tok | loss {rec['loss']:.4f} "
                  f"(ce {rec['ce']:.4f}) | lr {rec['lr']:.2e} | gnorm {rec['gnorm']:.2f} | {tps:,.0f} tok/s"
                  f"{mfu} | ETA {eta_h:.1f} h", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            last_metrics = rec
            t_log, tok_log = time.time(), 0

        if vals and (step % args.eval_every == 0 or step == end):
            ev = evaluate(model, vals, args.eval_windows, args.micro, dev)
            print("   val ce: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
            log.write(json.dumps({"step": step, "val": ev, "time": time.time()}) + "\n")
            log.flush()
            last_metrics["val"] = ev
            t_log = time.time()  # keep eval time out of tok/s

        if stop["flag"] or step == end or time.time() - last_ckpt > args.ckpt_minutes * 60:
            p = save_ckpt(out, model, opts, step, args, cfg, args.keep)
            print(f"checkpoint {p}", flush=True)
            last_ckpt = time.time()
            t_log = time.time()
            if stop["flag"]:
                break

    if step >= total:
        save_ckpt(out, model, opts, step, args, cfg, args.keep, milestone="final")
        print(f"done: {out / 'ckpt' / 'final.pt'}")
    log.close()
    return last_metrics


if __name__ == "__main__":
    main()
