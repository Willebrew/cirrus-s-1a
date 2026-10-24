#!/usr/bin/env python3
"""Training-throughput benchmark for cirrus-s-1a. Runs unchanged on an RTX 4090
(sm_89, under Linux/WSL2) or a DGX Spark (sm_121).

    pip install flash-linear-attention        # fused chunked Gated DeltaNet (Triton)
    python scripts/bench_train.py             # ~5-10 minutes
    python scripts/bench_train.py --seq 8192 --micro 1 2

What it does:
  1. checks fla's chunked Gated DeltaNet kernel against our exact reference
     recurrence (forward and gradients); training is impossible without it,
     because the reference runs as a Python loop over time
  2. times full training steps (forward + backward + AdamW) of the real
     155M model on random tokens, sweeping micro-batch size until OOM
  3. reports tokens/s, peak memory, model-FLOPs utilisation, and how long the
     planned token budgets would take on this one GPU

Pretraining runs the cross-decoder with dense attention (the sparse
indexer is introduced in mid-training, DSA-style), so sparse_topk is off here.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cirrus.config import get_config  # noqa: E402

# dense BF16 tensor throughput with FP32 accumulation, TFLOPS: spec for the
# 4090, our run-1 measured best GEMM for GB10
PEAK_TFLOPS = {"8.9": 165.2, "12.1": 93.0}


def check_fla() -> dict:
    import cirrus.model.layers as L

    if L._fla_chunk_gdr is None:
        return {"available": False}
    torch.manual_seed(0)
    B, T, H, D = 2, 256, 6, 128
    dev = "cuda"
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1)
    v = torch.randn(B, T, H, D, device=dev)
    g = -torch.rand(B, T, H, device=dev) * 0.5
    beta = torch.rand(B, T, H, device=dev)
    S0 = torch.randn(B, H, D, D, device=dev) * 0.1
    scale = D**-0.5
    out = {"available": True}
    # forward, fp32 reference vs bf16 kernel
    o_ref, s_ref = L.gated_delta_rule_ref(q, k, v, g, beta, scale, S0)
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    o_f, s_f = L._fla_chunk_gdr(qb, kb, vb, g, beta.bfloat16(), scale=scale, initial_state=S0,
                                output_final_state=True)
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()  # noqa: E731
    out["fwd_rel_err"] = rel(o_f, o_ref)
    out["state_rel_err"] = rel(s_f, s_ref)
    # gradients on a short sequence (the reference loop is slow)
    Ts = 64
    leaves = [t[:, :Ts].clone().requires_grad_(True) for t in (q, k, v)]
    o_ref, _ = L.gated_delta_rule_ref(*leaves, g[:, :Ts], beta[:, :Ts], scale, None)
    o_ref.square().sum().backward()
    g_ref = [t.grad.clone() for t in leaves]
    leaves2 = [t.detach().bfloat16().requires_grad_(True) for t in leaves]
    o_f, _ = L._fla_chunk_gdr(*leaves2, g[:, :Ts], beta[:, :Ts].bfloat16(), scale=scale)
    o_f.float().square().sum().backward()
    out["grad_rel_err"] = max(rel(a.grad, b) for a, b in zip(leaves2, g_ref))
    out["ok"] = out["fwd_rel_err"] < 3e-2 and out["state_rel_err"] < 3e-2 and out["grad_rel_err"] < 5e-2
    return out


def time_steps(model, opt, micro: int, seq: int, steps: int, warmup: int, vocab: int, mtp: float) -> dict:
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocab, (micro, seq), device="cuda")
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = model(ids, labels=ids, mtp_weight=mtp)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return {"micro": micro, "step_s": round(dt, 4), "tok_s": round(micro * seq / dt),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "loss": round(out.loss.item(), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--micro", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mtp", type=float, default=0.3, help="MTP loss weight (0 disables the MTP passes)")
    ap.add_argument("--compile", action="store_true", help="also try torch.compile")
    ap.add_argument("--allow-slow", action="store_true", help="run even without the fused GDN kernel")
    ap.add_argument("--out", default=str(ROOT / "results" / "local"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to this torch build")

    p = torch.cuda.get_device_properties(0)
    cap = f"{p.major}.{p.minor}"
    report = {"host": socket.gethostname(), "device": p.name, "capability": cap,
              "mem_gb": round(p.total_memory / 1e9, 1), "torch": torch.__version__, "seq": args.seq}
    try:
        import fla

        report["fla"] = fla.__version__
    except Exception:
        report["fla"] = None
    print(f"device {p.name} (sm_{p.major}{p.minor}, {report['mem_gb']} GB), torch {torch.__version__}, fla {report['fla']}")

    print("\n== fused Gated DeltaNet kernel vs exact reference ==")
    report["fla_check"] = check_fla()
    print(f"   {report['fla_check']}")
    if not report["fla_check"].get("ok") and not args.allow_slow:
        sys.exit("fused GDN kernel missing or wrong: `pip install flash-linear-attention`, or pass --allow-slow")

    cfg = replace(get_config("cirrus-s-1a"), sparse_topk=None)  # dense cross-decoder in pretraining
    from cirrus.model.model import CirrusModel

    flops_per_token = 6 * cfg.total_params() * 1.15  # +~15% attention / GDN mixing at 4K context
    peak = PEAK_TFLOPS.get(cap)
    rows = []
    for compiled in ([False, True] if args.compile else [False]):
        torch.manual_seed(0)
        model = CirrusModel(cfg).cuda().to(torch.bfloat16).train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
        fwd = torch.compile(model) if compiled else model
        print(f"\n== training step, seq {args.seq}, {'torch.compile' if compiled else 'eager'} ==")
        for micro in args.micro:
            try:
                r = time_steps(fwd, opt, micro, args.seq, args.steps, args.warmup, cfg.vocab_size, args.mtp)
            except torch.cuda.OutOfMemoryError:
                print(f"   micro {micro:2d}: out of memory")
                torch.cuda.empty_cache()
                break
            r["compiled"] = compiled
            r["mfu"] = round(r["tok_s"] * flops_per_token / (peak * 1e12), 3) if peak else None
            rows.append(r)
            print(f"   micro {micro:2d}: {r['tok_s']:>9,d} tok/s  step {r['step_s'] * 1e3:7.1f} ms  "
                  f"peak {r['peak_gb']:5.1f} GB  MFU {r['mfu']}  loss {r['loss']}")
        del model, opt, fwd
        torch.cuda.empty_cache()

    report["runs"] = rows
    if rows:
        best = max(rows, key=lambda r: r["tok_s"])
        report["best"] = best
        tps = best["tok_s"]
        print(f"\nbest: {tps:,} tok/s (micro {best['micro']}, {'compiled' if best['compiled'] else 'eager'})")
        print("time on this one GPU:")
        for label, toks in (("ablation, 3B tokens", 3e9), ("pretrain, 100B", 100e9), ("pretrain, 200B", 200e9),
                            ("pretrain, 500B", 500e9), ("pretrain, 1.5T", 1.5e12)):
            days = toks / tps / 86400
            print(f"   {label:22s} {days:7.1f} days" if days >= 1 else f"   {label:22s} {days * 24:7.1f} hours")
        report["days_per_100B"] = round(100e9 / tps / 86400, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"train-{socket.gethostname()}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
