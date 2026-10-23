#!/usr/bin/env python3
"""Print the DGX Spark speed budget for cirrus-s-1a presets.

    python scripts/perf_report.py                 # all presets, markdown
    python scripts/perf_report.py --preset cirrus-s-1a --breakdown

Numbers come from the analytical roofline model in cirrus/perf; see
docs/design/inference.md for the assumptions and how to replace estimates
with measurements.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cirrus.config import PRESETS  # noqa: E402
from cirrus.perf.hardware import HARDWARE  # noqa: E402
from cirrus.perf.roofline import InferencePlan, decode_step, prefill  # noqa: E402

CONTEXTS = (4096, 16384, 32768, 65536)
TPS = (1.0, 2.0, 2.5, 3.0)


def decode_table(cfg, hw, base: InferencePlan) -> str:
    head = "| context | " + " | ".join(f"τ={t:g}" for t in TPS) + " | MB/step (τ>1) |"
    rows = [head, "|" + "---|" * (len(TPS) + 2)]
    for ctx in CONTEXTS:
        cells, mb = [], 0.0
        for t in TPS:
            plan = replace(base, tokens_per_step=t, use_mtp=t > 1)
            s = decode_step(cfg, hw, plan, ctx)
            mark = " ✅" if s.tokens_per_second >= 3000 else ""
            cells.append(f"{s.tokens_per_second:,.0f}{mark}")
            mb = s.total_dram_bytes / 1e6
        rows.append(f"| {ctx // 1024}K | " + " | ".join(cells) + f" | {mb:.0f} |")
    return "\n".join(rows)


def breakdown(cfg, hw, plan, ctx) -> str:
    s = decode_step(cfg, hw, plan, ctx)
    lines = [f"Step at {ctx // 1024}K context, τ={plan.tokens_per_step:g}: "
             f"{s.step_us:.0f} µs (mem {s.t_mem_us:.0f}, compute {s.t_compute_us:.0f}, fixed {s.t_fixed_us:.0f}), "
             f"{s.tokens_per_second:,.0f} tok/s, {s.bound}-bound", "",
             "| item | DRAM MB | L2 MB |", "|---|---|---|"]
    for k, v in sorted(s.dram_bytes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v / 1e6:.2f} | {s.l2_bytes.get(k, 0) / 1e6:.2f} |")
    return "\n".join(lines)


def latency_story(cfg, hw, plan, search_ms: float, fetch_ms: float) -> str:
    """One realistic two-round query, end to end."""
    p = replace(plan, tokens_per_step=2.5)
    tps = lambda ctx: decode_step(cfg, hw, p, ctx).tokens_per_second  # noqa: E731
    steps = []
    t = 0.0

    def add(name, ms):
        nonlocal t
        t += ms
        steps.append((name, ms, t))

    add("prefill query + system (300 tok)", prefill(cfg, hw, plan, 300).t_us / 1e3)
    add("write 3 parallel searches (45 tok)", 45 / tps(512) * 1e3)
    add(f"search API, parallel (assumed {search_ms:.0f} ms)", search_ms)
    add("prefill 18 results (2,000 tok)", prefill(cfg, hw, plan, 2000).t_us / 1e3)
    add("write 2 opens (12 tok)", 12 / tps(2600) * 1e3)
    add(f"fetch 2 pages, parallel (assumed {fetch_ms:.0f} ms)", fetch_ms)
    add("prefill 2 pages (8,000 tok)", prefill(cfg, hw, plan, 8000).t_us / 1e3)
    first = t + 1 / tps(10600) * 1e3
    add("write cited answer (500 tok)", 500 / tps(10600) * 1e3)
    rows = ["| phase | ms | cumulative ms |", "|---|---|---|"]
    rows += [f"| {n} | {ms:.1f} | {c:.1f} |" for n, ms, c in steps]
    model_ms = sum(ms for n, ms, _ in steps if "assumed" not in n)
    rows.append("")
    rows.append(f"First answer token at **{first:.0f} ms**; full answer at **{t:.0f} ms**, "
                f"of which the model itself is **{model_ms:.0f} ms** and the network is the rest.")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), action="append")
    ap.add_argument("--hardware", default="dgx-spark", choices=sorted(HARDWARE))
    ap.add_argument("--attention", default="sparse", choices=["sparse", "dense"])
    ap.add_argument("--launch", default="megakernel", choices=["megakernel", "cuda_graph"])
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--search-ms", type=float, default=350.0)
    ap.add_argument("--fetch-ms", type=float, default=450.0)
    args = ap.parse_args()

    hw = HARDWARE[args.hardware]
    plan = InferencePlan(attention=args.attention, launch=args.launch)
    names = args.preset or [n for n in PRESETS if not n.endswith("debug")]
    print(f"# Speed budget on {hw.name}\n")
    print(f"attention={args.attention}, launch={args.launch}, weights={plan.body_format}, "
          f"effective DRAM BW={hw.eff_dram_bw / 1e9:.0f} GB/s. τ = accepted tokens per verify step.\n")
    for n in names:
        cfg = PRESETS[n]
        print(f"## {n} ({cfg.total_params() / 1e6:.0f}M params, {cfg.n_layers} layers, d={cfg.d_model})\n")
        print("Decode tok/s:\n")
        print(decode_table(cfg, hw, plan) + "\n")
        p16 = prefill(cfg, hw, plan, 16384)
        print(f"Prefill 16K tokens: {p16.t_us / 1e3:.1f} ms ({16384 / (p16.t_us * 1e-6):,.0f} tok/s)\n")
        if args.breakdown:
            print(breakdown(cfg, hw, replace(plan, tokens_per_step=2.5), 32768) + "\n")
            print("End-to-end example query:\n")
            print(latency_story(cfg, hw, plan, args.search_ms, args.fetch_ms) + "\n")


if __name__ == "__main__":
    main()
