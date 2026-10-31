#!/usr/bin/env python3
"""Compare training runs by validation loss, per source, at matched token counts.

    python scripts/compare_runs.py runs/abl-muon runs/abl-adamw
    python scripts/compare_runs.py runs/abl-*  --at 0.5e9 1e9 2e9 3e9

Reads each run's log.jsonl (written by scripts/train.py). For every requested
token count it uses the latest eval at or before it, so runs with different
eval spacing still line up. Lower is better; the best run per row is starred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(run: Path) -> tuple[list[dict], dict]:
    evals, last = [], {}
    batch = None
    for line in (run / "log.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if "val" in rec:
            evals.append(rec)
        elif "tokens" in rec:
            last = rec
            batch = rec["tokens"] // rec["step"]
    for e in evals:  # eval records carry the step; convert to tokens
        e["tokens"] = e["step"] * (batch or 0)
    return evals, last


def at(evals: list[dict], tokens: float) -> dict | None:
    hit = [e for e in evals if e["tokens"] <= tokens]
    return hit[-1] if hit else None


def _fmt(tokens: float) -> str:
    return f"{tokens / 1e9:.2f}B" if tokens >= 1e7 else f"{tokens:,.0f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--at", type=float, nargs="*", default=None,
                    help="token counts to compare at (default: the shared final eval)")
    args = ap.parse_args(argv)

    data = {}
    for r in args.runs:
        p = Path(r)
        if not (p / "log.jsonl").exists():
            sys.exit(f"{p}/log.jsonl not found")
        data[p.name] = load(p)
    finals = [ev[-1]["tokens"] for ev, _ in data.values() if ev]
    if not finals:
        sys.exit("no eval records yet (the first one comes at --eval-every steps)")
    points = args.at or [min(finals)]
    sources = sorted({k for ev, _ in data.values() for e in ev for k in e["val"]} - {"mean"}) + ["mean"]
    names = list(data)
    width = max(12, *(len(n) for n in names))

    for tok in points:
        print(f"\n== val ce at {_fmt(tok)} tokens ==")
        print(f"{'source':20s}" + "".join(f"{n:>{width + 2}s}" for n in names))
        for src in sources:
            vals = []
            for n in names:
                e = at(data[n][0], tok)
                vals.append(e["val"].get(src) if e else None)
            best = min((v for v in vals if v is not None), default=None)
            cells = []
            for v in vals:
                cell = "-" if v is None else f"{v:.4f}" + ("*" if v == best and len(names) > 1 else " ")
                cells.append(f"{cell:>{width + 2}s}")
            print(f"{src:20s}" + "".join(cells))
    print()
    for n in names:
        last = data[n][1]
        if last:
            print(f"{n}: step {last['step']:,}, {_fmt(last['tokens'])} tokens, train ce {last['ce']:.4f}, "
                  f"{last['tok_s']:,} tok/s")


if __name__ == "__main__":
    main()
