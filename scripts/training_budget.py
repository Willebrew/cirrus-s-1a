#!/usr/bin/env python3
"""Training budget for cirrus-s-1a: GPU-hours and dollars per phase.

    python scripts/training_budget.py            # three scenarios, markdown
    python scripts/training_budget.py --spark-offload

Every assumption is a named field below; prices are from public trackers as
of October 2026 (see docs/plans/training-budget.md for sources) and should be
re-checked before booking capacity.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cirrus.config import get_config  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    # training throughput, tokens/s per H100 (155M params; see notes in the doc)
    tok_s_per_gpu: float
    long_ctx_efficiency: float  # mid-training at 8K-64K context runs slower
    # tokens
    ablation_tokens: float
    pretrain_tokens: float
    midtrain_tokens: float
    sft_tokens: float  # trajectories x epochs
    rl_gpu_hours: float  # rollouts + updates + NLI reward model
    contingency: float  # reruns, debugging, eval runs (fraction of GPU hours)
    gpu_price: float  # $/H100-hour
    # teacher-generated SFT data (API)
    sft_kept: int
    keep_rate: float
    traj_in_tokens: int  # summed over turns
    traj_out_tokens: int
    api_in_price: float  # $/M tokens
    api_out_price: float
    counterfactual_llm_tokens: float  # LLM-rewritten docs (most swaps are rule-based, CPU)
    judge_dollars: float  # LLM judge for long-form RL rewards
    search_api_dollars: float  # fresh-corpus snapshots + live evals
    storage_dollars: float  # ~5-10 TB of shards for ~3 months


SCENARIOS = [
    Scenario("optimistic", tok_s_per_gpu=350e3, long_ctx_efficiency=0.6, ablation_tokens=50e9,
             pretrain_tokens=1.0e12, midtrain_tokens=50e9, sft_tokens=3.6e9, rl_gpu_hours=100,
             contingency=0.2, gpu_price=2.20, sft_kept=200_000, keep_rate=0.5, traj_in_tokens=25_000,
             traj_out_tokens=1_500, api_in_price=0.15, api_out_price=0.80, counterfactual_llm_tokens=0.5e9,
             judge_dollars=100, search_api_dollars=250, storage_dollars=200),
    Scenario("base", tok_s_per_gpu=250e3, long_ctx_efficiency=0.6, ablation_tokens=100e9,
             pretrain_tokens=1.5e12, midtrain_tokens=75e9, sft_tokens=4.5e9, rl_gpu_hours=200,
             contingency=0.3, gpu_price=2.90, sft_kept=200_000, keep_rate=0.35, traj_in_tokens=25_000,
             traj_out_tokens=1_500, api_in_price=0.15, api_out_price=0.80, counterfactual_llm_tokens=1.0e9,
             judge_dollars=150, search_api_dollars=500, storage_dollars=350),
    Scenario("conservative", tok_s_per_gpu=150e3, long_ctx_efficiency=0.6, ablation_tokens=200e9,
             pretrain_tokens=1.5e12, midtrain_tokens=100e9, sft_tokens=6e9, rl_gpu_hours=400,
             contingency=0.4, gpu_price=4.00, sft_kept=200_000, keep_rate=0.25, traj_in_tokens=25_000,
             traj_out_tokens=1_500, api_in_price=0.27, api_out_price=1.10, counterfactual_llm_tokens=2.0e9,
             judge_dollars=300, search_api_dollars=1000, storage_dollars=700),
]


def gpu_hours(s: Scenario, spark_offload: bool) -> dict[str, float]:
    h = lambda tokens, eff=1.0: tokens / (s.tok_s_per_gpu * eff) / 3600  # noqa: E731
    phases = {
        "ablations": 0.0 if spark_offload else h(s.ablation_tokens),
        "pretrain": h(s.pretrain_tokens),
        "midtrain": h(s.midtrain_tokens, s.long_ctx_efficiency),
        "sft": 0.0 if spark_offload else h(s.sft_tokens),
        "rl": s.rl_gpu_hours,
    }
    phases["contingency"] = s.contingency * sum(phases.values())
    return phases


def dollars(s: Scenario, spark_offload: bool) -> dict[str, float]:
    gh = gpu_hours(s, spark_offload)
    traj = s.sft_kept / s.keep_rate
    teacher = traj * (s.traj_in_tokens * s.api_in_price + s.traj_out_tokens * s.api_out_price) / 1e6
    return {
        "gpu": sum(gh.values()) * s.gpu_price,
        "teacher_api": teacher,
        "counterfactual_llm": s.counterfactual_llm_tokens * s.api_out_price / 1e6,
        "judge_api": s.judge_dollars,
        "search_api": s.search_api_dollars,
        "storage": s.storage_dollars,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-offload", action="store_true",
                    help="run small ablations and SFT on the two DGX Sparks instead of rented GPUs")
    args = ap.parse_args()
    cfg = get_config("cirrus-s-1a")
    print(f"# cirrus-s-1a training budget ({cfg.total_params() / 1e6:.0f}M params"
          f"{', ablations + SFT on the Sparks' if args.spark_offload else ''})\n")
    print("| | " + " | ".join(s.name for s in SCENARIOS) + " |")
    print("|---|" + "---|" * len(SCENARIOS))
    ghs = [gpu_hours(s, args.spark_offload) for s in SCENARIOS]
    ds = [dollars(s, args.spark_offload) for s in SCENARIOS]
    for k in ghs[0]:
        print(f"| {k} (H100-h) | " + " | ".join(f"{g[k]:,.0f}" for g in ghs) + " |")
    print("| **GPU total (H100-h)** | " + " | ".join(f"**{sum(g.values()):,.0f}**" for g in ghs) + " |")
    print("| $/H100-h | " + " | ".join(f"${s.gpu_price:.2f}" for s in SCENARIOS) + " |")
    for k in ds[0]:
        print(f"| {k} ($) | " + " | ".join(f"${d[k]:,.0f}" for d in ds) + " |")
    print("| **total ($)** | " + " | ".join(f"**${sum(d.values()):,.0f}**" for d in ds) + " |")
    pre = [g["pretrain"] for g in ghs]
    print("\nPretraining wall-clock on 64 GPUs: " + ", ".join(f"{s.name} {p / 64:.0f} h" for s, p in zip(SCENARIOS, pre)))


if __name__ == "__main__":
    main()
