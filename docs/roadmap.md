# cirrus-s-1a roadmap

The order is deliberate: **prove the speed on real hardware with random weights before spending
a single GPU-hour on training.** Architecture changes are cheap now and very expensive later.

## Phase 0: research + spec (this commit) ✅

- [x] DGX Spark hardware/software research → `docs/research/dgx-spark.md`
- [x] Architecture survey → `docs/research/architectures.md`
- [x] Training data survey and staged plan → `docs/research/datasets.md`
- [x] Architecture spec → `docs/design/architecture.md`, `cirrus/config.py`
- [x] Analytical roofline model → `cirrus/perf/`, `scripts/perf_report.py`
- [x] PyTorch reference model (spec for kernels) → `cirrus/model/`
- [x] Agent protocol, streaming harness, citation guard, prompt-lookup drafter → `cirrus/agent/`, `cirrus/decoding/`

## Phase 1: prove the speed on a DGX Spark (gate before training)

Everything here runs with **random weights at the real shapes**. Speed does not depend on what
the weights have learned (except τ, which is measured separately in 1d).

1a. ✅ **Microbenchmarks** (`scripts/bench_spark.py`, run 1 on 2026-10-23; see
    `docs/research/spark-measurements.md`). They replaced every `ESTIMATE` in
    `cirrus/perf/hardware.py` except FP4 tensor throughput:
    - achievable DRAM bandwidth for streaming reads (nvbandwidth + a custom GEMV)
    - NVFP4 / FP8 GEMV at M = 1..16 against the bandwidth roofline
    - grid-barrier latency inside a persistent kernel across 48 SMs
    - L2 persisting-access window: how much of the L2 we can pin, and its hit bandwidth
    - GPU↔CPU round trip on unified memory (for CPU-side drafting)

1b. **Decode megakernel.** Run 1 showed a per-matrix CUDA-core GEMV reaches only 94 GB/s.
    Kernel v1 adds a tensor-core NVFP4 GEMV and a persistent whole-step weight stream (run 2
    pending). Full target: one persistent kernel per verify step covering all layers. GEMVs
    dequantize NVFP4 in registers, GDN state lives in fp32 registers/SMEM during the step, and
    there is a grid barrier per sync point. Check it against `cirrus.model` outputs.
    **Exit: ≥ 90% of the roofline prediction at τ=1** for the chosen preset.

1c. **Prefill path**: CUTLASS / cuBLASLt block-scaled NVFP4 GEMMs plus the fla chunked GDN
    kernel. **Exit: 16K tokens in ≤ 40 ms.**

1d. **Measure τ** (accepted tokens per step), `scripts/measure_tau.py`: open models answer
    real questions from real retrieved sources in our format. Replaying the prompt-lookup
    drafter over their greedy answers gives the exact copy-drafting τ; the MTP head is modeled on
    a grid. Each τ is converted to tok/s with the measured run-6 stream.

**Decision (made):** build `cirrus-s-1a` at ~155M params (the former "mini"). Its predicted
throughput is ≥ 3,000 tok/s at τ ≥ 1.65. Phase 1 confirms it on hardware; `cirrus-s-1a-large`
(~290M) is only reconsidered if the *measured* numbers show at least 1.8× headroom.

## Phase 2: tokenizer + data pipeline + trainer (code ready; runbook in `docs/plans/training-runbook.md`)

- [x] 32K byte-level BPE, digits split individually, 64 reserved ids (control + protocol
  tokens), all other ids frequency ordered so the FR-Spec draft vocabulary is the id prefix
  (`cirrus/data/tokenizer.py`, `scripts/train_tokenizer.py`). To run on a Spark.
- [x] Streaming, resumable, tokenized shards for the lean mix, with per-source held-out val
  (`cirrus/data/`, `scripts/prepare_data.py`). Documents are rendered source-first,
  answer-after. To run on the Sparks.
- [x] Pretraining loop for one GPU: Muon + AdamW, WSD with a saved branch point, bf16
  autocast, chunked LM-head loss, exact resume, per-source val (`scripts/train.py`).
- [ ] Offline search environment: BM25 + small dense retriever over wiki-18, a FineWeb-Edu slice,
  a frozen fresh Common Crawl snapshot and BrowseComp-Plus, with a counterfactual rewriter.

## Phase 3: pretraining on own hardware (lean recipe, 200B tokens on the RTX 4090)

**Decision (made):** train locally. Six 3B-token ablations on the 4090 first, then the
200B-token lean run (6-11 weeks depending on the measured tok/s); the Sparks prepare data,
generate teacher trajectories and evaluate in parallel. The rented-cluster budget below is
kept for reference, in case the ablations argue for the full 1-1.5T plan.

Rented budget (docs/plans/training-budget.md, `scripts/training_budget.py`): **≈2,760 H100-hours and
≈$12.6k all-in in the base case** (range ≈$5.6k-$33k). Pretraining is ~60% of the GPU hours,
about 26 hours on 64 H100s. A single Spark would take more than a year, so the Sparks handle
inference, evaluation and small ablations. Before booking: measure training tokens/s on one
rented H100 (the most uncertain input). Plan: BF16 with FP8 matmuls, and WSD
(warmup-stable-decay) so mid-training can branch off the stable phase.

## Phase 4: mid-training (50-100B tokens)

Long context 8K→32K→64K, citation formats, **15% counterfactual documents**, and protocol
transcripts. Dense-attention warmup of the sparse indexer (KL to dense attention), then a switch
to top-k sparse training.

## Phase 5: SFT (150-300k trajectories)

Teacher-distilled trajectories over offline indexes in our exact protocol, filtered by rejection
sampling: correct answer, NLI-entailed citations, teacher fails closed-book, and ≥2 calls for
multi-hop. 25% counterfactual/fictional, 15% abstention. Tool-result tokens are loss-masked.

## Phase 6: RL (GRPO/DAPO)

Offline environment, reward = correctness + citation precision/recall (NLI) + calibrated
abstention + format gate − search-count/latency penalty, and a strong penalty when a
counterfactual episode's answer matches the pre-swap fact.

## Phase 7: quantize, ship, evaluate

- NVFP4 QAT in the last ~5% of SFT/RL (weights), FP8 KV, bf16 GDN state.
- Train the FR-Spec draft vocab and MTP head on the final model's own outputs (self-distill).
- Evals: LiveBrowseComp, FreshQA, FaithEval-counterfactual, MuSiQue-Full, ALCE citation F1,
  BrowseComp-Plus, SimpleQA-with-search, plus **measured** tok/s and query latency on Spark.
