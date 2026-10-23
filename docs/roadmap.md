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

1a. **Microbenchmarks** (`scripts/bench_spark.py`), which replace every `ESTIMATE` in
    `cirrus/perf/hardware.py`:
    - achievable DRAM bandwidth for streaming reads (nvbandwidth + a custom GEMV)
    - NVFP4 / FP8 GEMV at M = 1..16 against the bandwidth roofline
    - grid-barrier latency inside a persistent kernel across 48 SMs
    - L2 persisting-access window: how much of the L2 we can pin, and its hit bandwidth
    - GPU↔CPU round trip on unified memory (for CPU-side drafting)

1b. **Decode megakernel v0**: one persistent kernel per verify step covering all layers. GEMVs
    dequantize NVFP4 in registers, GDN state lives in fp32 registers/SMEM during the step, and
    there is a grid barrier per sync point. Check it against `cirrus.model` outputs.
    **Exit: ≥ 90% of the roofline prediction at τ=1** for the chosen preset.

1c. **Prefill path**: CUTLASS / cuBLASLt block-scaled NVFP4 GEMMs plus the fla chunked GDN
    kernel. **Exit: 16K tokens in ≤ 40 ms.**

1d. **Measure τ** (accepted tokens per step): run an existing small open model through our
    protocol on search-style prompts, and measure prompt-lookup acceptance on its outputs. This
    gives the copy-rate prior before cirrus exists.

**Decision gate:** pick `cirrus-s-1a` (~290M) or `cirrus-s-1a-mini` (~155M) using the measured
roofline and τ. The rule: take the largest model whose *measured* 32K-context throughput is
≥ 3,000 tok/s at the *measured* τ.

## Phase 2: tokenizer + data pipeline

- 32K BPE trained on the pretraining mix, digits split individually, protocol special tokens
  reserved (`cirrus/agent/protocol.py::SPECIAL_TOKENS`), ids roughly frequency-ordered so the
  FR-Spec draft vocabulary is a prefix.
- Streaming, deduplicated, tokenized shards for each mix in `docs/research/datasets.md`.
- Offline search environment: BM25 + small dense retriever over wiki-18, a FineWeb-Edu slice,
  a frozen fresh Common Crawl snapshot and BrowseComp-Plus, with a counterfactual rewriter.

## Phase 3: pretraining (≈1-1.5T tokens), which needs rented GPUs

Compute: 6·N·D ≈ 1.7-2.6e21 FLOPs. That is **~1.6-2.4k H100-hours** at 300 TFLOPS effective
(about 1-2 days on 64 H100s). The same run on one Spark would take more than a year, so the
Spark is for inference and evaluation, not pretraining. Plan: BF16 with FP8 matmuls, and
WSD (warmup-stable-decay) so mid-training can branch off the stable phase.

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
