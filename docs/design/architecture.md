# cirrus-s-1a architecture

**Status:** spec v1, before any hardware run. The code in `cirrus/config.py` (shapes) and
`cirrus/model/` (reference math) is authoritative. The performance numbers come from the
analytical model in `cirrus/perf/` and are **predictions** until Phase 1 measures them on a
DGX Spark.

## 1. What the model is for

One job: take a search-engine query, run as many web searches and page opens as it needs, and
write an organized answer in which every claim is cited to a retrieved source. If the sources
don't support an answer, it says so.

That job shapes the whole design:

- **Knowledge lives in the sources, not the weights.** The model needs to be a strong *reader*,
  *planner* and *writer*, not an encyclopedia. That is why a 155M-parameter model is plausible
  here even though it would be a poor general chatbot.
- **Outputs copy a lot from inputs** (names, numbers, dates, quotes, URLs). That is good for
  faithfulness and, as section 5 shows, a big speed multiplier.
- **Inputs are long** (tens of thousands of tokens of results and pages), so attention cost per
  context token must be close to zero.

## 2. The shape: a YOCO decoder-decoder

```
 tokens ─► embed (32K vocab, tied) ─────────────────────────────────────────┐
                                                                            │
 SELF-DECODER  (12 layers, runs on every token, including prompt/results)   │
   [GDN, GDN, GDN, SWA] × 3                                                 │
     GDN = Gated DeltaNet: constant-size recurrent state, no KV cache       │
     SWA = sliding-window GQA attention, 1,024-token window                 │
                                                                            │
 SHARED KV  (computed ONCE per token from the self-decoder output)          │
   one K/V cache (2 KV heads × 128, FP8)  +  lightning-indexer keys (64-d)  │
                                                                            │
 CROSS-DECODER  (6 layers; for prompt tokens it is SKIPPED entirely)        │
   each layer: own Q projection → top-k sparse attention over the shared KV │
                                                                            │
 final norm ─► LM head (tied with embed)                                    │
 MTP draft module (FFN-only, reused twice, 8K "frequent token" draft head) ◄┘
```

| | value |
|---|---|
| Parameters | **155.2M** total (123.9M backbone, 25.2M tied embedding, 5.9M MTP) |
| d_model / layers | 768 / 18 (12 self + 6 cross) |
| GDN | 6 heads × (128 key, 128 value), short conv k=4, gated output |
| Attention | 6 query heads, 2 KV heads, head dim 128, QK-norm, partial RoPE (50%), head-wise sigmoid output gate |
| FFN | SwiGLU, hidden 2,048 |
| Sparse attention | top-2,048 tokens chosen by a 4-head, 64-dim indexer, shared by all cross layers |
| Vocab | 32,768 BPE, digits split, protocol tokens reserved |
| Inference precision | NVFP4 weights (4.5 bits/param incl. scales), FP8 KV, bf16 GDN state, fp32 accumulation |

### Why each piece is there

**Gated DeltaNet for most layers.** Each GDN layer keeps a fixed 128×128 matrix of "memory" per
head, updated with the delta rule. It *corrects* what it stores instead of only accumulating, and
a learned gate lets it forget quickly. Reading it costs the same at 100 tokens as at 64K tokens.
This is the linear-attention design used in Qwen3-Next (and, refined as KDA, in Kimi Linear),
among the strongest linear mixers at in-context recall. Weakness: like every fixed-size memory, it is lossy and can't copy
an arbitrary exact span from 30K tokens ago. That is why we don't rely on it alone.

**Sliding-window attention every 4th layer.** Exact attention over the most recent 1,024 tokens
covers local precision (finishing a quote, matching a number) at a fixed cost of about 0.5 MB per
layer.

**Shared KV + cross-decoder (YOCO).** This is the key structural choice. A normal transformer
keeps a KV cache *per layer*, and every decoded token re-reads all of them. At 32K context with
6 attention layers that would be about 100 MB per token, more than the whole model. YOCO keeps
**one** KV cache, written once per token, that every cross-decoder layer queries with its own
questions. That gives exact, softmax-attention retrieval over the entire context for the price
of one cache. Bonus: when reading search results (prefill), the cross-decoder isn't needed at all
for those tokens, because they only have to *contribute* keys and values. So prefill runs only
12 of the 18 layers.

**Sparse top-k attention with a lightning indexer (DeepSeek Sparse Attention style).** Even one
cache is 16 MB at 32K tokens. Without help, each of 6 cross layers would stream it every step
(about 100 MB). A tiny indexer instead scores every cached token with 64-dim FP8 keys (2 MB at
32K) and selects the 2,048 most relevant tokens once per step. All 6 cross layers then read only
those 1 MB of rows, which stay in L2 after the first layer. Result: **decode speed is almost
independent of context length** (−2% from 4K to 64K; with dense attention it roughly halves by 64K). The
indexer is trained by distillation from dense attention (KL loss, `indexer_weight`), first in a
dense warm-up and then in sparse mode.

**Tied 32K vocabulary.** The LM head is read in full every step, so vocab size is a direct speed
cost: 128K vocab × 768 dims would be ≈57 MB per step on its own. At 32K it is ≈14 MB. A small vocab
is also known to be fine at this model size.

**Gated attention and QK-norm.** A head-wise sigmoid gate on attention outputs, as in Qwen's
Gated Attention work, removes attention sinks and loss spikes. QK-norm keeps low-precision
training and inference stable.

**Partial RoPE.** Half of each head's dimensions are position-free. That helps long-context
retrieval and length generalization (64K trained, longer inference).

## 3. How a query flows through it

```
user query
  │  prefill ~300 tokens                                    < 1 ms
  ▼
model writes 2-4 searches  ──►  harness fires EACH search the instant
  │                             its <|/search|> streams out (overlap)
  │  <|run|>                                                ~10 ms of model time
  ▼
results arrive (snippets, deduped, numbered [1]..[n])       network: ~300-400 ms
  │  prefill ~2K tokens (self-decoder only)                 ~2 ms
  ▼
model decides: answer from snippets, search again, or <|open|> pages
  │                                                         network: ~400 ms if opened
  ▼
<|answer|> organized markdown, every claim cited [n]        ~500 tokens ≈ 110 ms
  │  citation grammar + audit run while streaming
  ▼
answer  (or <|insufficient|> with what was found)
```

The protocol is in `cirrus/agent/protocol.py` and the loop in `cirrus/agent/loop.py`. With the
estimated hardware numbers, a two-round query with page opens spends about **130 ms in the
model** and about 800 ms on the network (`scripts/perf_report.py --breakdown`). **The network is
the bottleneck, not the model.** That is why the loop overlaps everything and why training rewards
answering from snippets when they are enough.

## 4. How hallucination and accuracy are addressed

Honest framing first: no architecture makes a language model incapable of making things up.
What we can do is stack independent defences, so a fabricated claim has to get past all of them.

### Layer 1: a model that has little to hallucinate *from*

- At 155M parameters and ~1-1.5T reading-focused tokens (deliberately *fewer* than the 4-10T
  that recall-oriented small models use), the weights hold little world trivia. The model's
  strongest learned habit is reading the context.
- **Counterfactual training** is the central idea (docs/research/datasets.md §6-7). In
  mid-training (15%), SFT (25%) and RL, answer-bearing entities, numbers and dates in the
  sources are consistently swapped. The model is only rewarded when it reports what the
  *documents* say. In RL, reporting the original real-world fact gets a strong penalty. This
  directly trains "trust the sources over memory", which is exactly the behaviour needed for
  topics newer than the training data.
- **Fictional-world corpora** (invented entities with linked pages) make recall impossible, so
  multi-hop questions can only be solved by searching.

### Layer 2: trained to cite and to abstain

- Every SFT trajectory passes a rejection filter: correct answer, every cited sentence
  **entailed by its cited source** (NLI verifier), the teacher *fails* the question without tools,
  and multi-hop answers use ≥2 tool calls.
- 15% of SFT is **abstention**: the gold documents are removed from the index, and the target
  is `<|insufficient|>` plus what was found and what's missing.
- The RL reward combines answer correctness, citation precision and recall (NLI), and calibrated
  abstention: +1 for a correct abstain, −1 for a confident wrong answer, and a small cost for
  abstaining when the answer was findable.

### Layer 3: decode-time constraints (`cirrus/agent/citations.py`)

- **Citation grammar:** inside `[...]` the decoder can only produce ids of sources that were
  actually retrieved. The model *cannot* cite a source that doesn't exist.
- **Prompt-injection guard:** retrieved text is sanitized so a web page can't emit protocol
  tokens (e.g. a fake `<|answer|>`).

### Layer 4: verification before the user sees it

- **Structural audit** (implemented): flags sentences that make a claim with no citation, and
  citations to unknown ids. The harness can strip or mark them.
- **Groundedness check** (implemented as a lexical baseline, upgraded in Phase 5): each sentence
  is checked against the sources it cites, with numbers weighted double because a wrong number is
  the most common and most harmful search hallucination. The production version is a small NLI
  cross-encoder trained on AttributionBench + RAGTruth that runs on the Spark in parallel with
  generation and drops or flags unsupported sentences.

### Accuracy risks we are knowingly taking, and their mitigations

| Risk | Mitigation | How we'll know |
|---|---|---|
| A 155M model is a weak reasoner | Multi-hop is done by *searching again* (up to 4 rounds), not by thinking in-weights; short tool-dense SFT traces; distillation from a large teacher; RL only after SFT cold start (small search agents collapse under pure RL) | MuSiQue, FRAMES, BrowseComp-Plus |
| Linear layers are lossy at exact recall | Exact softmax attention over the *whole* context in all 6 cross layers, plus local SWA | Needle-in-haystack and copy tests in mid-training evals |
| Sparse top-k misses evidence | 2,048 tokens per step (≈ several passages); dense warm-up; ablate k ∈ {1K, 2K, 4K}. Option: per-layer selection at ~6% speed cost | Compare dense vs sparse on the same checkpoint |
| Shared selection across cross layers loses diversity | Same option as above; measure first | Same |
| Garbage in from the search engine | Multiple parallel queries, dedup, opens for detail, abstention when evidence conflicts or is thin | FreshQA, LiveBrowseComp |

## 5. How the speed target is addressed

### The budget

Decode on GB10 is **memory-bandwidth bound**. At 273 GB/s peak (≈218 GB/s assumed achievable),
each forward step can stream about 0.07 MB per microsecond. 3,000 tok/s means 333 µs per token.

Per verify step at 32K context, cirrus-s-1a streams **≈99 MB**:

| item | MB |
|---|---|
| backbone weights (NVFP4) | 69.8 |
| LM head (tied embedding, NVFP4) | 14.2 |
| MTP module + 8K draft head (then L2-resident across passes) | 6.9 |
| GDN recurrent state read + write | 3.5 |
| indexer keys (32K tokens) | 2.1 |
| SWA KV | 1.6 |
| top-k KV rows (then L2-resident across cross layers) | 1.0 |

That costs ≈543 µs including ≈80 µs of fixed synchronization. So:

- **No speculation (τ=1): ≈2,000 tok/s.**
- **3,000 tok/s needs τ ≥ ~1.65 accepted tokens per step.**

### Where τ comes from (speculative decoding, `cirrus/decoding/`, `cirrus/model/generate.py`)

1. **Prompt-lookup copy drafts (free).** When the last few generated tokens also appear in a
   source, the tokens that followed them there are proposed as a draft, up to 12 of them. Search
   answers quote names, figures and phrases constantly. Verifying 12 drafted tokens costs about
   the same bandwidth as decoding one, so a copied span is nearly free.
2. **MTP draft (cheap).** For novel text, one FFN-only multi-token-prediction module is applied
   twice to draft 2 tokens. It predicts only over the 8K most frequent tokens (FR-Spec), so its
   head stays in L2. Total cost is about 7 MB per step.
3. **Verification is exact.** Greedy draft-and-verify produces *exactly* the tokens plain greedy
   decoding would (there's a test for this). Speculation never changes the answer, only its speed.

Published prompt-lookup results on summarization and RAG suggest τ of about 2-3. We'll train the
model to quote and copy (good for faithfulness anyway), and Phase 1d measures τ before training.

### What each design choice is worth (τ=2, 32K context, model predictions)

| configuration | tok/s |
|---|---|
| **full design** | **3,684** |
| dense attention instead of shared sparse | 2,029 |
| CUDA graphs instead of a persistent megakernel | 2,376 |
| FP8 weights instead of NVFP4 | 2,299 |
| BF16 weights | 1,200 |
| the 290M `cirrus-s-1a-large` | 2,193 |

### Kernel plan (Phase 1)

- **One persistent "megakernel" per verify step**, spanning all 18 layers, the LM head and the
  MTP passes. A per-kernel launch-and-drain gap of a few µs × ~130 kernels would cost more than a
  quarter of our budget. Inside the megakernel, layers are separated by grid barriers, or better,
  fine-grained per-tile dependency counters (Hazy Research's low-latency megakernel / Mirage MPK
  approach).
- GEMVs at M = 1-13 rows dequantize NVFP4 in registers and stream weights at full bandwidth.
  Tensor cores are not the bottleneck in decode; bandwidth is.
- GDN state is kept in fp32 registers during the step and stored as bf16.
- **Prefill** uses block-scaled NVFP4 tensor-core GEMMs plus the flash-linear-attention chunked
  GDN kernel. The prediction is ≈17 ms for 16K tokens.

### What could go wrong, ranked by sensitivity (from the roofline model)

| assumption | baseline | if worse | tok/s at τ=2 |
|---|---|---|---|
| grid barrier latency | 1 µs | 3 µs | 3,684 → 2,964 |
| achievable DRAM bandwidth | 80% | 60% | 3,684 → 2,881 |
| host round trip per step | 15 µs | 50 µs | 3,684 → 3,461 |
| usable L2 for pinning | 24 MB | 8 MB | 3,684 → 3,593 |
| all of the above, moderately bad | | | 2,745 (still 3,431 at τ=2.5) |

These are exactly what the first hardware run measures (roadmap Phase 1a). If barriers are
slow, we cut syncs per layer by fusing and by using dependency counters. If τ is low, we raise
copy behaviour in training and add tree verification (lookup + MTP drafts verified together).

## 6. What exists today vs. next

| component | status |
|---|---|
| Config + parameter accounting | ✅ `cirrus/config.py` |
| Roofline model + report | ✅ `cirrus/perf/`, `scripts/perf_report.py` |
| Reference model (GDN, SWA, YOCO shared KV, sparse indexer, MTP, caches, spec decoding) | ✅ written, ⚠️ **not yet executed** (PyTorch wasn't installed in the dev container; `tests/test_model.py` runs on the Spark) |
| Agent protocol, streaming parallel loop, citation grammar + audit | ✅ tested |
| Spark microbenchmarks + megakernel | ⏭ Phase 1 |
| NLI citation verifier | ⏭ Phase 5 |
