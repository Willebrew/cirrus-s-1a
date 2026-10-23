# Training data for cirrus-s-1a

Research snapshot: 2026-10-23. Sources were read via web search and Hugging Face dataset cards.
Items marked (≈) or *unverified* came from search snippets. **Re-check every license on the
dataset card before ingestion**; this document is a map, not legal clearance.

The guiding principle: cirrus-s-1a should have **weak parametric knowledge and strong reading**.
Every stage is chosen to reward extracting and synthesizing from provided documents, and to
punish answering from memory.

---

## 1. Pretraining corpora (quality per token, reading over trivia)

| Corpus | HF id | Size | License | Notes |
|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ≈1.3T tok (score≥3) | ODC-BY | Strongest English base for small models; explanatory text. |
| Smol-Data 100BT mixes (Feb 2026) | `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT` and siblings | ~100B each | ODC-BY | HF's ablation-tested mixes; ready-made recipe. |
| FinePhrase (2026) | `HuggingFaceFW/finephrase` | 486B tok (FAQ 148B, tutorial 147B, math 98B, table+QA 92B) | ODC-BY | FAQ and table+QA configs are question→answer-from-document format. |
| FinePDFs / FinePDFs-Edu | `HuggingFaceFW/finepdfs` | ≈3T tok (EN >1.1T) | ODC-BY | Long structured docs, good for long-context mid-training. |
| DCLM-baseline 1.0 | `mlfoundations/dclm-baseline-1.0` | ≈4T | CC-BY-4.0 | Pairs well with FW-Edu. |
| Nemotron-CC v2 / v2.1 | `nvidia/Nemotron-CC-v2(.1)` | 6.6T + 2.5T | NVIDIA Open Data License (gated) | Diverse-QA synthetic subset is QA-from-passage. Review license. |
| Ultra-FineWeb | `openbmb/Ultra-FineWeb` | ≈1T EN | Apache-2.0 card (FineWeb upstream ODC-BY) | Verifier-filtered FineWeb. |
| Dolma 3 (Olmo 3) | `allenai/dolma3_mix-6T-1025`, `dolma3_dolmino_mix-100B-1025`, `dolma3_longmino_mix-50B-1025` | 5.9T / 100B / 50B | ODC-BY | Dolmino (mid-train, QA-heavy) and Longmino (long-ctx) are drop-in. |
| Essential-Web v1.0 | `EssentialAI/essential-web-v1.0` | 24T tok with taxonomy labels | Apache-2.0 / ODC-BY (conflicting listings) | Filter to "reference/explanatory" content to build a reading-heavy slice. |
| **PleIAs SYNTH** | `PleIAs/SYNTH` | ≈75B tok | CC-BY-4.0 | **Most relevant to our size.** Synthetic, includes RAG exercises. Baguettotron (321M) is SOTA-for-size on ~200B SYNTH tokens. |
| Common Pile v0.1 | `common-pile/*` | 8TB | open / public domain | Cleanest IP story, lower quality per token. |

## 2. Real query distributions

| Dataset | Size | License | Use |
|---|---|---|---|
| Researchy Questions `corbyrosset/researchy_questions` | ≈96k high-effort Bing queries + decompositions + clicked docs | **CDLA-Permissive-2.0** | Best-licensed real query log. SFT/RL prompts. |
| Natural Questions | 307k | CC-BY-SA-3.0 | Prompts |
| Search Arena 24k `lmarena-ai/search-arena-24k` | 24k real conversations w/ search traces | prompts CC-BY-4.0 (**not** model outputs) | Prompts / eval |
| WildChat-1M | 1M (filter search intents) | ODC-BY | Prompt diversity |
| MS MARCO / ORCAS / v2.1 segmented (113.5M segments) | large | **Non-commercial** | Research & eval only |
| LMSYS-Chat-1M, AOL 2006 | — | restrictive / privacy-tainted | **Avoid** |

## 3. Multi-hop and agentic search QA

| Dataset | HF id | Size | License | Trajectories? |
|---|---|---|---|---|
| HotpotQA | `hotpotqa/hotpot_qa` | 113k | CC-BY-SA-4.0 | No |
| 2WikiMultiHopQA | `xanhho/2WikiMultihopQA` | 192k | Apache-2.0 | No |
| MuSiQue (+ unanswerable twins) | `dgslibisey/MuSiQue` | ≈25k | CC-BY-4.0 | No; great for abstention |
| ASearcher | `inclusionAI/ASearcher-train-data` | 70k | Apache-2.0 | QA (RL prompts) |
| InfoSeek | `Lk123/InfoSeek` | ≈50k *unverified* | Apache-2.0 | QA + research trees |
| DeepDive | `zai-org/DeepDive` | 3.2k QA + 858 traj | Apache-2.0 | **Yes** |
| OpenSeeker-v1 | `PolarSeeker/OpenSeeker-v1-Data` | 11.7k QA + traj | MIT | **Yes** |
| UltraData-SFT-Agent-2609 | `openbmb/UltraData-SFT-Agent-2609` (Search-Agent config) | 20k | Apache-2.0 | **Yes** |
| DR Tulu SFT/RL | `rl-research/dr-tulu-sft-data` | ≈16k | ODC-BY | **Yes, with citations** |
| MiroVerse v0.1 | `miromind-ai/MiroVerse-v0.1` | 100k+ | **CC-BY-NC** | Yes (non-commercial only) |
| Search-R1 data | `PeterJinGo/nq_hotpotqa_train` | ≈170k | inherits | RL prompts |
| Eval only | Bamboogle, FRAMES (`google/frames-benchmark`), SimpleQA, BrowseComp, BrowseComp-Plus (`Tevatron/browsecomp-plus`, fixed 100k-doc corpus) | — | various | Eval |

Most open trajectories come from 30B+ "thinking" teachers with very long traces. For a 0.3B model
we **re-render them into short, tool-call-dense trajectories**: keep queries, fetched evidence spans
and a cited answer; drop long reasoning.

## 4. Attribution, citation, abstention, freshness

| Dataset | Size | License | Use |
|---|---|---|---|
| LongCite-45k `zai-org/LongCite-45k` | 44.6k, sentence-level citations | Apache-2.0 | Mid-training / SFT citation format |
| ALCE `princeton-nlp/ALCE-data` | ≈3k | MIT (ELI5 Reddit caveat) | Citation precision/recall eval |
| ExpertQA, HAGRID | 2.2k / 2.6k | MIT / Apache | SFT + eval |
| AttributionBench `osunlp/AttributionBench` | ≈26k | Apache-2.0 | **Train the NLI citation verifier** |
| RAGTruth `wandb/RAGTruth-processed` | ≈18k, span-level hallucination labels | MIT | Verifier training |
| Faithfulness-QA (arXiv 2604.25313) | 99k entity-substituted QA | check | Counterfactual SFT |
| SQuAD 2.0 | 150k (≈50k unanswerable) | CC-BY-SA-4.0 | Abstention |
| FaithEval `Salesforce/FaithEval-*` | ≈4.9k | research *unverified* | Eval: unanswerable / counterfactual |
| CRAG | 4.4k + mock API | CC-BY-NC | Eval |
| FreshQA, RealTimeQA, LiveSearchBench, LiveNewsBench | rolling | various | Freshness eval |
| **LiveBrowseComp** `Forival/LiveBrowseComp` | 335 Q on last-90-day facts | MIT | **Primary "no-memory" eval** |

## 5. Search environments

**Offline (for SFT generation and RL; no per-query cost, reproducible):**
- Search-R1 wiki-18 corpus + E5 index (`PeterJinGo/wiki-18-corpus`, `PeterJinGo/wiki-18-e5-index`):
  standard but memorized by every teacher, so use it **only with counterfactual rewriting**.
- BrowseComp-Plus corpus + prebuilt BM25/dense indexes (`Tevatron/browsecomp-plus-indexes`).
- FineWeb-Edu 10-50BT slice with BM25 (bm25s / Pyserini) + small dense retriever.
- `inclusionAI/ASearcher-Local-Knowledge`, `Tevatron/webshaper-fineweb-1m-corpus`.
- **Fresh Common Crawl snapshots (CC-MAIN-2026-xx)**: our own post-cutoff corpus; the key freshness source.
- ZeroSearch-style LLM-simulated search: cheap supplement only (teacher knowledge leaks in).

**Live APIs** (prices from comparison sites, verify on vendor pages): Brave ≈$5/1k; Tavily $5-8/1k;
Exa ≈$7/1k with contents; Serper $0.30-1/1k (SERP only). The Bing Search API was retired 2025-08-11.
SearXNG is free but scrapes upstream engines (ToS + rate limits). **Do RL offline**; use live APIs to
snapshot fresh documents and for final evaluation. 1M rollouts × 4 searches ≈ $20k live.

## 6. Synthetic data recipes

1. **Teacher trajectory distillation** over our offline indexes with a strong open teacher
   (Qwen3-235B / GLM-4.x / Kimi-K2 / DeepSeek-V3.x / gpt-oss-120b) running our exact protocol
   (`cirrus/agent/protocol.py`). Question synthesis recipes: OpenSeeker (web-graph expansion +
   entity obfuscation), DeepDive (KG random walks), InfoSeek (research trees), WebShaper, TaskCraft,
   ASearcher.
2. **Small models need a cold start.** DGPO (arXiv 2508.20324) finds 0.5B search agents collapse
   under pure RL; SFT first, then teacher-guided RL. Pleias-RAG-350M (arXiv 2504.18225) shows
   mid-training on retrieval-with-quotes data gives native citation ability at 350M.
3. **Rejection sampling filter.** Keep a trajectory only if:
   (a) the answer matches gold (EM/F1 or judge);
   (b) every cited sentence is entailed by its cited source (NLI verifier);
   (c) the teacher **fails closed-book** on the question (it requires search);
   (d) multi-hop items use ≥2 tool calls.
4. **Cutoff-proof construction.** (i) QA only from documents newer than both teacher and student
   cutoffs; (ii) **entity-swap / counterfactual documents**: consistently replace answer-bearing
   entities, numbers and dates across a mini-corpus (NQ-Swap, DisentQA, ConflictBank, KnowPO);
   (iii) **fictional-world corpora**: generated wiki pages about invented entities, multi-hop
   questions over them.
5. **Abstention.** Unanswerable twins, gold-removed episodes (target: `<|insufficient|>` plus
   partial findings), MASH-style search-cost incentives.

## 7. Staged plan for a ~0.3B model

| Stage | Tokens | Mix (approx.) |
|---|---|---|
| **(a) Pretrain** | **1-1.5T** (deliberately below the 4-10T used by recall-heavy small models) | 35% FineWeb-Edu, 15% DCLM, 10% FinePDFs-Edu, 10% FinePhrase (FAQ/tutorial/table), 8% SYNTH, 7% Nemotron Diverse-QA (if license OK), 8% code (Stack-Edu), 5% math (FineMath), 2% multilingual. Low Wikipedia upsampling. |
| **(b) Mid-train** | 50-100B, context 8K→32K→64K | 30% long docs, 20% reading-comprehension formats, 15% citation-format (LongCite, HAGRID, SYNTH RAG, synthetic "N results → cited answer"), **15% counterfactual docs**, 10% tool-call transcripts in our protocol, 10% replay |
| **(c) SFT** | 150-300k trajectories (0.5-1.5B tok) | ≈40k open trajectories re-rendered + 100-200k teacher trajectories over offline indexes; **25% counterfactual / fictional**, **15% abstention** |
| **(d) RL** (GRPO/DAPO from SFT) | 20-50k prompts × 8 rollouts | Env: local BM25+dense over wiki-18 + FW-Edu slice + frozen fresh-CC + BrowseComp-Plus + per-episode counterfactual shards. Reward: correctness + NLI citation precision/recall + calibrated abstention (+1 correct abstain, −1 confident wrong) + format gate + mild search-count and latency penalty. **Strong negative reward when a counterfactual episode's answer matches the original (pre-swap) entity.** |

Evaluation suite: LiveBrowseComp, FreshQA, FaithEval-counterfactual, MuSiQue-Full, ALCE citation
F1, BrowseComp-Plus, SimpleQA (with search), FRAMES.

## License flags

- **Non-commercial**: MS MARCO family, ORCAS, MiroVerse, CRAG, ClueWeb22.
- **Gated/custom**: Nemotron-CC (NVIDIA ODL), LMSYS-Chat-1M.
- **ToS issues**: ELI5 (Reddit), Search Arena model outputs, SearXNG scraping.
- **Unconfirmed**: WildChat-4.8M, FaithEval, TaskCraft licenses; InfoSeek/TaskCraft sizes.
