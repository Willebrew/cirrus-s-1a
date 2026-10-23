# cirrus-s-1a

**s** for search, **1** for version 1, **a** for autoregressive.

A ~155M-parameter model that does one thing: take a search query, call web search and page
fetches as many times as needed, and write an organized answer in which every claim is cited to
a retrieved source. It abstains when the sources don't support an answer. The target is
**> 3,000 tokens/s on a single NVIDIA DGX Spark**, so that end-to-end latency is set by the
network, not the model.

## Design in one paragraph

A YOCO-style decoder-decoder. A 12-layer self-decoder, mostly **Gated DeltaNet** (constant-size
recurrent state) with a sliding-window attention layer every 4th layer, produces **one shared KV
cache**. Six cross-decoder layers query it with **DeepSeek-style top-k sparse attention**, so
decode cost barely grows with context. Weights are **NVFP4**. A single persistent CUDA
megakernel runs each decode step. **Prompt-lookup copy drafts** plus a tiny **MTP draft head**
let each step accept several tokens. Faithfulness comes from counterfactual training,
NLI-filtered citation data, abstention training, a decode-time citation grammar and post-hoc
verification. Full walkthrough: [`docs/design/architecture.md`](docs/design/architecture.md).

## Predicted speed on DGX Spark (roofline model, before hardware validation)

| | τ=1 (no speculation) | τ=2 | τ=2.5 |
|---|---|---|---|
| 4K context | ~2,050 tok/s | ~3,740 | ~4,680 |
| 32K context | ~2,010 tok/s | ~3,680 | ~4,600 |

τ = accepted tokens per verify step. Reproduce with `python scripts/perf_report.py --breakdown`.

## Layout

```
cirrus/config.py          model shapes + exact parameter accounting
cirrus/perf/              DGX Spark hardware spec + roofline model (decode, prefill)
cirrus/model/             PyTorch reference model: the spec the kernels must match
cirrus/decoding/          prompt-lookup drafter, greedy verification
cirrus/agent/             tool protocol, streaming parallel agent loop, citation guard, search backends
scripts/perf_report.py    speed budget tables
docs/research/            DGX Spark, architectures, datasets
docs/design/              architecture + inference design
docs/roadmap.md           phases and exit criteria
```

## Quick start

```bash
pip install -e ".[dev]"          # harness + perf model are stdlib-only
pytest -q                        # model tests auto-skip without torch
pip install -e ".[model]"        # on the Spark: enables tests/test_model.py
python scripts/perf_report.py --breakdown
```

## Status

Phase 0 (research + spec) is done. Next is Phase 1: prove the speed on DGX Spark hardware with
random weights, before any training. See [`docs/roadmap.md`](docs/roadmap.md).
