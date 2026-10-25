# Training runbook (own hardware)

The order of operations for the lean recipe on two DGX Sparks and one RTX 4090
(`docs/plans/training-budget.md` has the reasoning and the timeline). Every
long-running step below is resumable: rerun the same command after a crash,
reboot or Ctrl-C.

| step | where | command | time |
|---|---|---|---|
| 0. measure | 4090 | `python scripts/bench_train.py` | ~10 min |
| 0. measure | Spark | `python scripts/measure_tau.py` | ~1-2 h |
| 1. tokenizer | Spark A | `python scripts/train_tokenizer.py` | ~1-2 h |
| 2. ablation data | Spark A | `python scripts/prepare_data.py --tokens 3e9 --out data/ablate` | ~1-3 h |
| 3. full data | Spark A + B | `python scripts/prepare_data.py --tokens 200e9 --out data/lean` | ~1-3 days, download-bound |
| 4. ablations | 4090 | `python scripts/train.py --data data/ablate --out runs/abl-muon --tokens 3e9` (and variants) | ~43 h each at the measured 19K tok/s (eager) |
| 5. pretrain | 4090 | `python scripts/train.py --data data/lean --out runs/lean --tokens 200e9 --compile` | 6-11 weeks |

Steps 2-3 run while step 4 trains; the full data set is only needed at step 5.

## Setup

```bash
git clone <repo> && cd cirrus-s-1a
pip install -e ".[train]" zstandard      # torch, flash-linear-attention, datasets, tokenizers
huggingface-cli login                    # optional: higher Hub rate limits for streaming
```

On the 4090, Triton needs Linux: native Linux, or WSL2 on Windows (keep `data/` on the
Linux filesystem, not `/mnt/c`, or reads crawl). Disk: the full lean set is ~400 GB of
uint16 shards (2 bytes/token), the ablation set ~6 GB, checkpoints ~2 GB each.

## 1. Tokenizer

`scripts/train_tokenizer.py` trains a 32,768-token byte-level BPE on ~4 GB sampled from the
mix in proportion to its weights, then renumbers ids by frequency on a separate held-out
slice. Properties the rest of the system relies on, all covered by `tests/test_data.py`:

- ids 0-63 are reserved: `<|pad|>`=0, `<|bos|>`=1, `<|eos|>`=2, the protocol tokens of
  `cirrus/agent/protocol.py`, and spare `<|reserved_i|>` slots for later formats
- digits are always single tokens (prices, dates, quantities: the facts a search answer
  must copy exactly)
- every other id is frequency ordered, so the MTP head's FR-Spec draft vocabulary is just
  the id prefix `[0, 8192)`

Commit `tokenizer/cirrus-32k.json` and its `.meta.json`: every shard records the tokenizer
hash, and `prepare_data.py` refuses to mix shards from two tokenizers.

## 2-3. Data

`scripts/prepare_data.py` streams each source from the Hub, renders rows into documents,
tokenizes and writes ~1 GB shards. Sources are in `cirrus/data/sources.py`; the lean mix:

| source | share | rendering |
|---|---|---|
| PleIAs SYNTH (English) | 30% | `Source:` seed section, `Question:`, `Answer:` (reasoning drafts dropped: they cost decode latency) |
| FineWeb-Edu, score ≥ 3.5 | 25% | plain text |
| FinePhrase FAQ | 20% | the source document, then the FAQ written from it |
| FinePhrase table | 10% | the source document, then a table and a QA pair |
| Dolmino QA (wiki→RCQA, Nemotron synth QA, Reddit flashcards) | 10% | as shipped |
| FinePhrase math | 5% | the source document, then a worked problem |

The common thread is *document first, answer after*: the model learns to answer from text
it was just shown, which is the skill a search model needs and the opposite of answering
from memory.

Workers split each dataset by file (nobody downloads rows they discard) and checkpoint the
stream position with every shard, so a resume jumps straight back to where it stopped.
Rank 0 of each source holds out the first ~2M tokens as `val/<source>.bin`; the trainer
reports validation loss per source, which is what the ablations compare.

**Memory.** `--workers` sets how many slices the data is split into (locked once an `--out`
directory exists, since each slice's resume point lives in its own files); `--procs` sets how
many run at once and can change on every run. By default `--procs` fits `--max-mem-gb 20` at
~2 GB per process (shards stream to disk; the remaining cost is one decoded parquet row group
of the columns we render). Each progress line prints that worker's RSS: if it runs higher on
some source, lower `--max-mem-gb` or pass `--procs` explicitly.

To split step 3 across both Sparks, give each a disjoint `--sources` list with otherwise
identical arguments, then merge the `train/` and `val/` directories:

```bash
# Spark A
python scripts/prepare_data.py --tokens 200e9 --out data/lean --sources synth,finephrase_faq,finephrase_math
# Spark B
python scripts/prepare_data.py --tokens 200e9 --out data/lean --sources fineweb_edu,finephrase_table,dolmino_qa
```

## 4. Ablations (decide before the long run)

Each ablation is 3B tokens: about 43 hours on the 4090 at the measured 19,241 tok/s (eager, micro 2).
Compare `val ce` per source at equal tokens (all runs log to `runs/<name>/log.jsonl`).

| # | question | runs |
|---|---|---|
| 1 | Muon vs AdamW for this architecture | `--optim muon` vs `--optim adamw` |
| 2 | learning rate | best optimizer at `--lr 1e-3`, `2e-3`, `4e-3` |
| 3 | sequence length | `--seq 2048` vs `4096` (speed vs long-range reading) |
| 4 | MTP weight | `--mtp 0.3` vs `0.1` (draft quality vs main loss) |
| 5 | mix | the lean mix vs more FineWeb-Edu (needs a second `--out` from `prepare_data.py`) |
| 6 | scale check | `--model-overrides '{"d_model": 640}'` to see the loss-vs-size slope |

## 5. Pretraining

```bash
python scripts/train.py --data data/lean --out runs/lean --tokens 200e9 --compile \
    --micro <largest that fits, from bench_train.py> [winning ablation flags]
```

- **Schedule:** warmup-stable-decay. At the start of the decay (last 20%) the trainer saves
  `ckpt/stable.pt` permanently: if the ablations or evals later say "train longer" or
  "anneal on a different mix", branch from there instead of restarting.
- **Checkpoints:** every 30 minutes and on Ctrl-C / SIGTERM, atomically, last 3 kept.
  Resume by rerunning the identical command; changing the schedule-defining flags
  (`--tokens`, `--seq`, `--batch-tokens`, `--data`, `--optim`, `--seed`, `--config`) is
  refused, since the data order and LR curve depend on them.
- **Health:** non-finite steps are skipped; five in a row stops the run (resume, perhaps
  with a lower `--lr`). Watch `gnorm` and the per-source `val ce` in the log.
- **Consumer card:** no ECC and weeks of runtime. Keep the case cool, cap power if
  thermals are marginal (`nvidia-smi -pl 380` costs ~5% speed), and keep checkpoints on a
  different disk from the data if possible.
