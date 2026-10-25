#!/usr/bin/env python3
"""Train the cirrus 32K tokenizer on a sample of the pretraining mix.

    pip install tokenizers datasets zstandard
    python scripts/train_tokenizer.py                       # ~4 GB of text, lean mix
    python scripts/train_tokenizer.py --gb 1 --out tokenizer/test.json

Run it on a DGX Spark (lots of RAM and cores). The sample is drawn from each
source in proportion to its weight in the mix; a separate held-out slice
then counts token frequencies so ids can be renumbered most-frequent-first
(the MTP head's FR-Spec draft vocab is the id prefix).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cirrus.data.sources import MIXES, SOURCES  # noqa: E402
from cirrus.data.tokenizer import reorder_by_frequency, special_tokens, train  # noqa: E402


def sample(mix_name: str, total_bytes: float, skip_bytes: float = 0.0):
    """Yield documents, ``total_bytes`` overall, split across sources by mix weight."""
    mix = MIXES[mix_name]
    weights = {k: w / sum(mix.weights.values()) for k, w in mix.weights.items()}
    for name, w in weights.items():
        want, skip, got, t0 = w * total_bytes, w * skip_bytes, 0, time.time()
        for doc in SOURCES[name].stream():
            n = len(doc.encode())
            if skip > 0:
                skip -= n
                continue
            yield doc
            got += n
            if got >= want:
                break
        print(f"   {name}: {got / 1e6:,.0f} MB in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="lean", choices=sorted(MIXES))
    ap.add_argument("--gb", type=float, default=4.0, help="training sample size (text GB)")
    ap.add_argument("--count-gb", type=float, default=0.5, help="held-out sample for frequency ordering")
    ap.add_argument("--vocab", type=int, default=32_768)
    ap.add_argument("--out", default=str(ROOT / "tokenizer" / "cirrus-32k.json"))
    args = ap.parse_args()

    print(f"training BPE ({args.vocab:,} tokens) on {args.gb} GB of the {args.mix} mix", flush=True)
    t0 = time.time()
    tok = train(sample(args.mix, args.gb * 1e9), vocab_size=args.vocab)
    print(f"trained in {(time.time() - t0) / 60:.1f} min; counting frequencies on a held-out "
          f"{args.count_gb} GB", flush=True)
    # held out: skip past the training sample in every source
    held = sample(args.mix, args.count_gb * 1e9, skip_bytes=args.gb * 1e9)
    tok = reorder_by_frequency(tok, held)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    probe = "The Foo X1 costs $1,299 (2026) [2]. <|search|>best e-bike<|/search|><|run|>"
    enc = tok.encode(probe)
    meta = {"vocab_size": tok.get_vocab_size(), "mix": args.mix, "train_gb": args.gb, "count_gb": args.count_gb,
            "n_special": len(special_tokens()), "probe_tokens": enc.tokens,
            "top_tokens": [tok.id_to_token(i) for i in itertools.islice(range(64, 10_000), 40)]}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out} ({meta['vocab_size']:,} tokens)\nprobe: {enc.tokens}")


if __name__ == "__main__":
    main()
