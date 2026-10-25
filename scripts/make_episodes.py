#!/usr/bin/env python3
"""Build protocol-format search transcripts from HotpotQA and SQuAD 2.0 (cirrus/data/episodes.py).

    pip install datasets
    python scripts/make_episodes.py --out data/episodes/train.jsonl                 # ~220K transcripts
    python scripts/make_episodes.py --split validation --out data/episodes/val.jsonl
    python scripts/make_episodes.py --n 3 --preview 3 --out /tmp/peek.jsonl           # look at a few

Each line: {"id", "source", "kind", "segments": [[text, train], ...]}. ``train`` is
false for harness-written text (query, results, pages), which gets no loss.
Transcripts whose answer fails the citation audit are dropped and counted.
Small (both datasets together are ~0.5 GB), single process, a few minutes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cirrus.agent import protocol as P  # noqa: E402
from cirrus.agent.citations import audit_answer  # noqa: E402
from cirrus.data.episodes import Doc, EpisodeBuilder, EpisodeMix, from_hotpot, from_squad, to_text  # noqa: E402


def hotpot(split: str, n: int | None):
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        ex = from_hotpot(row)
        if ex:
            yield ex


def squad(split: str, n: int | None, seed: int):
    """SQuAD paragraphs get 4 distractor paragraphs from other articles seen recently."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split, streaming=True)
    recent: deque[Doc] = deque(maxlen=400)
    seen_ctx: set[str] = set()
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        rng = random.Random(f"{seed}:{row['id']}")
        title = row["title"].replace("_", " ")
        pool = [d for d in recent if d.title != title]
        distractors = [Doc(d.title, d.text) for d in rng.sample(pool, min(4, len(pool)))]
        if len(distractors) >= 2:
            yield from_squad(row, distractors)
        if row["context"] not in seen_ctx:
            seen_ctx.add(row["context"])
            recent.append(Doc(title, row["context"].strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["hotpot", "squad"], choices=["hotpot", "squad"])
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--n", type=int, default=None, help="rows per source (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-counterfactual", type=float, default=0.25)
    ap.add_argument("--p-unanswerable", type=float, default=0.15, help="added on top of SQuAD's native ones")
    ap.add_argument("--preview", type=int, default=0, help="print this many transcripts")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    t0 = time.time()
    shown = 0
    with open(out, "w") as f:
        for name in args.sources:
            builder = EpisodeBuilder(EpisodeMix(args.p_counterfactual, args.p_unanswerable, args.seed))
            stream = hotpot(args.split, args.n) if name == "hotpot" else squad(args.split, args.n, args.seed)
            for ex in stream:
                variant, segs = builder.build(ex)
                if variant.answer is not None:
                    answer = P.parse_turn(segs[-1].text).answer or ""
                    if not audit_answer(answer, set(range(1, 100))).ok:
                        stats[f"{name}/dropped_audit"] += 1
                        continue
                stats[f"{name}/{variant.kind}"] += 1
                f.write(json.dumps({"id": ex.id, "source": name, "kind": variant.kind,
                                    "segments": [[s.text, s.train] for s in segs]}) + "\n")
                if shown < args.preview:
                    shown += 1
                    print(f"\n===== {name} {variant.kind} {ex.id} =====\n{to_text(segs)}")
            print(f"{name}: {dict((k, v) for k, v in stats.items() if k.startswith(name))}", flush=True)
    total = sum(v for k, v in stats.items() if "dropped" not in k)
    print(f"\n{total:,} transcripts -> {out} in {time.time() - t0:.0f}s")
    for kind in ("answerable", "counterfactual", "unanswerable"):
        k = sum(v for key, v in stats.items() if key.endswith("/" + kind))
        print(f"   {kind:15s} {k:8,d}  ({k / max(1, total):.0%})")


if __name__ == "__main__":
    main()
