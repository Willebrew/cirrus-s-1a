"""Reference decoding: plain greedy and speculative greedy (prompt lookup +
MTP drafts). Speculative output is token-identical to plain greedy; the test
suite checks this. The Spark engine implements the same algorithm in a
persistent kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cirrus.decoding.prompt_lookup import PromptLookupDrafter, accept_greedy, choose_draft
from cirrus.model.model import CirrusModel


@dataclass
class GenStats:
    steps: int = 0
    tokens: int = 0

    @property
    def tokens_per_step(self) -> float:
        return self.tokens / max(1, self.steps)


@torch.no_grad()
def greedy(model: CirrusModel, prompt: list[int], max_new: int, stop: set[int] = frozenset()):
    ids = torch.tensor([prompt])
    out = model(ids, use_cache=True, n_logits=1)
    gen, stats = [], GenStats()
    while True:
        tok = int(out.logits[0, -1].argmax())
        gen.append(tok)
        stats.steps += 1
        stats.tokens += 1
        if tok in stop or len(gen) >= max_new:
            return gen, stats
        out = model(torch.tensor([[tok]]), cache=out.cache, n_logits=1)


@torch.no_grad()
def speculative_greedy(
    model: CirrusModel,
    prompt: list[int],
    max_new: int,
    stop: set[int] = frozenset(),
    max_draft: int = 8,
    use_mtp: bool = True,
    drafter: PromptLookupDrafter | None = None,
):
    """Greedy decoding with draft-and-verify.

    Each step feeds [pending token] + draft and verifies every draft position
    in one forward. On partial acceptance the cache is restored from a
    snapshot and only the accepted prefix is re-run; recurrent (GDN) state
    cannot be rewound, and recomputing a few tokens is cheaper than storing
    a state per draft position. The Spark engine instead keeps per-position
    states in shared memory for the (small) accepted window.
    """
    drafter = drafter or PromptLookupDrafter(max_draft=max_draft)
    out = model(torch.tensor([prompt]), use_cache=True, n_logits=1)
    cache, h_last = out.cache, out.hidden[:, -1]
    pending = int(out.logits[0, -1].argmax())
    drafter.extend(prompt + [pending])
    gen, stats = [pending], GenStats(steps=1, tokens=1)

    while pending not in stop and len(gen) < max_new:
        mtp = model.mtp_draft(h_last, torch.tensor([pending])) if use_mtp else []
        draft = choose_draft(drafter.draft(max_draft), mtp)[: max(0, max_new - len(gen) - 1)]
        snapshot = cache.clone() if draft else None
        inp = [pending] + draft
        o = model(torch.tensor([inp]), cache=cache, n_logits=len(inp))
        target = o.logits[0].argmax(-1).tolist()
        n_acc, emit = accept_greedy(draft, target)
        if n_acc < len(draft):
            o = model(torch.tensor([[pending] + draft[:n_acc]]), cache=snapshot, n_logits=1)
        cache, h_last = o.cache, o.hidden[:, -1]
        # never emit past a stop token
        for i, t in enumerate(emit):
            if t in stop:
                emit = emit[: i + 1]
                break
        gen.extend(emit)
        drafter.extend(emit)
        pending = emit[-1]
        stats.steps += 1
        stats.tokens += len(emit)
    return gen[:max_new], stats
