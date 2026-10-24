"""Offline measurement of tau (accepted tokens per verify step).

With greedy verification, speculative decoding emits exactly the tokens plain
greedy decoding would. So tau for prompt-lookup drafting can be measured
*exactly* without any speculative hardware: generate an answer once, then
replay the drafter over it and count how many verify steps it would take.

The MTP draft head has no trained weights yet, so its contribution is
modeled: each of its 2 drafted tokens is accepted with a conditional
probability (p1, then p2 given the first). Two drafting policies are
simulated:
  * "fallback": use the lookup draft when it has >= min_lookup tokens, else
    the MTP draft (cirrus.decoding.prompt_lookup.choose_draft, as shipped)
  * "tree": verify both drafts in one step and keep the longer accepted one
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from cirrus.decoding.prompt_lookup import PromptLookupDrafter


@dataclass
class TauStats:
    tokens: int = 0  # tokens produced by decode steps (the first token comes from prefill)
    steps: int = 0
    emitted_hist: Counter = field(default_factory=Counter)  # tokens emitted per step -> count
    lookup_steps: int = 0  # steps that used (or, in tree mode, won with) the lookup draft

    @property
    def tau(self) -> float:
        return self.tokens / self.steps if self.steps else 1.0

    def merge(self, other: "TauStats") -> "TauStats":
        self.tokens += other.tokens
        self.steps += other.steps
        self.emitted_hist.update(other.emitted_hist)
        self.lookup_steps += other.lookup_steps
        return self


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def simulate(
    prompt: Sequence[int],
    output: Sequence[int],
    max_draft: int = 7,
    min_ngram: int = 2,
    max_ngram: int = 4,
    mtp: tuple[float, ...] = (),
    mode: str = "fallback",
    min_lookup: int = 2,
    seed: int = 0,
) -> TauStats:
    """Replay draft-and-verify over a greedy ``output`` that followed ``prompt``.

    ``mtp`` holds the conditional acceptance probabilities of the MTP draft
    tokens, e.g. (0.8, 0.6); empty means no MTP head (lookup only).
    """
    if mode not in ("fallback", "tree"):
        raise ValueError("mode must be 'fallback' or 'tree'")
    st = TauStats()
    L = len(output)
    if L <= 1:
        return st
    rng = random.Random(seed)
    d = PromptLookupDrafter(max_ngram=max_ngram, min_ngram=min_ngram, max_draft=max_draft)
    d.extend(prompt)
    d.extend(output[:1])  # produced by prefill; it is the first "pending" token
    i = 1
    while i < L:
        look = d.draft(max_draft)
        acc_look = _lcp(look, output[i:])
        acc_mtp = 0
        if mtp:
            for p in mtp:
                if rng.random() >= p:
                    break
                acc_mtp += 1
        if mode == "fallback":
            use_look = len(look) >= min_lookup or not mtp
            acc = acc_look if use_look else acc_mtp
        else:
            use_look = acc_look >= acc_mtp
            acc = max(acc_look, acc_mtp)
        emit = min(acc + 1, L - i)  # accepted drafts + the verifier's own next token
        d.extend(output[i : i + emit])
        i += emit
        st.tokens += emit
        st.steps += 1
        st.emitted_hist[emit] += 1
        st.lookup_steps += int(use_look and acc_look > 0)
    return st


def copy_fraction(prompt: Sequence[int], output: Sequence[int], n: int = 4) -> float:
    """Fraction of output tokens that end an n-gram also present in the prompt."""
    if len(output) < n:
        return 0.0
    grams = {tuple(prompt[j : j + n]) for j in range(len(prompt) - n + 1)}
    hits = sum(1 for j in range(n - 1, len(output)) if tuple(output[j - n + 1 : j + 1]) in grams)
    return hits / (len(output) - n + 1)
