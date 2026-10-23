"""Prompt-lookup (copy) drafting for speculative decoding.

Search answers copy heavily from retrieved sources: names, numbers, dates,
quoted phrases, URLs. When the last ``n`` generated tokens also appear in the
context, the tokens that followed them there are an excellent draft. The
draft costs no GPU work; the main model verifies all drafted positions in one
forward pass, which on a bandwidth-bound device costs about the same as
decoding one token.

The index is incremental: each appended token registers the n-grams ending at
it, so lookup is O(max_n) per step regardless of context length.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class PromptLookupDrafter:
    def __init__(self, max_ngram: int = 4, min_ngram: int = 2, max_draft: int = 12):
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.max_draft = max_draft
        self.tokens: list[int] = []
        # n-gram -> position just after its most recent occurrence
        self._index: list[dict[tuple[int, ...], int]] = [dict() for _ in range(max_ngram + 1)]

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Iterable[int]) -> None:
        for t in tokens:
            self.tokens.append(int(t))
            end = len(self.tokens)
            for n in range(self.min_ngram, self.max_ngram + 1):
                if end < n:
                    break
                # Register the n-gram that ends *before* the newest token, so
                # its continuation (the newest token) is already known.
                if end - 1 >= n:
                    key = tuple(self.tokens[end - 1 - n : end - 1])
                    self._index[n][key] = end - 1

    def draft(self, k: int | None = None) -> list[int]:
        """Return up to ``k`` drafted tokens continuing the current sequence."""
        k = self.max_draft if k is None else min(k, self.max_draft)
        toks = self.tokens
        for n in range(self.max_ngram, self.min_ngram - 1, -1):
            if len(toks) < n + 1:
                continue
            key = tuple(toks[-n:])
            start = self._index[n].get(key)
            if start is None:
                continue
            return toks[start : start + k]
        return []


def accept_greedy(draft: Sequence[int], target: Sequence[int]) -> tuple[int, list[int]]:
    """Greedy verification.

    ``target[i]`` is the main model's argmax after consuming the prefix plus
    ``draft[:i]``; ``len(target) == len(draft) + 1``. Returns the number of
    accepted draft tokens and the tokens to emit (accepted drafts plus the
    model's own next token), so every step emits at least one token.
    """
    if len(target) != len(draft) + 1:
        raise ValueError("target must have exactly one more position than draft")
    n = 0
    while n < len(draft) and draft[n] == target[n]:
        n += 1
    return n, list(draft[:n]) + [target[n]]


def choose_draft(lookup: Sequence[int], mtp: Sequence[int], min_lookup: int = 2) -> list[int]:
    """Prefer a long copy draft; fall back to the MTP head's draft.

    Copy drafts are cheap to verify and, when a span is being quoted, are
    usually right for many tokens in a row. MTP drafts are short (2 tokens)
    but cover novel text. A tree verifier could check both; v1 picks one.
    """
    if len(lookup) >= min_lookup and len(lookup) >= len(mtp):
        return list(lookup)
    return list(mtp) if mtp else list(lookup)
