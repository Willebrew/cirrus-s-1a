"""The cirrus 32K byte-level BPE tokenizer.

* digits are split one per token (arithmetic, prices, dates)
* protocol tokens (cirrus/agent/protocol.py) and a few control tokens are
  reserved at the lowest ids, plus spare slots for later formats
* every other id is ordered by corpus frequency, so the FR-Spec draft
  vocabulary of the MTP head is simply the id prefix [0, mtp_draft_vocab)
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from cirrus.agent.protocol import SPECIAL_TOKENS

PAD, BOS, EOS = "<|pad|>", "<|bos|>", "<|eos|>"
N_RESERVED = 64  # ids [0, 64) are special; unused ones are <|reserved_i|>


def special_tokens() -> list[str]:
    toks = [PAD, BOS, EOS, *SPECIAL_TOKENS]
    toks += [f"<|reserved_{i}|>" for i in range(N_RESERVED - len(toks))]
    assert len(toks) == N_RESERVED and len(set(toks)) == N_RESERVED
    return toks


def _new_tokenizer():
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(texts: Iterable[str], vocab_size: int = 32_768, min_frequency: int = 2):
    """Train BPE on ``texts``. Ids are not yet frequency ordered (see ``reorder_by_frequency``)."""
    from tokenizers import pre_tokenizers, trainers

    tok = _new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def reorder_by_frequency(tok, texts: Iterable[str], batch: int = 1024):
    """Renumber non-special tokens by how often they occur in ``texts``.

    BPE merges are stored as token strings, so renumbering the vocab leaves
    tokenization unchanged: only the ids move.
    """
    from tokenizers import Tokenizer

    counts: Counter[int] = Counter()
    buf: list[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) == batch:
            for enc in tok.encode_batch(buf, add_special_tokens=False):
                counts.update(enc.ids)
            buf = []
    for enc in tok.encode_batch(buf, add_special_tokens=False):
        counts.update(enc.ids)

    spec = json.loads(tok.to_str())
    vocab: dict[str, int] = spec["model"]["vocab"]
    specials = special_tokens()
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    rest = [t for t, _ in by_id if t not in specials]
    rest.sort(key=lambda t: (-counts.get(vocab[t], 0), vocab[t]))  # frequent first, ties keep BPE order
    new_vocab = {t: i for i, t in enumerate(specials)} | {t: N_RESERVED + i for i, t in enumerate(rest)}
    spec["model"]["vocab"] = new_vocab
    for at in spec["added_tokens"]:
        at["id"] = new_vocab[at["content"]]
    return Tokenizer.from_str(json.dumps(spec))


class CirrusTokenizer:
    """Thin wrapper with the ids the rest of the code needs."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.tok = Tokenizer.from_file(str(path))
        self.pad_id, self.bos_id, self.eos_id = (self.tok.token_to_id(t) for t in (PAD, BOS, EOS))
        assert (self.pad_id, self.bos_id, self.eos_id) == (0, 1, 2), "special ids moved"

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)
