"""Pretraining sources and mixes (docs/research/datasets.md, docs/plans/training-budget.md).

Each source knows how to stream itself from the Hugging Face Hub and how to
render one row into a training document. The renderings favour *reading*:
wherever a dataset ships the document a synthetic answer was generated from,
the document comes first and the answer after it, so the model learns to
answer from text it was shown rather than from memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

Row = dict
PARQUET_BATCH_ROWS = 1024  # rows decoded at a time per worker
Render = Callable[[Row], "str | None"]


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    render: Render
    config: str | None = None
    data_files: str | list[str] | None = None  # for repos whose configs don't match their layout
    builder: str | None = None  # e.g. "json" for raw jsonl.zst files
    split: str = "train"
    columns: tuple[str, ...] | None = None  # parquet: read only these (what render uses)
    note: str = ""

    def stream(self, rank: int = 0, world: int = 1, state: dict | None = None) -> DocStream:
        """Rendered documents from this worker's slice of the dataset.

        Workers split the dataset by *files* (``IterableDataset.shard``), so
        nobody downloads rows they throw away. ``state`` resumes exactly
        where a previous ``state_dict()`` was taken.
        """
        ds = self.load()
        if world > 1:
            if ds.num_shards < world:
                raise ValueError(f"{self.name} has {ds.num_shards} files, fewer than {world} workers")
            ds = ds.shard(num_shards=world, index=rank)
        if state is not None:
            ds.load_state_dict(state)
        return DocStream(ds, self.render)

    def load(self):
        """The streaming ``datasets.IterableDataset`` for this source."""
        from datasets import load_dataset

        kw = {}
        if self.builder in (None, "parquet"):
            # By default datasets reads parquet one whole row group, all columns, at a time:
            # hundreds of MB per worker on these files. Bound it.
            kw["batch_size"] = PARQUET_BATCH_ROWS
            if self.columns:
                kw["columns"] = list(self.columns)
        if self.builder:
            return load_dataset(self.builder, data_files=self.data_files, split=self.split, streaming=True, **kw)
        return load_dataset(self.repo, self.config, split=self.split, streaming=True, **kw)


class DocStream:
    """Iterator of rendered docs that can report its exact resume point."""

    def __init__(self, ds, render: Render):
        self.ds = ds
        self._it = (d for d in map(render, ds) if d)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        return next(self._it)

    def state_dict(self) -> dict:
        return self.ds.state_dict()


# ------------------------------------------------------------------ renderers
def _clean(s) -> str:
    return (s or "").strip()


def render_text(row: Row) -> str | None:
    return _clean(row.get("text")) or None


def render_fineweb_edu(min_score: float) -> Render:
    def r(row: Row) -> str | None:
        if (row.get("score") or 0.0) < min_score:
            return None
        return render_text(row)

    return r


def render_finephrase(header: str) -> Render:
    """Source document first, then the rewrite of it (FAQ, table + QA, ...).

    ``text`` is the FineWeb-Edu source; the generated output is in
    ``rollout_results[0].text``. Truncated generations are dropped.
    """

    def r(row: Row) -> str | None:
        rolls = row.get("rollout_results") or []
        if not rolls:
            return None
        out = rolls[0]
        if out.get("finish_reason") not in (None, "stop"):
            return None
        doc, gen = _clean(row.get("text")), _clean(out.get("text"))
        if not doc or not gen:
            return None
        return f"{doc}\n\n{header}\n\n{gen}"

    return r


def render_synth(languages: tuple[str, ...] = ("en",), reasoning: bool = False) -> Render:
    """SYNTH rows as source -> question -> answer.

    The seed section is the source; showing it makes every exercise a
    reading exercise. Reasoning drafts are off by default: a search model
    that writes long drafts pays for them in decode latency.
    """

    def r(row: Row) -> str | None:
        if languages and row.get("language") not in languages:
            return None
        q, a = _clean(row.get("query")), _clean(row.get("synthetic_answer"))
        if not q or not a:
            return None
        parts = []
        seed = _clean(row.get("query_seed_text"))
        if seed:
            parts.append(f"Source:\n{seed}")
        parts.append(f"Question: {q}")
        if reasoning and _clean(row.get("synthetic_reasoning")):
            parts.append(f"Notes:\n{_clean(row['synthetic_reasoning'])}")
        parts.append(f"Answer:\n{a}")
        return "\n\n".join(parts)

    return r


# -------------------------------------------------------------------- registry
_DOLMINO = "hf://datasets/allenai/dolma3_dolmino_mix-100B-1025/data/"

SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("synth", "PleIAs/SYNTH", render_synth(),
               columns=("language", "query", "query_seed_text", "synthetic_answer"),
               note="~60B English tokens; RAG-style exercises"),
        Source("finephrase_faq", "HuggingFaceFW/finephrase", render_finephrase("Frequently asked questions"),
               config="faq", columns=("text", "rollout_results"), note="document followed by an FAQ written from it"),
        Source("finephrase_table", "HuggingFaceFW/finephrase", render_finephrase("Key facts"),
               config="table", columns=("text", "rollout_results"),
               note="document followed by a table and one QA pair"),
        Source("finephrase_math", "HuggingFaceFW/finephrase", render_finephrase("Worked problem"),
               config="math", columns=("text", "rollout_results"), note="numbers and arithmetic in context"),
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", render_fineweb_edu(3.5), config="default",
               columns=("text", "score"), note="explanatory web text, score >= 3.5"),
        Source("dolmino_qa", "allenai/dolma3_dolmino_mix-100B-1025", render_text, builder="json",
               data_files=[_DOLMINO + "wiki_to_rcqa-part*/*.jsonl.zst", _DOLMINO + "nemotron-synth-qa/*.jsonl.zst",
                           _DOLMINO + "reddit_to_flashcards/**/*.jsonl.zst"],
               note="reading-comprehension QA, ~14B tokens"),
    )
}


@dataclass(frozen=True)
class Mix:
    name: str
    weights: dict[str, float]  # fraction of tokens per source, sums to 1
    max_passes: dict[str, float] = field(default_factory=dict)  # cap repetition of small sources

    def budgets(self, total_tokens: float) -> dict[str, int]:
        s = sum(self.weights.values())
        return {k: int(total_tokens * w / s) for k, w in self.weights.items()}


MIXES: dict[str, Mix] = {
    # the lean own-hardware recipe: a strong reader, not an encyclopedia
    "lean": Mix("lean", {"synth": 0.30, "finephrase_faq": 0.20, "fineweb_edu": 0.25, "finephrase_table": 0.10,
                         "dolmino_qa": 0.10, "finephrase_math": 0.05}, max_passes={"synth": 2, "dolmino_qa": 2}),
}
