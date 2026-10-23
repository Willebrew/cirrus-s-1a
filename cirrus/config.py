"""Model configuration for cirrus-s-1a.

Pure Python (no torch) so the performance model, docs tooling and tests can
import it anywhere. The PyTorch reference model in ``cirrus.model`` consumes
the same dataclass.

Layout (YOCO-style decoder-decoder):

    tokens -> embed
           -> self-decoder:  N_self layers, mostly Gated DeltaNet (O(1) state)
                             with a few sliding-window attention layers
           -> shared KV projection (computed ONCE per token, cached)
           -> cross-decoder: N_cross layers, each only owns a Q projection and
                             attends to the single shared KV cache
           -> final norm -> tied LM head
           -> MTP heads (optional draft heads for speculative decoding)

Why this shape: on a 273 GB/s device every decoded token pays for every byte
it reads. A per-layer KV cache over 32K tokens of search results would cost
more bandwidth than the weights themselves. YOCO pays for exactly one KV
cache, and prefill only needs to run the self-decoder over the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

LayerKind = Literal["gdn", "swa", "cross"]


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cirrus-s-1a"
    vocab_size: int = 32_768
    d_model: int = 768
    # Self-decoder pattern, repeated. "gdn" = Gated DeltaNet, "swa" = sliding
    # window GQA attention.
    self_pattern: tuple[LayerKind, ...] = ("gdn", "gdn", "gdn", "swa")
    n_self_layers: int = 12
    n_cross_layers: int = 6

    # Gated DeltaNet
    gdn_heads: int = 6
    gdn_head_dim_k: int = 128
    gdn_head_dim_v: int = 128
    gdn_conv_size: int = 4

    # Attention (shared by swa + cross layers)
    n_q_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 128
    swa_window: int = 1024
    rope_theta: float = 1_000_000.0
    partial_rotary: float = 0.5  # fraction of head_dim that gets RoPE

    # Sparse attention for the cross-decoder (DeepSeek-Sparse-Attention-style
    # "lightning indexer"). One top-k selection per step is shared by every
    # cross layer because they all read the same KV cache. None = dense.
    sparse_topk: int | None = 2048
    indexer_heads: int = 4
    indexer_dim: int = 64

    # FFN (SwiGLU)
    ffn_hidden: int = 2048

    # Multi-token prediction draft module (DeepSeek-V3 style, but FFN-only and
    # reused recursively for ``mtp_passes`` draft positions). It predicts over
    # the ``mtp_draft_vocab`` most frequent tokens only (FR-Spec style) so the
    # draft LM head is small enough to stay resident in L2.
    n_mtp_heads: int = 1
    mtp_passes: int = 2
    mtp_draft_vocab: int = 8192

    tie_embeddings: bool = True
    max_seq_len: int = 65_536
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if self.n_self_layers <= 0 or self.n_cross_layers <= 0:
            raise ValueError("need at least one self and one cross layer")

    # ------------------------------------------------------------------ layout
    @property
    def self_layer_kinds(self) -> list[LayerKind]:
        p = self.self_pattern
        return [p[i % len(p)] for i in range(self.n_self_layers)]

    @property
    def layer_kinds(self) -> list[LayerKind]:
        return self.self_layer_kinds + ["cross"] * self.n_cross_layers

    @property
    def n_layers(self) -> int:
        return self.n_self_layers + self.n_cross_layers

    @property
    def q_dim(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # ------------------------------------------------------------ param counts
    def ffn_params(self) -> int:
        return 3 * self.d_model * self.ffn_hidden

    def gdn_mixer_params(self) -> int:
        d, h = self.d_model, self.gdn_heads
        qk = h * self.gdn_head_dim_k
        v = h * self.gdn_head_dim_v
        proj_in = d * (2 * qk + v)      # q, k, v
        gate_out = d * v                # output gate g
        ab = d * 2 * h                  # alpha (decay) and beta (write strength)
        conv = (2 * qk + v) * self.gdn_conv_size
        proj_out = v * d
        head_norm = self.gdn_head_dim_v
        decay = 2 * h  # A_log, dt_bias
        return proj_in + gate_out + ab + conv + proj_out + head_norm + decay

    def swa_mixer_params(self) -> int:
        d = self.d_model
        gate = d * self.n_q_heads  # head-wise sigmoid output gate
        return d * self.q_dim + 2 * d * self.kv_dim + self.q_dim * d + gate + 2 * self.head_dim

    def cross_mixer_params(self) -> int:
        d = self.d_model
        gate = d * self.n_q_heads
        return d * self.q_dim + self.q_dim * d + gate + self.head_dim

    def indexer_params(self) -> int:
        if self.sparse_topk is None:
            return 0
        d, h, k = self.d_model, self.indexer_heads, self.indexer_dim
        return d * k + d * h * k + d * h  # keys, queries, per-head weights

    def shared_kv_params(self) -> int:
        kv = 2 * self.d_model * self.kv_dim + self.d_model + self.head_dim
        return kv + self.indexer_params()

    def layer_params(self, kind: LayerKind) -> int:
        mixer = {
            "gdn": self.gdn_mixer_params,
            "swa": self.swa_mixer_params,
            "cross": self.cross_mixer_params,
        }[kind]()
        return mixer + self.ffn_params() + 2 * self.d_model  # + 2 RMSNorms

    def embedding_params(self) -> int:
        return self.vocab_size * self.d_model

    def mtp_head_params(self) -> int:
        """One MTP module: RMSNorm x2, concat-projection 2d->d, one SwiGLU FFN.

        Its LM head is a row-subset of the tied embedding (the draft vocab), so
        it adds no parameters, only bytes read (see the roofline model).
        """
        d = self.d_model
        return 3 * d + 2 * d * d + self.ffn_params()

    def backbone_params(self) -> int:
        body = sum(self.layer_params(k) for k in self.layer_kinds)
        return body + self.shared_kv_params() + self.d_model  # + final norm

    def total_params(self, include_mtp: bool = True) -> int:
        emb = self.embedding_params()
        head = 0 if self.tie_embeddings else emb
        mtp = self.n_mtp_heads * self.mtp_head_params() if include_mtp else 0
        return emb + head + self.backbone_params() + mtp

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_kinds"] = self.layer_kinds
        d["total_params"] = self.total_params()
        return d


# ----------------------------------------------------------------- presets
PRESETS: dict[str, ModelConfig] = {
    # The model we are building: ~155M params, ~99 MB streamed per verify
    # step (NVFP4), which clears 3,000 tok/s on Spark at >= ~1.7 accepted
    # tokens per step and ~2,000 tok/s with no speculation at all.
    "cirrus-s-1a": ModelConfig(),
    # Bigger sibling, kept for comparison: only viable if measured τ >= ~2.8.
    "cirrus-s-1a-large": ModelConfig(
        name="cirrus-s-1a-large",
        d_model=1024,
        n_self_layers=12,
        n_cross_layers=8,
        gdn_heads=8,
        n_q_heads=8,
        n_kv_heads=2,
        ffn_hidden=2816,
    ),
    # Tiny config for unit tests / CPU smoke runs.
    "cirrus-s-1a-debug": ModelConfig(
        name="cirrus-s-1a-debug",
        vocab_size=512,
        d_model=64,
        n_self_layers=4,
        n_cross_layers=2,
        gdn_heads=2,
        gdn_head_dim_k=16,
        gdn_head_dim_v=16,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=16,
        swa_window=8,
        sparse_topk=6,
        indexer_heads=2,
        indexer_dim=8,
        ffn_hidden=128,
        mtp_draft_vocab=128,
        max_seq_len=256,
    ),
}


def get_config(name: str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError as e:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from e
