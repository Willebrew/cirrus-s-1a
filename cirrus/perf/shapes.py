"""Weight matrices streamed by one cirrus-s-1a decode step, in execution order.

Used by the Spark benchmark to replay the model's exact GEMV sequence with
random weights; speed does not depend on what the weights have learned.
"""

from __future__ import annotations

from cirrus.config import ModelConfig


def decode_weight_matrices(cfg: ModelConfig, include_mtp: bool = True) -> list[tuple[str, int, int]]:
    """(name, out_features N, in_features K) for every matrix of a decode step.

    Tiny projections (GDN alpha/beta, attention gates, indexer weights) are
    included so the byte count matches the roofline model; norms and conv
    taps are omitted (a few KB).
    """
    d, ffn = cfg.d_model, cfg.ffn_hidden
    qk = cfg.gdn_heads * cfg.gdn_head_dim_k
    vd = cfg.gdn_heads * cfg.gdn_head_dim_v
    mats: list[tuple[str, int, int]] = []

    def ffn_mats(p: str) -> None:
        mats.append((f"{p}.ffn.gate_up", 2 * ffn, d))
        mats.append((f"{p}.ffn.down", d, ffn))

    for i, kind in enumerate(cfg.self_layer_kinds):
        p = f"self{i}"
        if kind == "gdn":
            mats += [
                (f"{p}.gdn.qkv", 2 * qk + vd, d),
                (f"{p}.gdn.gate", vd, d),
                (f"{p}.gdn.ab", 2 * cfg.gdn_heads, d),
                (f"{p}.gdn.o", d, vd),
            ]
        else:
            mats += [
                (f"{p}.swa.q", cfg.q_dim, d),
                (f"{p}.swa.kv", 2 * cfg.kv_dim, d),
                (f"{p}.swa.gate", cfg.n_q_heads, d),
                (f"{p}.swa.o", d, cfg.q_dim),
            ]
        ffn_mats(p)

    mats.append(("shared.kv", 2 * cfg.kv_dim, d))
    if cfg.sparse_topk is not None:
        mats += [
            ("shared.idx_k", cfg.indexer_dim, d),
            ("shared.idx_q", cfg.indexer_heads * cfg.indexer_dim, d),
            ("shared.idx_w", cfg.indexer_heads, d),
        ]
    for j in range(cfg.n_cross_layers):
        p = f"cross{j}"
        mats += [(f"{p}.q", cfg.q_dim, d), (f"{p}.gate", cfg.n_q_heads, d), (f"{p}.o", d, cfg.q_dim)]
        ffn_mats(p)

    mats.append(("lm_head", cfg.vocab_size, d))
    if include_mtp and cfg.n_mtp_heads:
        mats += [("mtp.proj", d, 2 * d), ("mtp.ffn.gate_up", 2 * ffn, d), ("mtp.ffn.down", d, ffn)]
        mats.append(("mtp.draft_head", min(cfg.mtp_draft_vocab, cfg.vocab_size), d))
    return mats
