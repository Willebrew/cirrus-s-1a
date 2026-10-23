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


def _stage_key(name: str) -> str:
    """Matrices that read the same input vector run in one fused stage."""
    if "." not in name:
        return name
    prefix, last = name.rsplit(".", 1)
    if prefix.endswith(".gdn") and last in ("qkv", "gate", "ab"):
        return prefix + ".in"
    if prefix.endswith(".swa") and last in ("q", "kv", "gate"):
        return prefix + ".in"
    if prefix == "shared":
        return "shared"
    if prefix.startswith("cross") and last in ("q", "gate"):
        return prefix + ".in"
    return name


def decode_stages(cfg: ModelConfig, include_mtp: bool = True) -> list[list[tuple[str, int, int]]]:
    """Group the decode-step matrices into dependent stages (one grid barrier
    each in the megakernel). Every matrix in a stage has the same K."""
    stages: list[list[tuple[str, int, int]]] = []
    last_key = None
    for m in decode_weight_matrices(cfg, include_mtp):
        key = _stage_key(m[0])
        if stages and key == last_key:
            stages[-1].append(m)
        else:
            stages.append([m])
        last_key = key
    return stages
