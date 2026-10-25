"""cirrus-s-1a reference model (YOCO decoder-decoder, hybrid GDN + attention).

See cirrus/config.py for the layout and docs/design/architecture.md for the
reasoning behind every choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cirrus.config import ModelConfig
from cirrus.model.layers import (
    Block,
    CrossAttention,
    GatedDeltaNet,
    MTPModule,
    PartialRotary,
    RMSNorm,
    SharedKV,
    SlidingWindowAttention,
    _acc,
    indexer_scores,
    topk_mask,
)


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


@dataclass
class Cache:
    """Everything decode needs. Size is O(window + context * kv_row), with the
    GDN state constant in sequence length."""

    layers: list = field(default_factory=list)  # per self-decoder layer
    shared: dict = field(default_factory=dict)  # k, v, ik over all tokens
    pos: int = 0

    def clone(self) -> "Cache":
        return Cache([_clone(l) for l in self.layers], _clone(self.shared), self.pos)


@dataclass
class Output:
    logits: torch.Tensor | None  # None in training when labels are given (the loss chunks the LM head)
    hidden: torch.Tensor  # pre-final-norm hidden of the returned positions
    cache: Cache | None = None
    loss: torch.Tensor | None = None
    metrics: dict = field(default_factory=dict)


class CirrusModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        rope = PartialRotary(cfg.head_dim, cfg.partial_rotary, cfg.rope_theta)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.self_layers = nn.ModuleList()
        for kind in cfg.self_layer_kinds:
            mixer = GatedDeltaNet(cfg) if kind == "gdn" else SlidingWindowAttention(cfg, rope)
            self.self_layers.append(Block(cfg, mixer))
        self.shared = SharedKV(cfg, rope)
        self.cross_layers = nn.ModuleList(Block(cfg, CrossAttention(cfg, rope)) for _ in range(cfg.n_cross_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = nn.ModuleList(MTPModule(cfg) for _ in range(cfg.n_mtp_heads))
        # FR-Spec draft vocabulary: ids of the most frequent tokens. Defaults
        # to the first N ids (BPE ids are roughly frequency ordered); replace
        # with corpus counts via ``set_draft_vocab``.
        self.register_buffer("draft_ids", torch.arange(min(cfg.mtp_draft_vocab, cfg.vocab_size)), persistent=True)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        std = 0.02
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)

    def set_draft_vocab(self, ids: torch.Tensor) -> None:
        self.draft_ids = ids.to(self.draft_ids.device, torch.long)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
        n_logits: int | None = None,
        mtp_weight: float = 0.3,
        indexer_weight: float = 0.0,
    ) -> Output:
        """
        input_ids: [B, T]. labels: [B, T] aligned with input_ids (HF style;
        shifted internally, -100 = ignore, which is how harness-written tool
        results are masked). ``n_logits`` restricts the cross-decoder and LM
        head to the last n positions: this is YOCO's prefill shortcut, since
        prompt tokens only need to contribute shared KV.
        """
        cfg = self.cfg
        B, T = input_ids.shape
        start = cache.pos if cache is not None else 0
        dev = input_ids.device
        pos = torch.arange(start, start + T, device=dev)
        want_cache = use_cache or cache is not None

        # ---------------------------------------------------- self-decoder
        h = self.embed(input_ids)
        new_layers = []
        for i, blk in enumerate(self.self_layers):
            lc = cache.layers[i] if cache is not None else None
            if isinstance(blk.mixer, GatedDeltaNet):
                y, nc = blk.mixer(blk.norm1(h), lc)
            else:
                y, nc = blk.mixer(blk.norm1(h), pos, lc)
            h = h + y
            h = h + blk.ffn(blk.norm2(h))
            new_layers.append(nc)

        # ----------------------------------------------- shared KV (YOCO)
        skv = self.shared(h, pos)
        if cache is not None and cache.shared:
            skv = {k: torch.cat((cache.shared[k], v), dim=1) for k, v in skv.items() if k in ("k", "v", "ik")} | {
                k: v for k, v in skv.items() if k in ("iq", "iw")
            }
        k_all, v_all = skv["k"], skv["v"]
        Tk = k_all.shape[1]

        # ---------------------------------------------------- cross-decoder
        nq = T if n_logits is None else min(n_logits, T)
        x, qpos = h[:, -nq:], pos[-nq:]
        scores = None
        if cfg.sparse_topk is None and nq == Tk:
            mask = None  # dense and square (pretraining): plain causal, no [T, T] mask -> FlashAttention
        else:
            kpos = torch.arange(Tk, device=dev)
            mask = (kpos[None, :] <= qpos[:, None])[None]  # causal, [1, nq, Tk]
            if cfg.sparse_topk is not None:
                scores = indexer_scores(skv["iq"][:, -nq:], skv["iw"][:, -nq:], skv["ik"])
                mask = topk_mask(scores, mask, cfg.sparse_topk)
        probs0 = None
        for j, blk in enumerate(self.cross_layers):
            want_probs = indexer_weight > 0 and j == 0 and scores is not None
            y, probs = blk.mixer(blk.norm1(x), qpos, k_all, v_all, mask, return_probs=want_probs)
            probs0 = probs if want_probs else probs0
            x = x + y
            x = x + blk.ffn(blk.norm2(x))

        hn = self.final_norm(x)
        # training never needs the full [B, T, V] logits: the loss computes them chunk by chunk
        logits = None if (labels is not None and self.training) else self.lm_head(hn)
        out = Output(logits=logits, hidden=x)

        if want_cache:
            out.cache = Cache(new_layers, {k: skv[k].detach() for k in ("k", "v", "ik") if k in skv}, start + T)

        if labels is not None:
            out.loss, out.metrics = self._loss(input_ids, labels, hn, x, scores, probs0, mask, mtp_weight,
                                               indexer_weight)
        return out

    # ----------------------------------------------------------------- loss
    CE_CHUNK = 4096  # positions per LM-head chunk in the loss

    def _ce_sum(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(_acc(self.lm_head(h)), y, ignore_index=-100, reduction="sum")

    def _ce(self, hn: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean next-token CE of normed hidden ``hn`` [B, T, d] against ``labels`` [B, T].

        The LM head runs over CE_CHUNK positions at a time and is recomputed in
        backward, so peak memory is one chunk of fp32 logits instead of
        [B, T, V] (about 4 GB per pass at 8 x 4096 x 32K), for ~5% more FLOPs.
        """
        h, y = hn.reshape(-1, hn.shape[-1]), labels.reshape(-1)
        total = h.new_zeros((), dtype=torch.promote_types(h.dtype, torch.float32))
        for s in range(0, y.numel(), self.CE_CHUNK):
            hc, yc = h[s : s + self.CE_CHUNK], y[s : s + self.CE_CHUNK]
            if torch.is_grad_enabled():
                total = total + checkpoint(self._ce_sum, hc, yc, use_reentrant=False)
            else:
                total = total + self._ce_sum(hc, yc)
        return total / (y != -100).sum().clamp_min(1)

    def _loss(self, input_ids, labels, hn, hidden, scores, probs0, mask, mtp_weight, indexer_weight):
        T = input_ids.shape[1]
        ce = self._ce(hn[:, :-1], labels[:, 1:])
        loss, metrics = ce, {"ce": ce.detach()}
        if mtp_weight > 0 and len(self.mtp):
            h, mtp_losses = hidden, []
            for mod in self.mtp:
                for j in range(1, self.cfg.mtp_passes + 1):
                    if T - j - 1 <= 0:
                        break
                    # h[:, t] predicted token t+j; feed token t+j, predict t+j+1
                    h = mod(h[:, :-1], self.embed(input_ids[:, j:]))
                    mtp_losses.append(self._ce(self.final_norm(h)[:, : T - j - 1], labels[:, j + 1 :]))
                h = hidden
            if mtp_losses:
                mtp = torch.stack(mtp_losses).mean()
                loss = loss + mtp_weight * mtp
                metrics["mtp"] = mtp.detach()
        if indexer_weight > 0 and probs0 is not None:
            # KL(attention || indexer) over the keys attention could see
            target = (probs0 / probs0.sum(-1, keepdim=True).clamp_min(1e-9)).detach()
            logp = scores.masked_fill(~mask, float("-inf")).log_softmax(-1)
            kl = (target * (target.clamp_min(1e-9).log() - logp.masked_fill(~mask, 0.0))).sum(-1).mean()
            loss = loss + indexer_weight * kl
            metrics["indexer_kl"] = kl.detach()
        return loss, metrics

    # ---------------------------------------------------------------- draft
    @torch.no_grad()
    def mtp_draft(self, h_last: torch.Tensor, token: torch.Tensor, n: int | None = None) -> list[int]:
        """Draft ``n`` tokens after ``token`` with the FR-Spec restricted head.

        h_last: [1, d] hidden that predicted ``token``; token: [1] long.
        """
        if not len(self.mtp):
            return []
        n = self.cfg.mtp_passes if n is None else n
        W = self.embed.weight[self.draft_ids]  # [Vd, d]
        h, tok, out = h_last[:, None], token.view(1, 1), []
        for _ in range(n):
            h = self.mtp[0](h, self.embed(tok))
            idx = (self.final_norm(h)[:, -1] @ W.T).argmax(-1)
            tok = self.draft_ids[idx].view(1, 1)
            out.append(int(tok))
        return out
