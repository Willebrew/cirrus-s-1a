"""Reference PyTorch layers for cirrus-s-1a.

These are the *specification*: clear, numerically careful, and testable on
CPU. Training uses fused kernels (flash-linear-attention for Gated DeltaNet,
FlashAttention / FlexAttention for the attention layers) and inference uses
the custom Spark megakernel; both must match these modules to tolerance.

Tensor layout conventions: activations are [B, T, ...]; per-head tensors are
[B, T, H, D]; Gated DeltaNet state is [B, H, Dk, Dv] (fla-compatible).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cirrus.config import ModelConfig

try:  # fused chunked kernel for training on GPU
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
except Exception:  # pragma: no cover - optional dependency
    _fla_chunk_gdr = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


# --------------------------------------------------------------------- RoPE
class PartialRotary(nn.Module):
    """RoPE on the first ``rot_dim`` channels of each head (rest are NoPE).

    Partial rotary keeps a position-free subspace, which helps long-context
    retrieval and extrapolation (as in GLM / Qwen3-Next).
    """

    def __init__(self, head_dim: int, fraction: float, theta: float):
        super().__init__()
        rot = int(head_dim * fraction) // 2 * 2
        self.rot_dim = rot
        inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / max(rot, 1)))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H, D]; pos: [T] absolute positions."""
        if self.rot_dim == 0:
            return x
        freqs = pos.float()[:, None] * self.inv_freq[None, :]  # [T, rot/2]
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        xr, xp = x[..., : self.rot_dim].float(), x[..., self.rot_dim :]
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return torch.cat((out.to(x.dtype), xp), dim=-1)


# ----------------------------------------------------------- short conv
class ShortConv(nn.Module):
    """Causal depthwise conv with a rolling cache of the last K-1 inputs."""

    def __init__(self, dim: int, k: int):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(dim, k))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(k))

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None):
        """x: [B, T, C]; cache: [B, K-1, C] or None. Returns (y, new_cache)."""
        B, T, C = x.shape
        if cache is None:
            cache = x.new_zeros(B, self.k - 1, C)
        xx = torch.cat((cache, x), dim=1)  # [B, K-1+T, C]
        # y_t = sum_j w[:, j] * xx[t + j]
        windows = xx.unfold(1, self.k, 1)  # [B, T, C, K]
        y = (windows * self.weight).sum(-1)
        return F.silu(y), xx[:, -(self.k - 1) :].detach() if self.k > 1 else xx[:, :0]


# ------------------------------------------------------- Gated DeltaNet
def gated_delta_rule_ref(q, k, v, g, beta, scale: float, state=None):
    """Exact recurrent Gated Delta Rule.

        S_t = a_t * S_{t-1} + k_t (beta_t * (v_t - k_t^T (a_t S_{t-1})))^T
        o_t = S_t^T (scale * q_t)

    q, k: [B, T, H, Dk] (already L2-normalized); v: [B, T, H, Dv];
    g: [B, T, H] log-decay (a_t = exp(g_t)); beta: [B, T, H].
    state: [B, H, Dk, Dv] float32 or None. Returns (o [B,T,H,Dv], state).
    """
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv, dtype=torch.float32) if state is None else state.float()
    out = []
    for t in range(T):
        S = S * g[:, t].float().exp()[..., None, None]
        kt, vt = k[:, t].float(), v[:, t].float()
        pred = torch.einsum("bhkv,bhk->bhv", S, kt)
        delta = beta[:, t].float()[..., None] * (vt - pred)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t].float() * scale))
    return torch.stack(out, dim=1).to(v.dtype), S


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer (Yang et al., ICLR 2025; as used in Qwen3-Next)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.gdn_heads
        self.H, self.dk, self.dv = H, cfg.gdn_head_dim_k, cfg.gdn_head_dim_v
        qk, vd = H * self.dk, H * self.dv
        self.qkv = nn.Linear(d, 2 * qk + vd, bias=False)
        self.gate = nn.Linear(d, vd, bias=False)
        self.ab = nn.Linear(d, 2 * H, bias=False)
        self.conv = ShortConv(2 * qk + vd, cfg.gdn_conv_size)
        self.o_norm = RMSNorm(self.dv, cfg.norm_eps)
        self.o_proj = nn.Linear(vd, d, bias=False)
        # Mamba-2 style decay parametrization: a = exp(-exp(A_log) * softplus(x + dt_bias))
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1.0, 16.0)))
        dt = torch.exp(torch.empty(H).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, x: torch.Tensor, cache: dict | None = None):
        B, T, _ = x.shape
        conv_state = cache.get("conv") if cache else None
        S0 = cache.get("S") if cache else None
        qkv, conv_state = self.conv(self.qkv(x), conv_state)
        qk = self.H * self.dk
        q, k, v = qkv.split([qk, qk, self.H * self.dv], dim=-1)
        q = F.normalize(q.view(B, T, self.H, self.dk).float(), dim=-1).to(x.dtype)
        k = F.normalize(k.view(B, T, self.H, self.dk).float(), dim=-1).to(x.dtype)
        v = v.view(B, T, self.H, self.dv)
        a, b = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())  # [B, T, H]
        beta = b.sigmoid()
        scale = self.dk**-0.5
        if _fla_chunk_gdr is not None and x.is_cuda and T > 1:
            # fla wants beta in the activation dtype; the log-decay g stays fp32
            o, S = _fla_chunk_gdr(q, k, v, g, beta.to(q.dtype), scale=scale, initial_state=S0, output_final_state=True)
        else:
            o, S = gated_delta_rule_ref(q, k, v, g, beta, scale, S0)
        o = self.o_norm(o) * F.silu(self.gate(x)).view(B, T, self.H, self.dv)
        return self.o_proj(o.reshape(B, T, -1)), {"conv": conv_state, "S": S}


# --------------------------------------------------------------- attention
def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    """[B, Hkv, T, D] -> [B, Hkv*n, T, D]"""
    return x if n == 1 else x.repeat_interleave(n, dim=1)


def _attend(q, k, v, mask, n_rep: int):
    """q: [B, T, Hq, D]; k, v: [B, Tk, Hkv, D]; mask: bool [B or 1, T, Tk]."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    k, v = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    return o.transpose(1, 2)  # [B, T, Hq, D]


class SlidingWindowAttention(nn.Module):
    """GQA with QK-norm, partial RoPE, head-wise sigmoid output gate."""

    def __init__(self, cfg: ModelConfig, rope: PartialRotary):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.rope = cfg, rope
        self.q_proj = nn.Linear(d, cfg.q_dim, bias=False)
        self.kv_proj = nn.Linear(d, 2 * cfg.kv_dim, bias=False)
        self.gate = nn.Linear(d, cfg.n_q_heads, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, d, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, cache: dict | None = None):
        c = self.cfg
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, c.n_q_heads, c.head_dim))
        k, v = self.kv_proj(x).view(B, T, 2, c.n_kv_heads, c.head_dim).unbind(2)
        q, k = self.rope(q, pos), self.rope(self.k_norm(k), pos)
        if cache:
            k = torch.cat((cache["k"], k), dim=1)
            v = torch.cat((cache["v"], v), dim=1)
        Tk = k.shape[1]
        kpos = torch.arange(pos[-1] - Tk + 1, pos[-1] + 1, device=x.device)
        diff = pos[:, None] - kpos[None, :]
        mask = ((diff >= 0) & (diff < c.swa_window))[None]
        o = _attend(q, k, v, mask, c.n_q_heads // c.n_kv_heads)
        o = o * self.gate(x).sigmoid()[..., None]
        keep = c.swa_window
        new_cache = {"k": k[:, -keep:].detach(), "v": v[:, -keep:].detach()}
        return self.o_proj(o.reshape(B, T, -1)), new_cache


class SharedKV(nn.Module):
    """YOCO global KV, computed once per token from the self-decoder output,
    plus the lightning-indexer keys/queries used for shared top-k selection."""

    def __init__(self, cfg: ModelConfig, rope: PartialRotary):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.rope = cfg, rope
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.kv_proj = nn.Linear(d, 2 * cfg.kv_dim, bias=False)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        if cfg.sparse_topk is not None:
            self.idx_k = nn.Linear(d, cfg.indexer_dim, bias=False)
            self.idx_q = nn.Linear(d, cfg.indexer_heads * cfg.indexer_dim, bias=False)
            self.idx_w = nn.Linear(d, cfg.indexer_heads, bias=False)

    def forward(self, h: torch.Tensor, pos: torch.Tensor):
        c = self.cfg
        B, T, _ = h.shape
        x = self.norm(h)
        k, v = self.kv_proj(x).view(B, T, 2, c.n_kv_heads, c.head_dim).unbind(2)
        k = self.rope(self.k_norm(k), pos)
        out = {"k": k, "v": v}
        if c.sparse_topk is not None:
            # As in DSA, the indexer is trained only by its distillation loss,
            # so its input is detached from the language-model graph.
            xi = x.detach()
            out["ik"] = self.idx_k(xi)  # [B, T, Di]
            out["iq"] = self.idx_q(xi).view(B, T, c.indexer_heads, c.indexer_dim)
            out["iw"] = self.idx_w(xi)  # [B, T, Hi]
        return out


def indexer_scores(iq, iw, ik):
    """Lightning-indexer relevance: sum_h w_h * relu(q_h . k_s).

    iq: [B, T, Hi, Di]; iw: [B, T, Hi]; ik: [B, Tk, Di] -> [B, T, Tk]
    """
    s = torch.einsum("bthd,bsd->bths", iq.float(), ik.float()).relu()
    return torch.einsum("bths,bth->bts", s, iw.float())


def topk_mask(scores: torch.Tensor, causal: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top-k causally visible keys per query (always keeps the
    diagonal so a query can attend to itself).

    Ties are common (relu indexer scores are often exactly 0) and must break
    identically in prefill, where a row also holds masked future keys, and in
    decode, where it does not. ``topk`` gives no such guarantee, so sort
    stably with the most recent key first; the kernel must use the same rule.
    """
    Tk = scores.shape[-1]
    if k >= Tk:
        return causal
    s = scores.masked_fill(~causal, float("-inf"))
    order = s.flip(-1).sort(dim=-1, descending=True, stable=True).indices[..., :k]
    idx = Tk - 1 - order
    sel = torch.zeros(s.shape, dtype=torch.bool, device=s.device).scatter(-1, idx, True)
    diag = torch.zeros_like(sel)
    diag[..., :, -scores.shape[-2]:] = torch.eye(scores.shape[-2], dtype=torch.bool, device=scores.device)
    return (sel | diag) & causal


class CrossAttention(nn.Module):
    """Cross-decoder attention: owns only a query projection and reads the
    single shared KV cache (YOCO)."""

    def __init__(self, cfg: ModelConfig, rope: PartialRotary):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.rope = cfg, rope
        self.q_proj = nn.Linear(d, cfg.q_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.gate = nn.Linear(d, cfg.n_q_heads, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, d, bias=False)

    def forward(self, x, pos, k, v, mask, return_probs: bool = False):
        c = self.cfg
        B, T, _ = x.shape
        q = self.rope(self.q_norm(self.q_proj(x).view(B, T, c.n_q_heads, c.head_dim)), pos)
        o = _attend(q, k, v, mask, c.n_q_heads // c.n_kv_heads)
        o = o * self.gate(x).sigmoid()[..., None]
        out = self.o_proj(o.reshape(B, T, -1))
        probs = None
        if return_probs:  # for the indexer distillation loss
            qh = q.transpose(1, 2).float()
            kh = _repeat_kv(k.transpose(1, 2), c.n_q_heads // c.n_kv_heads).float()
            logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(c.head_dim)
            logits = logits.masked_fill(~mask[:, None], float("-inf"))
            probs = logits.softmax(-1).sum(1)  # [B, T, Tk], summed over heads
        return out, probs


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, mixer: nn.Module):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = mixer
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)


class MTPModule(nn.Module):
    """FFN-only multi-token-prediction draft module, reused recursively.

    Input: the hidden state that predicted token t+j and the embedding of
    token t+j; output: a hidden state that (through the shared final norm +
    LM head) predicts token t+j+1.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_h = RMSNorm(d, cfg.norm_eps)
        self.norm_e = RMSNorm(d, cfg.norm_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat((self.norm_h(h), self.norm_e(e)), dim=-1))
        return x + self.ffn(self.ffn_norm(x))
