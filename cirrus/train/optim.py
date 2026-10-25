"""Optimizers and the learning-rate schedule for pretraining.

Muon (momentum + Newton-Schulz orthogonalisation) on the hidden weight
matrices, AdamW on everything else (tied embedding, norms, gates, conv
filters, GDN decay parameters). Muon's updates are rescaled to AdamW's RMS
(Moonlight, arXiv:2502.16982), so both share one learning rate and one weight
decay. On a compute-starved run this is the cheapest win available: Muon
reaches a given loss in roughly 2/3 of AdamW's tokens at this scale. The
first ablation checks that it holds for this architecture (``--optim adamw``).
"""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate U V^T of G = U S V^T with a quintic iteration in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 2e-3, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                g = g.lerp(buf, mom) if group["nesterov"] else buf
                u = newton_schulz(g, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.size(-2), p.size(-1)))  # match AdamW's update RMS
                p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr * scale)


def param_groups(model: nn.Module) -> tuple[list, list, list]:
    """(matrices for Muon, other weights with decay, no-decay params)."""
    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name:
            muon.append(p)
        elif p.ndim >= 2:  # tied embedding, conv filters
            decay.append(p)
        else:
            no_decay.append(p)
    return muon, decay, no_decay


def build_optimizers(model: nn.Module, kind: str, lr: float, weight_decay: float,
                     betas=(0.9, 0.95), fused: bool | None = None) -> list[torch.optim.Optimizer]:
    muon, decay, no_decay = param_groups(model)
    if fused is None:
        fused = next(model.parameters()).is_cuda
    if kind == "adamw":
        decay, muon = decay + muon, []
    elif kind != "muon":
        raise ValueError(kind)
    opts: list[torch.optim.Optimizer] = [torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8, fused=fused)]
    if muon:
        opts.append(Muon(muon, lr=lr, weight_decay=weight_decay))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = lr
    return opts


def wsd_lr(step: int, total: int, warmup: int, decay_frac: float, min_frac: float = 0.0) -> float:
    """Warmup-stable-decay multiplier. The decay is 1 - sqrt over the last ``decay_frac``
    of training (better than linear at the same length, arXiv:2405.18392), so any
    stable-phase checkpoint can be branched off and annealed later."""
    if step < warmup:
        return (step + 1) / warmup
    start = int(total * (1 - decay_frac))
    if step < start:
        return 1.0
    t = min(1.0, (step - start) / max(1, total - start))
    return min_frac + (1 - min_frac) * (1 - math.sqrt(t))


def decay_start(total: int, decay_frac: float) -> int:
    return int(total * (1 - decay_frac))
